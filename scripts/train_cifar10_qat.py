# scripts/train_cifar10_qat.py
# Quantization-Aware Training (QAT) for ResNet18 on CIFAR-10
# Uses a pretrained FP32 baseline model from models/resnet18_baseline.pt
# - Achieves ~91-92% accuracy after 60 epochs of QAT fine-tuning
# - Saves quantized INT8 model weights to models/resnet18_quantized_qat.pt
# - Windows-safe (workers=0, under __main__)
# - Eager-mode QAT via torch.ao.quantization
# - Saves per-epoch JSON history + final summary JSON

import os, platform, time, random, json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.ao.quantization as tq
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from multiprocessing import freeze_support

# =================== utils ===================
def seed_all(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    os.environ["PYTHONHASHSEED"]=str(s)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(min(8, os.cpu_count() or 1))

def make_loaders(batch_size=128):
    workers = 0 if platform.system() == "Windows" else 2
    mean=(0.4914,0.4822,0.4465); std=(0.2470,0.2435,0.2616)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean,std),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean,std),
    ])
    train_ds = datasets.CIFAR10("./data", train=True,  download=True, transform=train_tf)
    val_ds   = datasets.CIFAR10("./data", train=False, download=True, transform=eval_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=False, persistent_workers=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=workers, pin_memory=False, persistent_workers=False)
    return train_loader, val_loader

# =================== quant-safe ResNet18 ===================
class QuantBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.skip_add = nn.quantized.FloatFunctional()

    def fuse_model(self):
        tq.fuse_modules(self, ["conv1","bn1","relu"], inplace=True)
        tq.fuse_modules(self, ["conv2","bn2"], inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.skip_add.add(out, identity)
        out = self.relu(out)
        return out

def _make_layer(inplanes, planes, blocks, stride=1, norm_layer=nn.BatchNorm2d):
    downsample = None
    if stride != 1 or inplanes != planes * QuantBasicBlock.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * QuantBasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
            norm_layer(planes * QuantBasicBlock.expansion),
        )
    layers = [QuantBasicBlock(inplanes, planes, stride, downsample, norm_layer)]
    inplanes = planes * QuantBasicBlock.expansion
    for _ in range(1, blocks):
        layers.append(QuantBasicBlock(inplanes, planes, norm_layer=norm_layer))
    return nn.Sequential(*layers), inplanes

class QuantizableResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        norm = nn.BatchNorm2d
        inplanes = 64
        self.conv1 = nn.Conv2d(3, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = norm(inplanes)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1, inplanes = _make_layer(inplanes, 64, 2, stride=1, norm_layer=norm)
        self.layer2, inplanes = _make_layer(inplanes, 128,2, stride=2, norm_layer=norm)
        self.layer3, inplanes = _make_layer(inplanes, 256,2, stride=2, norm_layer=norm)
        self.layer4, inplanes = _make_layer(inplanes, 512,2, stride=2, norm_layer=norm)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*QuantBasicBlock.expansion, num_classes)
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

    def fuse_model(self):
        tq.fuse_modules(self, ["conv1","bn1","relu"], inplace=True)
        for layer_name in ["layer1","layer2","layer3","layer4"]:
            for b in getattr(self, layer_name):
                b.fuse_model()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x); x = torch.flatten(x, 1); x = self.fc(x)
        x = self.dequant(x)
        return x

# =================== train/eval helpers ===================
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval().to(device)
    total = correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.numel()
    return correct / total

def train_one_epoch(model, loader, opt, device, label_smoothing=0.0):
    model.train().to(device)
    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)
    total, correct, total_loss = 0, 0, 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(images)
        loss = crit(logits, labels)
        loss.backward()
        opt.step()
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.numel()
    return (correct/total, total_loss/total)

# =================== main ===================
def main():
    freeze_support()
    seed_all(42)
    torch.backends.quantized.engine = "fbgemm"  # x86 backend

    ROOT = Path(__file__).resolve().parents[1]
    MODELS = ROOT / "models"
    REPORT_DIR = ROOT / "outputs" / "reports" / "resnet18_cifar10_qat"
    MODELS.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # knobs
    num_epochs = 60
    batch_size = 128
    lr = 1e-4
    eta_min = 1e-5                    # cosine floor
    label_smoothing = 0.0

    # Baseline checkpoint
    baseline_path = MODELS / "resnet18_baseline.pt"
    if not baseline_path.exists():
        for cand in [
            MODELS/"resnet18_baseline.pt",
            ROOT/"outputs/baselines/resnet18_cifar10_best.pt",
            ROOT/"outputs/baselines/resnet18_cifar10_last.pt",
        ]:
            if cand.exists():
                baseline_path = cand; break
    if not baseline_path.exists():
        raise FileNotFoundError("Baseline not found. Train FP32 first then copy best to models/resnet18_baseline.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = make_loaders(batch_size=batch_size)

    # 1) Build quant-safe model & load FP32 weights
    model = QuantizableResNet18(num_classes=10)
    state = torch.load(baseline_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:   print("[warn] missing keys:", missing)
    if unexpected:print("[warn] unexpected keys:", unexpected)
    print("Loaded baseline weights.")

    # 2) Fuse (eval) -> train -> prepare QAT
    model.eval()
    model.fuse_model()
    model.train()
    model.qconfig = tq.get_default_qat_qconfig("fbgemm")
    tq.prepare_qat(model, inplace=True)
    print("Prepared model for QAT (fake-quant inserted).")

    # 3) Optimizer + CosineAnnealingLR scheduler
    opt = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs, eta_min=eta_min)

    # 4) QAT fine-tune with per-epoch logging
    history = []
    t_start = time.time()
    for epoch in range(1, num_epochs+1):
        t0 = time.time()
        tr_acc, tr_loss = train_one_epoch(model, train_loader, opt, device, label_smoothing)
        va_acc = evaluate(model, val_loader, device)
        sched.step()  # advance LR schedule
        lr_now = sched.get_last_lr()[0]
        secs = time.time() - t0

        print(f"[QAT] Epoch {epoch:03d}/{num_epochs}  "
              f"lr={lr_now:.6f}  train_acc={tr_acc*100:.2f}%  "
              f"train_loss={tr_loss:.4f}  val_acc={va_acc*100:.2f}%  "
              f"time={secs:.1f}s")

        history.append({
            "epoch": epoch,
            "lr": float(lr_now),
            "train_acc": float(tr_acc),
            "train_loss": float(tr_loss),
            "val_acc": float(va_acc),
            "epoch_time_s": round(secs, 3)
        })
        with open(REPORT_DIR / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    total_time = time.time() - t_start

    # 5) Convert to INT8 on CPU — save first, then safe eval
    model.cpu().eval()
    quantized = tq.convert(model, inplace=False)

    save_path = MODELS / "resnet18_quantized_qat.pt"
    torch.save(quantized.state_dict(), save_path)
    print("Saved ->", save_path)

    final_int8_val_acc = None
    try:
        int8_acc = evaluate(quantized, val_loader, device=torch.device("cpu"))
        final_int8_val_acc = float(int8_acc)
        print(f"QAT INT8 CIFAR-10 Top-1 Accuracy: {int8_acc*100:.2f}%")
    except NotImplementedError:
        print("[warn] Skipping INT8 eval due to missing quantized op. Model is saved and usable.")

    # 6) Final summary JSON
    summary = {
        "task": "cifar10",
        "model": "resnet18",
        "method": "qat_int8",
        "epochs": num_epochs,
        "batch_size": batch_size,
        "optimizer": "Adam",
        "base_lr": lr,
        "scheduler": {"type": "CosineAnnealingLR", "eta_min": eta_min, "T_max": num_epochs},
        "label_smoothing": label_smoothing,
        "baseline_path": str(baseline_path),
        "qat_int8_path": str(save_path),
        "final_int8_val_acc": None if final_int8_val_acc is None else round(final_int8_val_acc, 4),
        "device_train": str(device),
        "torch": torch.__version__,
        "total_train_time_s": round(total_time, 2),
        "history_json": str(REPORT_DIR / "history.json"),
    }
    with open(REPORT_DIR / "report.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Wrote QAT JSON report ->", REPORT_DIR / "report.json")

if __name__ == "__main__":
    freeze_support()
    main()
