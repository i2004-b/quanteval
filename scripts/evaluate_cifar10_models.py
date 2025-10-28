# scripts/evaluate_cifar10_models.py
# Compare CIFAR-10 models: FP32 baseline vs PTQ INT8 vs QAT INT8 (CPU latency/accuracy/size)

import os, time, json, platform
from pathlib import Path

import torch
import torch.nn as nn
import torch.ao.quantization as tq
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
torch.backends.quantized.engine = "fbgemm"  # x86 CPU quant backend


# ---------- paths ----------
ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
OUT = ROOT / "outputs" / "reports" / "cifar10_eval"
OUT.mkdir(parents=True, exist_ok=True)

# Expected files (use whatever exists)
BASELINE_CANDIDATES = [
    MODELS / "resnet18_baseline.pt",
    ROOT / "outputs/baselines/resnet18_cifar10_best.pt",
    ROOT / "outputs/baselines/resnet18_cifar10_last.pt",
]
PTQ_CANDIDATES = [
    MODELS / "resnet18_quantized_ptq.pt",
]
QAT_CANDIDATES = [
    MODELS / "resnet18_quantized_qat.pt",
]

def pick_first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None

BASELINE = pick_first_existing(BASELINE_CANDIDATES)
PTQ_INT8 = pick_first_existing(PTQ_CANDIDATES)
QAT_INT8 = pick_first_existing(QAT_CANDIDATES)

# ---------- data ----------
def make_loader(batch_size=128):
    mean=(0.4914,0.4822,0.4465); std=(0.2470,0.2435,0.2616)
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean,std),
    ])
    test_ds = datasets.CIFAR10(str(ROOT / "data"), train=False, download=True, transform=tfm)
    workers = 0 if platform.system()=="Windows" else 2
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                      num_workers=workers, pin_memory=False, persistent_workers=False)

@torch.no_grad()
def eval_top1_latency(model, loader, warmup=20, limit=None, device=torch.device("cpu")):
    model.eval().to(device)
    total = correct = 0
    n_seen = 0

    # warmup (don’t time)
    it = iter(loader)
    for _ in range(min(warmup, len(loader))):
        try:
            x, y = next(it)
            _ = model(x.to(device))
        except StopIteration:
            break

    # timed pass
    start = time.time()
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        logits = model(x)
        pred = logits.argmax(1)
        correct += (pred==y).sum().item()
        total += y.numel()
        n_seen += y.numel()
        if limit is not None and n_seen >= limit:
            break
    elapsed = time.time() - start
    acc = correct/total if total else 0.0
    ms_per_image = (elapsed / (n_seen if n_seen else 1)) * 1000.0
    return acc, ms_per_image

# ---------- quant-safe QAT model (matches your QAT script) ----------
class QuantBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.skip_add = nn.quantized.FloatFunctional()
    def fuse_model(self):
        tq.fuse_modules(self, ["conv1","bn1","relu"], inplace=True)
        tq.fuse_modules(self, ["conv2","bn2"], inplace=True)
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.skip_add.add(out, identity)
        return self.relu(out)

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
        self.conv1 = nn.Conv2d(3, inplanes, 7, stride=2, padding=3, bias=False)
        self.bn1 = norm(inplanes); self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1, inplanes = _make_layer(inplanes, 64,  2, stride=1, norm_layer=norm)
        self.layer2, inplanes = _make_layer(inplanes, 128, 2, stride=2, norm_layer=norm)
        self.layer3, inplanes = _make_layer(inplanes, 256, 2, stride=2, norm_layer=norm)
        self.layer4, inplanes = _make_layer(inplanes, 512, 2, stride=2, norm_layer=norm)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*QuantBasicBlock.expansion, num_classes)
        self.quant = tq.QuantStub(); self.dequant = tq.DeQuantStub()
    def fuse_model(self):
        tq.fuse_modules(self, ["conv1","bn1","relu"], inplace=True)
        for layer_name in ["layer1","layer2","layer3","layer4"]:
            for b in getattr(self, layer_name):
                b.fuse_model()
    def forward(self, x):
        x = self.quant(x)
        x = self.relu(self.bn1(self.conv1(x))); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x); x = torch.flatten(x, 1); x = self.fc(x)
        x = self.dequant(x)
        return x

# ---------- helpers to (re)build models ----------
def build_baseline():
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 10)
    return m

def build_ptq_quantized_model():
    # Build a quantizable ResNet18 shape, convert to INT8 modules, then load your saved INT8 weights.
    qmodel = models.quantization.resnet18(weights=None, num_classes=10, quantize=False)
    qmodel.eval()                       # fuse requires eval mode
    qmodel.fuse_model()
    qmodel.qconfig = tq.get_default_qconfig("fbgemm")
    tq.prepare(qmodel, inplace=True)    # (Observer warning is fine; we load real INT8 params next.)
    # Optional: touch observers once to silence the warning:
    # with torch.no_grad():
    #     qmodel(torch.randn(1,3,224,224))
    qmodel = tq.convert(qmodel, inplace=False)
    qmodel.eval()
    return qmodel


def build_qat_quantized_model():
    m = QuantizableResNet18(num_classes=10)
    m.eval()
    m.fuse_model()                      # fuse in eval
    m.train()                           # QAT prepare requires TRAIN mode
    m.qconfig = tq.get_default_qat_qconfig("fbgemm")   # <<< IMPORTANT
    tq.prepare_qat(m, inplace=True)
    m = tq.convert(m, inplace=False)    # now swap in quantized ops so the INT8 state_dict matches
    m.eval()
    return m



# ---------- main ----------
def main():
    if BASELINE is None:
        raise FileNotFoundError("Baseline model not found in models/ or outputs/baselines/. Train it first.")

    print("Models:")
    print("  FP32 baseline:", BASELINE)
    print("  PTQ INT8     :", PTQ_INT8 if PTQ_INT8 else "(missing)")
    print("  QAT INT8     :", QAT_INT8 if QAT_INT8 else "(missing)")

    loader = make_loader(batch_size=256)
    device = torch.device("cpu")  # compare on CPU fairly

    results = []

    # --- Baseline FP32 ---
    baseline = build_baseline()
    baseline.load_state_dict(torch.load(BASELINE, map_location="cpu"))
    size_mb = os.path.getsize(BASELINE)/1e6
    acc, ms = eval_top1_latency(baseline, loader, warmup=10, device=device)
    results.append({"name":"fp32_baseline","acc":round(acc*100,2),"ms_per_img":round(ms,3),"size_mb":round(size_mb,2)})

    # --- PTQ INT8 (if present) ---
    if PTQ_INT8 is not None:
        ptq = build_ptq_quantized_model()
        # load INT8 state dict (already converted)
        state = torch.load(PTQ_INT8, map_location="cpu")
        missing, unexpected = ptq.load_state_dict(state, strict=False)
        if missing:   print("[ptq warn] missing keys:", missing)
        if unexpected:print("[ptq warn] unexpected keys:", unexpected)
        size_mb = os.path.getsize(PTQ_INT8)/1e6
        try:
            acc, ms = eval_top1_latency(ptq, loader, warmup=10, device=device)
            results.append({"name":"ptq_int8","acc":round(acc*100,2),"ms_per_img":round(ms,3),"size_mb":round(size_mb,2)})
        except NotImplementedError:
            print("[ptq warn] backend missing some quantized op; skipping accuracy. Reporting size only.")
            results.append({"name":"ptq_int8","acc":None,"ms_per_img":None,"size_mb":round(size_mb,2)})
    else:
        print("[info] PTQ model not found; skipping.")

    # --- QAT INT8 (if present) ---
    if QAT_INT8 is not None:
        qat = build_qat_quantized_model()
        state = torch.load(QAT_INT8, map_location="cpu")
        missing, unexpected = qat.load_state_dict(state, strict=False)
        if missing:   print("[qat warn] missing keys:", missing)
        if unexpected:print("[qat warn] unexpected keys:", unexpected)
        size_mb = os.path.getsize(QAT_INT8)/1e6
        try:
            acc, ms = eval_top1_latency(qat, loader, warmup=10, device=device)
            results.append({"name":"qat_int8","acc":round(acc*100,2),"ms_per_img":round(ms,3),"size_mb":round(size_mb,2)})
        except NotImplementedError:
            print("[qat warn] backend missing some quantized op; skipping accuracy. Reporting size only.")
            results.append({"name":"qat_int8","acc":None,"ms_per_img":None,"size_mb":round(size_mb,2)})
    else:
        print("[info] QAT model not found; skipping.")

    # print table
    print("\n=== CIFAR-10 Evaluation (CPU) ===")
    for r in results:
        print(f"{r['name']:>12}  acc={str(r['acc']).rjust(6)}  ms/img={str(r['ms_per_img']).rjust(7)}  sizeMB={r['size_mb']:>6}")

    with open(OUT / "cifar10_eval.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Wrote ->", OUT / "cifar10_eval.json")

if __name__ == "__main__":
    main()
