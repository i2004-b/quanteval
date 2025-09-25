# scripts/train_cifar10.py
# Train a ResNet18 on CIFAR-10 from scratch using PyTorch.
# Achieves ~92% accuracy after 60 epochs with standard augmentations.
# Outputs model checkpoints and training history to models/resnet18_baseline.pth
import os, platform, time, random, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from multiprocessing import freeze_support

def seed_all(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    os.environ["PYTHONHASHSEED"]=str(s)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Limit CPU threads (helps Windows/CPU stability)
    torch.set_num_threads(min(8, os.cpu_count() or 1))

def make_loaders(batch_size=128, workers=0):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_ds = datasets.CIFAR10(root="./data", train=True,  download=True, transform=train_tf)
    val_ds   = datasets.CIFAR10(root="./data", train=False, download=True, transform=eval_tf)

    # Keep workers=0 on Windows. You can try >0 later after it’s stable.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False, persistent_workers=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=False, persistent_workers=False)
    return train_loader, val_loader

def accuracy_top1(logits, targets):
    return (logits.argmax(1) == targets).float().mean().item()

def train_one_epoch(model, loader, opt, device, label_smoothing=0.0):
    model.train()
    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
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
        total += images.size(0)
    return correct/total, total_loss/total

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)
    return correct/total

def save_ckpt(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def main():
    freeze_support()
    seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # —— knobs you can tweak ——
    epochs = 60          # try 60–100 for a strong baseline
    batch_size = 128
    base_lr = 0.1        # classic for SGD on CIFAR-10
    weight_decay = 5e-4
    label_smoothing = 0.1  # small smoothing improves generalization
    # ————————————————

    train_loader, val_loader = make_loaders(batch_size=batch_size)
    print("DEBUG workers:", getattr(train_loader, "num_workers", "unknown"))  # should be 0

    # ResNet18 head for 10 classes
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.to(device)

    opt = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    # Cosine LR schedule over the whole run
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val = 0.0
    history = []

    t_start = time.time()
    for epoch in range(1, epochs+1):
        t0 = time.time()
        tr_acc, tr_loss = train_one_epoch(model, train_loader, opt, device, label_smoothing)
        va_acc = evaluate(model, val_loader, device)
        sched.step()

        secs = time.time() - t0
        lr_now = sched.get_last_lr()[0]
        print(f"Epoch {epoch:03d}/{epochs}  "
              f"lr={lr_now:.4f}  train_acc={tr_acc*100:.2f}%  val_acc={va_acc*100:.2f}%  time={secs:.1f}s")

        history.append({"epoch": epoch, "lr": lr_now, "train_acc": tr_acc, "val_acc": va_acc})

        # Save last every epoch
        save_ckpt(model, "outputs/baselines/resnet18_cifar10_last.pt")
        # Save best when improved
        if va_acc > best_val:
            best_val = va_acc
            save_ckpt(model, "models/resnet18_baseline.pt")

    total_time = time.time() - t_start
    print(f"Done. Best val_acc = {best_val*100:.2f}%  total_time={total_time/60:.1f} min")

    # Save history to JSON for later plots
    with open("outputs/baselines/resnet18_cifar10_history.json", "w") as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    main()
