# scripts/train_cifar10.py
import os, platform, time, random
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

    # Force num_workers=0 explicitly to avoid Windows spawn issues.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False, persistent_workers=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=False, persistent_workers=False)
    return train_loader, val_loader

def train_one_epoch(model, loader, opt, device):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total, correct, total_loss = 0, 0, 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
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

def main():
    print(f"RUNNING FILE: {__file__}")  # helps confirm you’re running the right script
    seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = make_loaders(batch_size=128)
    print("DEBUG workers:", getattr(train_loader, "num_workers", "unknown"))  # should be 0

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.to(device)

    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    epochs = 1  # keep tiny for smoke test
    for epoch in range(1, epochs+1):
        t0 = time.time()
        tr_acc, tr_loss = train_one_epoch(model, train_loader, opt, device)
        va_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:02d}  train_acc={tr_acc:.3f}  val_acc={va_acc:.3f}  time={time.time()-t0:.1f}s")

    os.makedirs("outputs/baselines", exist_ok=True)
    torch.save(model.state_dict(), "outputs/baselines/resnet18_cifar10.pt")
    print("Saved -> outputs/baselines/resnet18_cifar10.pt")

if __name__ == "__main__":
    # Required on Windows when using anything that could spawn processes (like DataLoader workers)
    freeze_support()
    main()
