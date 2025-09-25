import torch
from sklearn.metrics import f1_score

@torch.no_grad()
def top1(model, loader, device):
    model.eval()
    total = correct = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
    return correct / total

@torch.no_grad()
def f1(model, loader, device, average="binary"):
    model.eval()
    all_preds, all_labels = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).argmax(1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(yb.cpu().numpy())
    return f1_score(all_labels, all_preds, average=average)
