import torch
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
