import torch
from sklearn.metrics import f1_score

@torch.no_grad()
def top1(model, loader, device):
    model.eval()
    model.to(device)
    correct, total = 0, 0

    for batch in loader:
        # Handle dataset batch structure
        if isinstance(batch, (list, tuple)):
            xb, yb = batch
            xb, yb = xb.to(device), yb.to(device)
        elif isinstance(batch, dict):
            # Text datasets usually return dicts with input_ids, attention_mask, labels, etc.
            xb = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            yb = batch["labels"].to(device)
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

        # Forward pass
        outputs = model(xb)

        # Extract logits (some models return tuples)
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]

        preds = torch.argmax(outputs, dim=-1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

    return correct / total if total > 0 else 0

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
