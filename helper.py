import torch
from transformers import DistilBertForSequenceClassification

model_dir = r"C:\Users\12697\SENIOR DESIGN\quanteval\models\distilbert_baseline"
out_pt    = r"C:\Users\12697\SENIOR DESIGN\quanteval\models\distilbert_baseline.pt"

print("Loading fine-tuned DistilBERT from:", model_dir)
model = DistilBertForSequenceClassification.from_pretrained(model_dir)

model.cpu()  # make sure weights are on CPU for portability
torch.save(model.state_dict(), out_pt)

print("Wrote:", out_pt)
