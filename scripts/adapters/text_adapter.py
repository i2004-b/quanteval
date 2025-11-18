import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_hf_model(model_dir_or_name: str):
    tok = AutoTokenizer.from_pretrained(model_dir_or_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_dir_or_name)
    mdl.eval()
    return tok, mdl

