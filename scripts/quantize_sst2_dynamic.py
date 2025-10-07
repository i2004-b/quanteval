# scripts/quantize_sst2_dynamic.py
# Dynamic Quantization of a fine-tuned DistilBERT model on SST-2 using PyTorch.
# Loads a fine-tuned FP32 model from models/distilbert_baseline
# Saves quantized INT8 model weights to models/distilbert_quantized_dynamic.pth
import torch
from transformers import DistilBertForSequenceClassification

import time, os

# 1. Load the fine-tuned DistilBERT base model (FP32)
model_dir = "../models/distilbert_baseline"
model = DistilBertForSequenceClassification.from_pretrained(model_dir)
model.eval()
print("Loaded DistilBERT baseline model (FP32).")

# 2. Apply dynamic quantization on linear layers to get an INT8 model
quantized_model = torch.quantization.quantize_dynamic(
    model, 
    {torch.nn.Linear},  # specify which modules to quantize (Linear layers to int8)
    dtype=torch.qint8
)
print("Applied dynamic quantization to DistilBERT (INT8 weights for Linear layers).")

# 3. Evaluate the quantized model on SST-2 validation set
# Load SST-2 validation data and tokenizer
from datasets import load_dataset
from transformers import DistilBertTokenizerFast
datasets = load_dataset('glue', 'sst2')
val_dataset = datasets['validation']
tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)

# Tokenize the validation set
val_enc = val_dataset.map(lambda examples: tokenizer(examples['sentence'], truncation=True, padding=True, max_length=128), batched=True)
val_enc.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Create DataLoader for evaluation
from torch.utils.data import DataLoader
val_loader = DataLoader(val_enc, batch_size=32, shuffle=False)

# Measure accuracy and inference time
correct = 0
total = 0
start_time = time.time()
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        # Ensure model is on CPU (dynamic quant is CPU-only by default)
        outputs = quantized_model(input_ids, attention_mask=attention_mask)
        # outputs.logits is the classification head output
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        total += labels.size(0)
        correct += (predictions == labels).sum().item()
elapsed = time.time() - start_time
accuracy = correct / total
print(f"Dynamic INT8 DistilBERT Accuracy: {accuracy*100:.2f}% (vs FP32 baseline).")
print(f"Inference time on CPU for {total} sentences: {elapsed:.2f} seconds.")

# 4. Save the quantized model weights (state dict)
save_path = "../models/distilbert_quantized_dynamic.pth"
torch.save(quantized_model.state_dict(), save_path)
print(f"Saved dynamic quantized model state to {save_path}")

# 5. Report model size reduction
fp32_model_path = os.path.join(model_dir, "pytorch_model.bin")
if os.path.exists(fp32_model_path):
    fp32_size = os.path.getsize(fp32_model_path) / 1e6
    int8_size = os.path.getsize(save_path) / 1e6
    print(f"FP32 model size: {fp32_size:.2f} MB, INT8 model size: {int8_size:.2f} MB")
else:
    # If for some reason the model was saved differently, compute in-memory size
    torch.save(model.state_dict(), "temp_fp32.pth")
    torch.save(quantized_model.state_dict(), "temp_int8.pth")
    fp32_size = os.path.getsize("temp_fp32.pth")/1e6
    int8_size = os.path.getsize("temp_int8.pth")/1e6
    os.remove("temp_fp32.pth"); os.remove("temp_int8.pth")
    print(f"FP32 model size (state_dict): {fp32_size:.2f} MB, INT8 model size: {int8_size:.2f} MB")
