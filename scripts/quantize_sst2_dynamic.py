# scripts/quantize_sst2_dynamic.py
# Dynamic quantization for DistilBERT on GLUE/SST-2 with live progress.# Outputs match project style:
#   - models/distilbert_quantized_dynamic.pt
#   - outputs/reports/sst2_dynamic/history.json
#   - outputs/reports/sst2_dynamic/report.json


import os, time, json, math
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")  # <- skip torchvision import

from pathlib import Path
import torch
from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.utils.data import DataLoader
from tqdm import tqdm

def set_repro(seed=42):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.set_num_threads(min(8, os.cpu_count() or 1))

ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "models" / "distilbert_baseline"
REPORT_DIR   = ROOT / "outputs" / "reports" / "sst2_dynamic"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR   = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SAVE_WEIGHTS = MODELS_DIR / "distilbert_dynamic_int8.pt"
HISTORY_JSON = REPORT_DIR / "history.json"
REPORT_JSON  = REPORT_DIR / "report.json"

def collate(tokenizer):
    def _fn(batch):
        enc = tokenizer([ex["sentence"] for ex in batch], truncation=True, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([ex["label"] for ex in batch], dtype=torch.long)
        return enc
    return _fn

@torch.no_grad()
def evaluate(model, loader, device, print_every=50):
    model.eval().to(device)
    total = 0
    correct = 0
    t0 = time.time()
    running = []
    pbar = tqdm(total=len(loader), desc="Eval (INT8)", ncols=80)
    for i, batch in enumerate(loader, 1):
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**{k: batch[k] for k in ("input_ids","attention_mask")}).logits
        preds = logits.argmax(-1)
        correct += (preds == batch["labels"]).sum().item()
        total += batch["labels"].numel()
        acc = correct / max(total, 1)
        running.append({"step": i, "running_acc": acc, "seen": total, "elapsed_s": time.time()-t0})
        if i % print_every == 0:
            print(f"[eval] step {i:5d}/{len(loader)} | running_acc={acc*100:5.2f}% | seen={total}")
        pbar.update(1)
    pbar.close()
    with open(HISTORY_JSON, "w") as f:
        json.dump(running, f, indent=2)
    return correct / max(total, 1), time.time() - t0

def main():
    set_repro(42)
    device = torch.device("cpu")  # dynamic quant runs on CPU
    print(f"Device: {device}")

    # 1) Load baseline FP32 model + tokenizer
    if not BASELINE_DIR.exists():
        raise FileNotFoundError(f"Baseline not found at {BASELINE_DIR}. Run train_sst2.py first.")
    print("Loading tokenizer/model from:", BASELINE_DIR)
    tokenizer = DistilBertTokenizerFast.from_pretrained(str(BASELINE_DIR))
    fp32 = DistilBertForSequenceClassification.from_pretrained(str(BASELINE_DIR)).to(device).eval()

    # 2) Load SST-2 validation for evaluation
    print("Loading GLUE/SST-2 (validation)…")
    ds = load_dataset("glue", "sst2", split="validation")
    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate(tokenizer))

    # 3) Evaluate FP32 baseline
    print("Evaluating FP32 baseline…")
    fp32_acc, fp32_time = evaluate(fp32, loader, device, print_every=50)
    print(f"FP32  val_acc={fp32_acc*100:.2f}% | time={fp32_time:.2f}s")

    # 4) Dynamic quantization (Linear layers -> int8)
    print("Applying dynamic quantization (nn.Linear->int8)…")
    qmodel = torch.quantization.quantize_dynamic(
        fp32.cpu(), {torch.nn.Linear}, dtype=torch.qint8
    ).to(device).eval()

    # 5) Evaluate INT8
    print("Evaluating INT8 quantized model…")
    int8_acc, int8_time = evaluate(qmodel, loader, device, print_every=50)
    print(f"INT8  val_acc={int8_acc*100:.2f}% | time={int8_time:.2f}s")

    # 6) Save quantized weights
    torch.save(qmodel.state_dict(), SAVE_WEIGHTS)
    print(f"Saved INT8 state_dict -> {SAVE_WEIGHTS}")

    # 7) Report
    report = {
        "task": "sst2",
        "model": "distilbert-base-uncased",
        "method": "dynamic_int8",
        "paths": {
            "baseline_dir": str(BASELINE_DIR),
            "int8_state_dict": str(SAVE_WEIGHTS),
            "history_json": str(HISTORY_JSON),
            "report_json": str(REPORT_JSON),
        },
        "metrics": {
            "fp32": {"val_acc": round(float(fp32_acc), 4), "eval_time_s": round(float(fp32_time), 2)},
            "int8": {"val_acc": round(float(int8_acc), 4), "eval_time_s": round(float(int8_time), 2)},
        },
        "env": {
            "device": str(device),
            "torch": torch.__version__,
        },
    }
    with open(REPORT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print("Wrote report ->", REPORT_JSON)

if __name__ == "__main__":
    main()
