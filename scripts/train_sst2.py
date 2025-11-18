# scripts/train_sst2.py
# Baseline FP32 fine-tuning of DistilBERT on GLUE/SST-2
# - Works with Transformers v4 and v5 (evaluation_strategy vs eval_strategy)
# - Saves best model + tokenizer to models/distilbert_baseline/
# - Logs per-epoch metrics to outputs/reports/sst2_baseline/history.json
# - Writes a final summary to outputs/reports/sst2_baseline/report.json

import os, json, time, random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
import transformers
from packaging import version


# -------------------- Repro --------------------
def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(min(8, os.cpu_count() or 1))


# -------------------- Paths --------------------
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models" / "distilbert_baseline"
REPORT_DIR = ROOT / "outputs" / "reports" / "sst2_baseline"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------- Metrics --------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean().item()
    return {"accuracy": float(acc)}


# -------------------- Per-epoch JSON logger --------------------
class JsonHistoryLogger(transformers.TrainerCallback):
    def __init__(self, out_path: Path):
        self.out_path = out_path
        self.history = []

    def on_epoch_end(self, args, state, control, **kwargs):
        rec = {
            "epoch": int(round(state.epoch or 0)),
            "global_step": state.global_step,
            "train_loss": _last_value(state.log_history, "loss"),
            "eval_accuracy": _last_value(state.log_history, "eval_accuracy"),
            "eval_loss": _last_value(state.log_history, "eval_loss"),
            "learning_rate": _last_value(state.log_history, "learning_rate"),
        }
        self.history.append(rec)
        with open(self.out_path, "w") as f:
            json.dump(self.history, f, indent=2)


def _last_value(log_history, key):
    if not log_history:
        return None
    for item in reversed(log_history):
        if key in item:
            v = item[key]
            if isinstance(v, (np.floating, np.integer)):
                return float(v)
            return v
    return None


# -------------------- Main --------------------
def main():
    seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) Load GLUE/SST-2
    ds = load_dataset("glue", "sst2")

    # 2) Tokenizer & tokenize (KEEP labels and rename to 'labels')
    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

    def preprocess(batch):
        enc = tokenizer(
            batch["sentence"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )
        # HF models expect 'labels'
        enc["labels"] = batch["label"]
        return enc

    # Remove only non-needed columns; keep nothing but model inputs + 'labels'
    remove_cols = [c for c in ds["train"].column_names if c not in ["sentence", "label"]]
    # First map: create tokenized inputs + labels
    ds_enc = ds.map(preprocess, batched=True, remove_columns=remove_cols)
    # Second map: now drop the original 'sentence' and 'label' (we have 'labels' already)
    # Keep input_ids, attention_mask, labels only
    ds_enc = ds_enc.remove_columns(
        [c for c in ds_enc["train"].column_names if c not in ["input_ids", "attention_mask", "labels"]]
    )

    # 3) Model
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    # 4) Training args (v4/v5 compatible)
    EVAL_KEY = "eval_strategy"
    args_kwargs = {
        "output_dir": str(ROOT / "outputs" / "sst2_baseline"),
        "per_device_train_batch_size": 32,
        "per_device_eval_batch_size": 64,
        "num_train_epochs": 3,
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "logging_steps": 100,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "accuracy",
        "greater_is_better": True,
        "report_to": [],  # no wandb/tensorboard
    }
    args_kwargs[EVAL_KEY] = "epoch"

    training_args = TrainingArguments(**args_kwargs)

    # 5) Data collator (dynamic padding) & Trainer
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_enc["train"],
        eval_dataset=ds_enc["validation"],
        tokenizer=tokenizer,  # v5 warns this is deprecated; harmless
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[JsonHistoryLogger(REPORT_DIR / "history.json")],
    )

    # 6) Train
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    # 7) Evaluate best model on validation
    eval_metrics = trainer.evaluate(ds_enc["validation"])
    val_acc = float(eval_metrics.get("eval_accuracy", 0.0))
    val_loss = float(eval_metrics.get("eval_loss", 0.0))

    # 8) Save best model + tokenizer
    print("Saving best model to:", MODELS_DIR)
    trainer.save_model(str(MODELS_DIR))    # saves model + config
    tokenizer.save_pretrained(str(MODELS_DIR))

    # 9) Final report JSON
    report = {
        "task": "sst2",
        "model": model_name,
        "method": "fp32_baseline",
        "epochs": training_args.num_train_epochs,
        "base_lr": training_args.learning_rate,
        "batch_size_train": training_args.per_device_train_batch_size,
        "batch_size_eval": training_args.per_device_eval_batch_size,
        "weight_decay": training_args.weight_decay,
        "device_train": str(device),
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "val_accuracy": round(val_acc, 4),
        "val_loss": round(val_loss, 4),
        "model_dir": str(MODELS_DIR),
        "history_json": str(REPORT_DIR / "history.json"),
        "total_train_time_s": round(train_time, 2),
    }
    with open(REPORT_DIR / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Wrote report ->", REPORT_DIR / "report.json")


if __name__ == "__main__":
    main()
