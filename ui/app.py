# ui/app.py
# Run: streamlit run ui/app.py
import streamlit as st
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os, sys
from torchvision.models import resnet18
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import pandas as pd
from datasets import load_dataset
from model_loader import load_model

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from eval.metrics import top1, f1
from eval.latency import measure_latency_s
from eval.memory import param_bytes, peak_gpu_mem_once
from eval.report import log_experiment

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================================
# MODEL REGISTRY — built-in models and quantized variants
# ==========================================================
MODEL_REGISTRY = {
    "ResNet18": {
        "Baseline": "models/resnet18_baseline.pt",
        "PTQ": "models/resnet18_quantized_ptq.pt",
        "QAT": "models/resnet18_quantized_qat.pt",
    },
    "DistilBERT": {
        "Baseline": "models/distilbert_baseline/",
        #"PTQ": "models/distilbert_ptq.pt",
        #"QAT": "models/distilbert_qat.pt",
        "int8": "models/distilbert_dynamic_int8.pt",
    },
}

# ==========================================================
# STREAMLIT PAGE SETUP
# ==========================================================
st.set_page_config(page_title="Quanteval — Model Evaluation", layout="wide")
st.title("🧠 Quanteval — Model Evaluation UI")

st.sidebar.header("Evaluation Options")

# ==========================================================
# MODEL SELECTION / UPLOAD SECTION
# ==========================================================
model_source = st.sidebar.radio(
    "Choose model source:",
    ["Built-in Model", "Upload Custom Model"]
)

if model_source == "Built-in Model":
    model_type = st.sidebar.selectbox("Model Architecture", list(MODEL_REGISTRY.keys()))
    variant = st.sidebar.selectbox("Variant", list(MODEL_REGISTRY[model_type].keys()))
    model_path = MODEL_REGISTRY[model_type][variant]
else:
    st.sidebar.write("Upload a `.pt` or `.pth` model file (max 2 GB):")
    uploaded_file = st.sidebar.file_uploader("Upload model file", type=["pt", "pth"])
    model_type = st.sidebar.selectbox("Architecture Type", ["ResNet18", "DistilBERT"])
    variant = "User Upload"
    model_path = None
    if uploaded_file:
        temp_path = os.path.join("uploads", uploaded_file.name)
        os.makedirs("uploads", exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        model_path = temp_path

device_choice = st.sidebar.selectbox("Run on", ["cpu", "cuda (if available)"])
device = torch.device("cuda" if (device_choice.startswith("cuda") and torch.cuda.is_available()) else "cpu")

eval_samples = st.sidebar.number_input("Num eval samples (subset)", min_value=32, max_value=5000, value=512, step=32)
latency_runs = st.sidebar.number_input("Latency runs", min_value=5, max_value=200, value=20)
latency_warmup = st.sidebar.number_input("Latency warmup", min_value=1, max_value=50, value=5)

# ==========================================================
# MODEL LOADING
# ==========================================================
def load_selected_model(model_name, device, user_model_path=None):
    try:
        model = load_model(model_name, device, user_path=user_model_path)
        return model, None
    except Exception as e:
        return None, str(e)

# ==========================================================
# EVALUATION PIPELINES
# ==========================================================
def evaluate_cifar10_model(model, eval_samples, latency_runs, latency_warmup, device):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470,0.2435,0.2616))
    ])
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
    subset = torch.utils.data.Subset(testset, list(range(min(len(testset), eval_samples))))
    loader = DataLoader(subset, batch_size=128, shuffle=False)

    acc = top1(model, loader, device=device)

    # ✅ Fix here — get only images, not (image, label)
    example_batch = next(iter(DataLoader(subset, batch_size=1)))
    if isinstance(example_batch, (tuple, list)):
        example_input = example_batch[0]  # take the image only
    else:
        example_input = example_batch

    # Add batch dimension if needed
    if example_input.ndim == 3:
        example_input = example_input.unsqueeze(0).to(device)
    else:
        example_input = example_input.to(device)

    latency_s = measure_latency_s(
        model, example_input,
        runs=int(latency_runs),
        warmup=int(latency_warmup),
        device=str(device.type)
    )

    peak_mem = peak_gpu_mem_once(model, example_input) if device.type == "cuda" else 0
    param_mb = param_bytes(model) / 1e6

    return {
        "Accuracy": float(acc),
        "Latency (s)": float(latency_s),
        "param_MB": float(param_mb),
        "peak memory on GPU (mb)": float(peak_mem)/1e6 if peak_mem else 0.0
    }

def evaluate_sst2_model(model, eval_samples, latency_runs, latency_warmup, device):
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    dataset = load_dataset("glue", "sst2", split="validation")

    def tok(batch):
        return tokenizer(batch["sentence"], truncation=True, padding="max_length", max_length=128)
    ds = dataset.map(tok, batched=True)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    ds = ds.select(range(min(len(ds), eval_samples)))
    loader = DataLoader(ds, batch_size=16)

    correct, total = 0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=mask)
            preds = torch.argmax(out.logits, dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    acc = correct / total

    example_batch = next(iter(DataLoader(ds, batch_size=1)))
    example_input = {"input_ids": example_batch["input_ids"].to(device),
                     "attention_mask": example_batch["attention_mask"].to(device)}

    latency_s = measure_latency_s(model, example_input, runs=int(latency_runs),
                                  warmup=int(latency_warmup), device=str(device.type))
    param_mb = param_bytes(model) / 1e6

    return {"accuracy": float(acc), "latency_s": float(latency_s),
            "param_MB": float(param_mb), "peak_mem_MB": 0.0}

# ==========================================================
# RUN EVALUATION
# ==========================================================
if st.button("Run Evaluation"):
    st.info(f"Loading {model_type} ({variant}) on {device.type.upper()}...")
    try:
        model = load_selected_model(model_type, variant, model_path, device)
        st.success("Model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    st.info("Running evaluation...")
    if model_type == "ResNet18":
        metrics = evaluate_cifar10_model(model, eval_samples, latency_runs, latency_warmup, device)
    elif model_type == "DistilBERT":
        metrics = evaluate_sst2_model(model, eval_samples, latency_runs, latency_warmup, device)
    else:
        st.error("Unsupported model type.")
        st.stop()

    # Display results
    st.subheader("📊 Results")
    for k, v in metrics.items():
        st.metric(k, round(v, 6) if isinstance(v, float) else v)
    st.bar_chart(pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"]))
    st.json(metrics)

    # Save report
    rec = log_experiment(metrics, model_name=f"{model_type}-{variant}", method="ui-eval", out_dir="outputs/reports")
    st.success("Logged experiment to outputs/reports")
    st.write(rec)

# ==========================================================
# FOOTER
# ==========================================================
if __name__ == "__main__":
    st.write("Run with:  streamlit run ui/app.py")
