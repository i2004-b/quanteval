#user interface app

# ui/app.py
import streamlit as st
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import sys, os
from torchvision.models import resnet18
from transformers import DistilBertForSequenceClassification
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import utilities 
from eval.metrics import top1, f1 #C:\Pytorch\quanteval\eval\memory.py
from eval.latency import measure_latency_s
from eval.memory import param_bytes, on_disk_bytes_state_dict, peak_gpu_mem_once
from eval.report import log_experiment, write_json, write_csv_row, add_timestamp, read_results

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CIFAR_PATH = "outputs/baselines/resnet18_cifar10.pt"
SST2_PATH = "outputs/baselines/distilbert_sst2"

st.set_page_config(page_title="Quanteval Demo", layout="wide")

st.title("Quanteval — Model Evaluation UI")

# --- Sidebar: options ---
st.sidebar.header("Evaluation options")
model_type = st.sidebar.selectbox("Model architecture", ["resnet18-cifar10", "distilbert-sst2"])
device_choice = st.sidebar.selectbox("Run on", ["cpu", "cuda (if available)"])
device = torch.device("cuda" if (device_choice.startswith("cuda") and torch.cuda.is_available()) else "cpu")

eval_samples = st.sidebar.number_input("Num eval samples (subset)", min_value=32, max_value=10000, value=1024, step=32)
latency_runs = st.sidebar.number_input("Latency runs", min_value=5, max_value=200, value=30)
latency_warmup = st.sidebar.number_input("Latency warmup", min_value=1, max_value=50, value=5)

# --- Upload / select model checkpoint ---
st.header("Load / Select Model")
uploaded = st.file_uploader("Upload model checkpoint (.pth state_dict recommended)", type=["pth","pt","bin","zip"])
use_example = st.checkbox("Use example (pretrained) instead of upload", value=True)

model = None
model_loaded = False
checkpoint_path = None

if uploaded is not None:
    # save uploaded file temporarily so torch can load it
    temp_path = os.path.join("ui", "uploaded_model.pth")
    with open(temp_path, "wb") as f:
        f.write(uploaded.getbuffer())
    checkpoint_path = temp_path
    st.success("Uploaded checkpoint saved.")
    use_example = False

if use_example:
    st.info("Using a local example/pretrained model (no upload).")

# Function to build model objects
def build_resnet18_for_cifar():
    m = torchvision.models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 10)
    return m

def build_distilbert_for_sst2():
    from transformers import DistilBertForSequenceClassification
    return DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Button to (attempt to) load the model
if st.button("Load model"):
    try:
        if model_type.startswith("resnet"):
            model = build_resnet18_for_cifar()
            if checkpoint_path:
                # prefer state_dict
                sd = torch.load(checkpoint_path, map_location="cpu")
                try:
                    model.load_state_dict(sd, strict=False)
                except Exception:
                    # maybe the checkpoint is a full model object
                    try:
                        model = sd
                    except Exception as e:
                        st.error(f"Couldn't load checkpoint as state_dict or model object: {e}")
                        model = None
            else:
                # use torchvision pretrained for quick demo (not ideal for CIFAR but OK)
                model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
                model.fc = nn.Linear(model.fc.in_features, 10)

        elif model_type.startswith("distilbert"):
            # DistilBERT is heavier — use CPU carefully
            from transformers import DistilBertForSequenceClassification
            if checkpoint_path:
                # loading huggingface saved model dir or state_dict is complicated;
                # here we try to load as a HuggingFace model or fallback to pretrained
                try:
                    model = DistilBertForSequenceClassification.from_pretrained(checkpoint_path)
                except Exception:
                    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
                    sd = torch.load(checkpoint_path, map_location="cpu")
                    try:
                        model.load_state_dict(sd, strict=False)
                    except Exception:
                        st.warning("Uploaded DistilBERT checkpoint could not be applied exactly; using default pretrained.")
            else:
                model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
        else:
            st.error("Unsupported model type")
            model = None

        if model is not None:
            model_loaded = True
            st.success("Model instantiated (on CPU).")
            # display param count
            pbytes = param_bytes(model)
            st.write(f"Parameter count: {pbytes / 1e6:.2f} MB")
    except Exception as e:
        st.error(f"Error while loading model: {e}")
        model = None

# --- Evaluation ---
st.header("Evaluate Model")
if st.button("Run evaluation"):

    if not model_loaded and not use_example:
        st.error("No model loaded. Upload or use example.")
    else:
        if use_example:
            if model_type.startswith("resnet"):
                #resnet arch and weights
                model = resnet18(weights="IMAGENET1K_V1")
                model.fc = torch.nn.Linear(model.fc.in_features, 10)
                state_dict = torch.load(CIFAR_PATH, map_location=DEVICE)
                model.load_state_dict(state_dict)
                model_loaded = True
                
            elif model_type.startswith("distilbert"):
                model = DistilBertForSequenceClassification.from_pretrained(SST2_PATH)
                model_loaded = True
                
            else:
                st.error("Unknown model type for evaluation")
                metrics = {}
        
            # move model to device
            model.to(DEVICE) 
            model.eval()

        # ---------- Prepare dataset loader (small subset) ----------
        if model_type.startswith("resnet"):
            # CIFAR-10 test loader
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470,0.2435,0.2616))
            ])
            testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
            # subset for quick run
            test_subset = torch.utils.data.Subset(testset, list(range(min(len(testset), eval_samples))))
            test_loader = DataLoader(test_subset, batch_size=128, shuffle=False)
            example_batch = next(iter(DataLoader(test_subset, batch_size=1)))
            # compute accuracy
            acc = top1(model, test_loader, device=device)
            # latency (uses example batch)
            latency_s = measure_latency_s(model, example_batch, runs=int(latency_runs), warmup=int(latency_warmup), device=str(device.type))
            peak_mem = peak_gpu_mem_once(model, example_batch) if device.type == "cuda" else 0
            param_mb = param_bytes(model) / 1e6
            metrics = {"accuracy": float(acc), "latency_s": float(latency_s),
                       "param_MB": float(param_mb), "peak_mem_MB": float(peak_mem)/1e6 if peak_mem else 0.0}

        elif model_type.startswith("distilbert"):
            # SST-2 quick eval using glue validation (small subset)
            from datasets import load_dataset
            tokenizer = __import__("transformers").AutoTokenizer.from_pretrained("distilbert-base-uncased")
            dataset = load_dataset("glue", "sst2", split="validation")
            # tokenize and make small subset
            def tok(batch):
                return tokenizer(batch["sentence"], truncation=True, padding="max_length", max_length=128)
            ds = dataset.map(tok, batched=True)
            ds.set_format(type="torch", columns=["input_ids","attention_mask","label"])
            ds = ds.select(range(min(len(ds), eval_samples)))
            loader = DataLoader(ds, batch_size=16)
            # evaluate accuracy (use top1 utility expects model returning logits)
            acc = top1(model, loader, device=device)
            # latency (create example batch)
            example_batch = next(iter(DataLoader(ds, batch_size=1)))
            latency_s = measure_latency_s(model, example_batch, runs=int(latency_runs), warmup=int(latency_warmup), device=str(device.type))
            param_mb = param_bytes(model) / 1e6
            metrics = {"accuracy": float(acc), "latency_s": float(latency_s), "param_MB": float(param_mb), "peak_mem_MB": 0.0}
        else:
            st.error("Unknown model type for evaluation")
            metrics = {}

        # show results
        st.subheader("Results")
        for k,v in metrics.items():
            st.metric(k, round(v, 6) if isinstance(v, float) else v)
        st.json(metrics)

        # save run record using your report utility
        rec = log_experiment(metrics, model_name=model_type, dataset=("CIFAR-10" if "resnet" in model_type else "SST-2"), method="ui-eval", out_dir="outputs/reports")
        st.success("Logged experiment to outputs/reports")
        st.write(rec)