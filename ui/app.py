# ui/app.py
# Run: streamlit run ui/app.py
import streamlit as st
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os, sys
import json
from transformers import DistilBertTokenizer
import pandas as pd
from datasets import load_dataset

# Try to import plotly, fallback to streamlit native charts
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    make_subplots = None

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ui.model_loader import load_model #built-in model
from ui.model_detection import (
    load_generic_pytorch_model, 
    ModelProfile,
    detect_architecture_type
) #import model
from ui.model_inspection import (
    inspect_model,
    inspect_model_from_registry,
    ModelMetadata,
    ModelSource
)
from ui.baseline_registry import get_baseline_registry
from ui.comparison_orchestrator import (
    get_comparison_orchestrator,
    ComparisonGroup,
    WorkflowType as OrchestratorWorkflowType
)
from eval.metrics import top1, f1
from eval.latency import measure_latency_s
from eval.memory import param_bytes, peak_gpu_mem_once
from eval.report import log_experiment

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================================
# MODEL REGISTRY (UI SIDE)
# Maps dropdown selections -> model_loader keys
# ==========================================================
MODEL_REGISTRY = {
    "ResNet18": {
        "Baseline": "resnet18_baseline",
        "PTQ": "resnet18_ptq",
        "QAT": "resnet18_qat",
    },

    "DistilBERT": {
        "Baseline": "distilbert_baseline",
        "int8 (dynamic)": "distilbert_dynamic_int8",
        "int8 (static)": "distilbert_static_int8", #doesnt exist yet
        "QAT int8": "distilbert_qat_int8",
    },
}

# ==========================================================
# STREAMLIT
# ==========================================================
st.set_page_config(page_title="Quanteval — Model Evaluation", layout="wide", page_icon="🧠")

# Main title with description
st.title("🧠 Quanteval — Model Evaluation UI")
st.markdown("""
**A comprehensive quantization benchmarking framework for PyTorch models**

Choose your evaluation mode:
- **Single Model**: Evaluate one model's performance
- **Compare Models**: Manually compare two models side-by-side
- **Intelligent Benchmarking**: Automatically detect and compare models (recommended for beginners)
""")

st.sidebar.header("⚙️ Evaluation Options")

# ==========================================================
# EVALUATION MODE SELECTION
# ==========================================================
evaluation_mode = st.sidebar.radio(
    "📊 Evaluation Mode:",
    ["Single Model", "Compare Models", "Intelligent Benchmarking"],
    help="Choose how you want to evaluate models. Intelligent Benchmarking is recommended for beginners."
)

# ==========================================================
# MODEL SELECTION / USER UPLOAD
# ==========================================================
if evaluation_mode == "Single Model":
    use_generic_loader = False 
    
    model_source = st.sidebar.radio(
        "Choose model source:",
        ["Built-in Model", "Upload Custom Model"]
    )

    if model_source == "Built-in Model":
        model_type = st.sidebar.selectbox("Model Architecture", list(MODEL_REGISTRY.keys()))
        variant = st.sidebar.selectbox("Variant", list(MODEL_REGISTRY[model_type].keys()))
        model_key = MODEL_REGISTRY[model_type][variant]
        user_model_path = None
        baseline_key = None
        quantized_key = None

    else: #upload custom model
        st.sidebar.write("Upload a `.pt` or `.pth` model file")
        uploaded_file = st.sidebar.file_uploader("Upload model file", type=["pt", "pth"])
        
        # For generic models, we'll auto-detect the type
        model_type = None
        variant = "Custom Upload"
        model_key = None
        user_model_path = None
        baseline_key = None
        quantized_key = None
        use_generic_loader = True

        if uploaded_file:
            os.makedirs("uploads", exist_ok=True)
            temp_path = os.path.join("uploads", uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            user_model_path = temp_path

elif evaluation_mode == "Compare Models":  # Compare Models mode
    comparison_type = st.sidebar.radio(
        "Comparison Type:",
        ["Built-in vs Built-in", "Baseline vs Uploaded"]
    )
    
    if comparison_type == "Built-in vs Built-in":
        use_generic_loader = False
        model_type = st.sidebar.selectbox("Model Architecture", list(MODEL_REGISTRY.keys()))
        
        variants = list(MODEL_REGISTRY[model_type].keys())
        baseline_variant = st.sidebar.selectbox("Baseline Model", variants, index=0)
        quantized_variant = st.sidebar.selectbox("Quantized Model", variants, index=1 if len(variants) > 1 else 0)
        
        baseline_key = MODEL_REGISTRY[model_type][baseline_variant]
        quantized_key = MODEL_REGISTRY[model_type][quantized_variant]
        model_key = None
        user_model_path = None
        variant = None
        baseline_use_generic = False
        quantized_use_generic = False
        baseline_profile = None
        quantized_profile = None
    else:  # Baseline vs Uploaded
        model_type = st.sidebar.selectbox("Baseline Model Architecture", list(MODEL_REGISTRY.keys()))
        variants = list(MODEL_REGISTRY[model_type].keys())
        baseline_variant = st.sidebar.selectbox("Baseline Model", variants, index=0)
        baseline_key = MODEL_REGISTRY[model_type][baseline_variant]
        baseline_use_generic = False
        
        st.sidebar.write("---")
        st.sidebar.write("Upload model to compare:")
        uploaded_file = st.sidebar.file_uploader("Upload model file", type=["pt", "pth"], key="compare_upload")
        
        quantized_key = None
        quantized_variant = None
        quantized_use_generic = True
        user_model_path = None
        quantized_profile = None
        
        if uploaded_file:
            os.makedirs("uploads", exist_ok=True)
            temp_path = os.path.join("uploads", uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            user_model_path = temp_path

elif evaluation_mode == "Intelligent Benchmarking":
    # Intelligent Benchmarking mode - skip model selection here
    # All logic is handled in the execution section below
    pass

# ----------
# Common settings (only for Single Model and Compare Models modes)
# ----------
if evaluation_mode in ["Single Model", "Compare Models"]:
    device_choice = st.sidebar.selectbox("Run on", ["cpu", "cuda (if available)"])
    device = torch.device("cuda" if (device_choice.startswith("cuda") and torch.cuda.is_available()) else "cpu")

    eval_samples = st.sidebar.number_input("Num eval samples", min_value=32, max_value=5000, value=512, step=32)
    latency_runs = st.sidebar.number_input("Latency runs", min_value=5, max_value=200, value=20)
    latency_warmup = st.sidebar.number_input("Latency warmup", min_value=1, max_value=50, value=5)
else:
    # For Intelligent Benchmarking, these will be set in its own section
    device = None
    eval_samples = None
    latency_runs = None
    latency_warmup = None

# ==========================================================
# IMPORTED MODEL PROFILING DISPLAY
# ==========================================================
def display_model_profile(profile: ModelProfile):
    """Display model profile in Streamlit UI."""
    st.subheader("🔍 Model Profile")
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Architecture", profile.arch_type)
    with col2:
        st.metric("Parameters", f"{profile.param_counts['total']:,}")
    with col3:
        st.metric("FP32 Size", f"{profile.size_fp32['size_mb']:.1f} MB")
    with col4:
        st.metric("Model Depth", profile.depth)
    
    # Expandable detailed info
    with st.expander("📊 Detailed Profile Information"):
        profile_dict = profile.to_dict()
        
        # Size estimates
        st.write("**Size Estimates:**")
        size_df = pd.DataFrame([profile_dict['size_estimates']]).T
        size_df.columns = ["Size (MB)"]
        st.dataframe(size_df, width='stretch')
        
        # Layer composition (top 10)
        st.write("**Layer Composition (Top 10):**")
        layer_comp = dict(list(profile.layer_composition.items())[:10])
        layer_df = pd.DataFrame(list(layer_comp.items()), columns=["Layer Type", "Count"])
        st.dataframe(layer_df, width='stretch')
        
        # I/O shapes
        if profile.input_shapes:
            st.write("**Input Shapes:**")
            st.json(profile.input_shapes)
        if profile.output_shapes:
            st.write("**Output Shapes:**")
            st.json(profile.output_shapes)
        
        # FLOPs
        if profile.flops:
            st.write(f"**FLOPs:** {profile.flops:,.0f}")

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

    # Accuracy
    acc = top1(model, loader, device=device)

    # Latency example batch
    example_batch = next(iter(DataLoader(subset, batch_size=1)))
    imgs = example_batch[0].to(device)  # image only
    if imgs.ndim == 3:
        imgs = imgs.unsqueeze(0)

    latency_s = measure_latency_s(
        model, imgs,
        runs=int(latency_runs),
        warmup=int(latency_warmup),
        device=str(device.type),
    )

    # peak_mem = peak_gpu_mem_once(model, imgs) if device.type == "cuda" else 0
    param_mb = param_bytes(model) / 1e6

    return {
        "Accuracy": float(acc),
        "Latency (s)": float(latency_s),
        "Latency (ms)": float(latency_s * 1000),
        "param_MB": float(param_mb),
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

    # Latency input
    batch = next(iter(DataLoader(ds, batch_size=1)))
    example_input = {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
    }

    latency_s = measure_latency_s(
        model, example_input,
        runs=int(latency_runs),
        warmup=int(latency_warmup),
        device=str(device.type),
    )
    param_mb = param_bytes(model) / 1e6

    return {
        "Accuracy": float(acc),
        "Latency (s)": float(latency_s),
        "Latency (ms)": float(latency_s * 1000),
        "param_MB": float(param_mb),
    }
    
def evaluate_generic_model(model, profile, eval_samples, latency_runs, latency_warmup, device):
    """
    Fallback evaluation for generic models when we don't have a specific dataset.
    Just measure latency and size.
    """
    st.warning("⚠️ No specific dataset available for this model. Performing basic profiling only.")
    
    # Generate sample input based on detected architecture
    if profile.input_shapes and isinstance(profile.input_shapes, dict):
        # Try to reconstruct input from shapes
        sample_input = torch.randn(*profile.input_shapes['shape']).to(device)
    else:
        # Fallback to common image input
        sample_input = torch.randn(1, 3, 224, 224).to(device)
    
    # Measure latency
    latency_s = measure_latency_s(
        model, sample_input,
        runs=int(latency_runs),
        warmup=int(latency_warmup),
        device=str(device.type),
    )
    
    param_mb = param_bytes(model) / 1e6
    
    return {
        "Accuracy": None,  # Can't measure without dataset
        "Latency (s)": float(latency_s),
        "Latency (ms)": float(latency_s * 1000),
        "param_MB": float(param_mb),
        "peak_mem_MB": 0.0,
    }

# ==========================================================
# VISUALIZATION HELPERS
# ==========================================================
def create_metrics_charts(metrics, model_name="Model"):
    """Create individual charts for different metric types."""
    if not PLOTLY_AVAILABLE:
        # Fallback to simple bar chart - display directly and return None
        chart_data = pd.DataFrame({
            "Metric": list(metrics.keys()),
            "Value": list(metrics.values())
        })
        st.bar_chart(chart_data.set_index("Metric"))
        return None
    
    charts_created = []
    
    # Accuracy chart (percentage)
    accuracy_metrics = {k: v for k, v in metrics.items() if "Accuracy" in k}
    if accuracy_metrics and any(v > 0 for v in accuracy_metrics.values()):
        acc_keys = list(accuracy_metrics.keys())
        acc_values = [v * 100 for v in accuracy_metrics.values()]  # Convert to percentage
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Bar(
            x=acc_keys,
            y=acc_values,
            marker_color='#1f77b4',
            text=[f"{v:.2f}%" for v in acc_values],
            textposition='outside'
        ))
        fig_acc.update_layout(
            title="Accuracy",
            xaxis_title="Metric",
            yaxis_title="Accuracy (%)",
            yaxis_range=[0, 100],
            height=400,
            showlegend=False
        )
        charts_created.append(fig_acc)
    
    # Latency chart (ms)
    latency_metrics = {k: v for k, v in metrics.items() if "Latency" in k and "ms" in k}
    if latency_metrics and any(v > 0 for v in latency_metrics.values()):
        lat_keys = list(latency_metrics.keys())
        lat_values = [latency_metrics[k] for k in lat_keys]
        fig_lat = go.Figure()
        fig_lat.add_trace(go.Bar(
            x=lat_keys,
            y=lat_values,
            marker_color='#ff7f0e',
            text=[f"{v:.3f} ms" for v in lat_values],
            textposition='outside'
        ))
        fig_lat.update_layout(
            title="Latency",
            xaxis_title="Metric",
            yaxis_title="Latency (ms)",
            height=400,
            showlegend=False
        )
        charts_created.append(fig_lat)
    
    # Model size chart
    size_metrics = {k: v for k, v in metrics.items() if "param" in k.lower() and "MB" in k}
    if size_metrics and any(v > 0 for v in size_metrics.values()):
        size_keys = list(size_metrics.keys())
        size_values = [size_metrics[k] for k in size_keys]
        fig_size = go.Figure()
        fig_size.add_trace(go.Bar(
            x=size_keys,
            y=size_values,
            marker_color='#2ca02c',
            text=[f"{v:.2f} MB" for v in size_values],
            textposition='outside'
        ))
        fig_size.update_layout(
            title="Model Size",
            xaxis_title="Metric",
            yaxis_title="Size (MB)",
            height=400,
            showlegend=False
        )
        charts_created.append(fig_size)    
    
    # Return list of charts (or None if no charts created)
    return charts_created if charts_created else None

# ==========================================================
# COMPARISON CHARTS
# ==========================================================

def create_comparison_chart(baseline_metrics, quantized_metrics, baseline_name, quantized_name):
    """Create individual comparison charts between baseline and quantized models."""
    if not PLOTLY_AVAILABLE:
        # Fallback to simple comparison - display directly and return None
        comparison_data = pd.DataFrame({
            baseline_name: baseline_metrics,
            quantized_name: quantized_metrics
        })
        st.bar_chart(comparison_data)
        return None
    
    charts_created = []
    
    # Accuracy comparison
    if "Accuracy" in baseline_metrics and "Accuracy" in quantized_metrics:
        baseline_acc = baseline_metrics["Accuracy"] * 100
        quantized_acc = quantized_metrics["Accuracy"] * 100
        if baseline_acc > 0 or quantized_acc > 0:
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Bar(
                x=[baseline_name, quantized_name],
                y=[baseline_acc, quantized_acc],
                marker_color=['#1f77b4', '#ff7f0e'],
                text=[f"{baseline_acc:.2f}%", f"{quantized_acc:.2f}%"],
                textposition='outside'
            ))
            fig_acc.update_layout(
                title="Accuracy Comparison",
                xaxis_title="Model",
                yaxis_title="Accuracy (%)",
                yaxis_range=[0, 100],
                height=400,
                showlegend=False
            )
            charts_created.append(fig_acc)
    
    # Latency comparison
    if "Latency (ms)" in baseline_metrics and "Latency (ms)" in quantized_metrics:
        baseline_lat = baseline_metrics["Latency (ms)"]
        quantized_lat = quantized_metrics["Latency (ms)"]
        if baseline_lat > 0 or quantized_lat > 0:
            fig_lat = go.Figure()
            fig_lat.add_trace(go.Bar(
                x=[baseline_name, quantized_name],
                y=[baseline_lat, quantized_lat],
                marker_color=['#1f77b4', '#ff7f0e'],
                text=[f"{baseline_lat:.3f} ms", f"{quantized_lat:.3f} ms"],
                textposition='outside'
            ))
            fig_lat.update_layout(
                title="Latency Comparison",
                xaxis_title="Model",
                yaxis_title="Latency (ms)",
                height=400,
                showlegend=False
            )
            charts_created.append(fig_lat)
    
    # Size comparison
    if "param_MB" in baseline_metrics and "param_MB" in quantized_metrics:
        baseline_size = baseline_metrics["param_MB"]
        quantized_size = quantized_metrics["param_MB"]
        if baseline_size > 0 or quantized_size > 0:
            fig_size = go.Figure()
            fig_size.add_trace(go.Bar(
                x=[baseline_name, quantized_name],
                y=[baseline_size, quantized_size],
                marker_color=['#1f77b4', '#ff7f0e'],
                text=[f"{baseline_size:.2f} MB", f"{quantized_size:.2f} MB"],
                textposition='outside'
            ))
            fig_size.update_layout(
                title="Model Size Comparison",
                xaxis_title="Model",
                yaxis_title="Size (MB)",
                height=400,
                showlegend=False
            )
            charts_created.append(fig_size)    
    
    # Return list of charts (or None if no charts created)
    return charts_created if charts_created else None


def calculate_improvements(baseline_metrics, quantized_metrics):
    """Calculate improvement percentages."""
    improvements = {}
    
    if "Accuracy" in baseline_metrics and "Accuracy" in quantized_metrics:
        acc_diff = quantized_metrics["Accuracy"] - baseline_metrics["Accuracy"]
        improvements["Accuracy"] = {
            "absolute": acc_diff,
            "relative": (acc_diff / baseline_metrics["Accuracy"]) * 100 if baseline_metrics["Accuracy"] > 0 else 0
        }
    
    if "Latency (ms)" in baseline_metrics and "Latency (ms)" in quantized_metrics:
        lat_diff = baseline_metrics["Latency (ms)"] - quantized_metrics["Latency (ms)"]
        improvements["Latency"] = {
            "absolute": lat_diff,
            "relative": (lat_diff / baseline_metrics["Latency (ms)"]) * 100 if baseline_metrics["Latency (ms)"] > 0 else 0
        }
    
    if "param_MB" in baseline_metrics and "param_MB" in quantized_metrics:
        size_diff = baseline_metrics["param_MB"] - quantized_metrics["param_MB"]
        improvements["Model Size"] = {
            "absolute": size_diff,
            "relative": (size_diff / baseline_metrics["param_MB"]) * 100 if baseline_metrics["param_MB"] > 0 else 0
        }
    
    return improvements


# ==========================================================
# SAVED MODELS MANAGER
# ==========================================================

def init_saved_models():
    """Initialize saved models in session state"""
    if "saved_models" not in st.session_state:
        st.session_state.saved_models = {}


def save_model_to_session(model_name: str, model_path: str, metadata: ModelMetadata):
    """Save a model to session state for later use"""
    init_saved_models()
    st.session_state.saved_models[model_name] = {
        "path": model_path,
        "metadata": metadata.to_dict(),
        "architecture": metadata.architecture,
        "task": metadata.task.value,
        "dataset": metadata.dataset.value,
        "precision": metadata.precision.value,
        "quantization_method": metadata.quantization_method.value,
    }


def get_saved_models() -> dict:
    """Get all saved models"""
    init_saved_models()
    return st.session_state.saved_models


def delete_saved_model(model_name: str):
    """Delete a saved model from session state"""
    init_saved_models()
    if model_name in st.session_state.saved_models:
        del st.session_state.saved_models[model_name]


# ==========================================================
# NEW WORKFLOW HELPERS
# ==========================================================

def display_model_metadata(metadata: ModelMetadata):
    """Display detected model metadata in the UI."""
    st.subheader("🔍 Detected Model Information")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Architecture", metadata.architecture)
    with col2:
        st.metric("Task", metadata.task.value.replace("_", " ").title())
    with col3:
        st.metric("Dataset", metadata.dataset.value.upper() if metadata.dataset.value != "unknown" else "Unknown")
    with col4:
        st.metric("Precision", metadata.precision.value.upper())
    
    # Quantization info
    with st.expander("📋 Detailed Metadata"):
        metadata_dict = metadata.to_dict()
        st.json(metadata_dict)
        
        if metadata.is_quantized():
            st.info(f"✅ **Quantized Model Detected**: {metadata.quantization_method.value.upper()}")
        elif metadata.is_baseline():
            st.info("✅ **Baseline FP32 Model Detected**")


def get_evaluation_pipeline(metadata: ModelMetadata):
    """
    Get the appropriate evaluation pipeline function based on metadata.
    
    Returns:
        (evaluation_function, dataset_name) tuple
    """
    if metadata.task.value == "image_classification":
        if metadata.dataset.value == "cifar10":
            return evaluate_cifar10_model, "CIFAR-10"
        else:
            # Generic image classification
            return None, None
    elif metadata.task.value == "text_classification":
        if metadata.dataset.value == "sst2":
            return evaluate_sst2_model, "SST-2"
        else:
            # Generic text classification
            return None, None
    else:
        return None, None


def evaluate_model_with_metadata(
    model,
    metadata: ModelMetadata,
    eval_samples,
    latency_runs,
    latency_warmup,
    device
):
    """
    Evaluate a model using the appropriate pipeline based on metadata.
    Falls back to generic evaluation if specific pipeline not available.
    """
    eval_func, dataset_name = get_evaluation_pipeline(metadata)
    
    if eval_func:
        return eval_func(model, eval_samples, latency_runs, latency_warmup, device)
    else:
        # Fallback to generic evaluation
        profile = ModelProfile(model, str(device))
        return evaluate_generic_model(model, profile, eval_samples, latency_runs, latency_warmup, device)


# ==========================================================
# RUN EVALUATION - Single Model
# ==========================================================
if evaluation_mode == "Single Model":
    if st.button("Load Model"):
        if use_generic_loader and not user_model_path:
            st.error("Please upload a model file first.")
            st.stop()
        
        # Load model
        if use_generic_loader:
            st.info(f"Loading custom model on {device.type.upper()}...")
            model, load_info = load_generic_pytorch_model(user_model_path, str(device))
            
            if not load_info["success"]:
                st.error(f"❌ Failed to load model: {load_info['error']}")
                if load_info["warnings"]:
                    for warning in load_info["warnings"]:
                        st.warning(warning)
                st.stop()
            
            st.success("✅ Model loaded successfully!")
            
            # Profile the model
            st.info("Profiling model...")
            profile = ModelProfile(model, str(device))
            display_model_profile(profile)
                
            # Auto-detect model type for evaluation
            model_type = profile.arch_type
            
        else:  # Built-in model
            st.info(f"Loading {model_type} — {variant} on {device.type.upper()}...")
            try:
                model = load_model(model_key, device)
                st.success("✅ Model loaded successfully.")
                profile = None
            except Exception as e:
                st.error(f"Error loading model: {e}")
                st.stop()

        # Evaluate
        st.info("Running evaluation...")
        
        # Determine which evaluation pipeline to use
        if use_generic_loader:
            # Check if we recognize the architecture
            if "ResNet" in str(type(model)) or profile.arch_type == "CNN":
                # Try CIFAR-10 evaluation
                try:
                    metrics = evaluate_cifar10_model(model, eval_samples, latency_runs, latency_warmup, device)
                except Exception as e:
                    st.warning(f"CIFAR-10 evaluation failed: {e}. Using generic evaluation.")
                    metrics = evaluate_generic_model(model, profile, eval_samples, latency_runs, latency_warmup, device)
            elif "DistilBert" in str(type(model)) or profile.arch_type == "Transformer":
                # Try SST-2 evaluation
                try:
                    metrics = evaluate_sst2_model(model, eval_samples, latency_runs, latency_warmup, device)
                except Exception as e:
                    st.warning(f"SST-2 evaluation failed: {e}. Using generic evaluation.")
                    metrics = evaluate_generic_model(model, profile, eval_samples, latency_runs, latency_warmup, device)
            else:
                # Generic evaluation
                metrics = evaluate_generic_model(model, profile, eval_samples, latency_runs, latency_warmup, device)
        else:
            # Built-in model - use known evaluation pipeline
            if model_type == "ResNet18":
                metrics = evaluate_cifar10_model(model, eval_samples, latency_runs, latency_warmup, device)
            elif model_type == "DistilBERT":
                metrics = evaluate_sst2_model(model, eval_samples, latency_runs, latency_warmup, device)
            else:
                st.error("Unsupported model architecture.")
                st.stop()

        # Results
        display_name = variant if not use_generic_loader else f"Custom {model_type}"
        st.subheader(f"📊 Results: {display_name}")
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if metrics['Accuracy'] is not None:
                st.metric("Accuracy", f"{metrics['Accuracy']*100:.2f}%")
            else:
                st.metric("Accuracy", "N/A")
        with col2:
            st.metric("Latency", f"{metrics['Latency (ms)']:.3f} ms")
        with col3:
            st.metric("Model Size", f"{metrics['param_MB']:.2f} MB")
        with col4:
            mem_key = "peak_memory_MB" if "peak_memory_MB" in metrics else "peak_mem_MB"
            mem_val = metrics.get(mem_key, 0.0)
            st.metric("Peak Memory", f"{mem_val:.2f} MB" if mem_val > 0 else "N/A")

        # Create and display charts
        charts = create_metrics_charts(metrics, display_name)
        if charts is not None:
            for chart in charts:
                st.plotly_chart(chart, width='stretch')

        # Detailed metrics table
        st.subheader("Detailed Metrics")
        metrics_df = pd.DataFrame([metrics]).T
        metrics_df.columns = ["Value"]
        st.dataframe(metrics_df, width='stretch')

        # Log report
        rec = log_experiment(
            metrics,
            model_name=display_name,
            method="ui-eval",
            out_dir="outputs/reports"
        )
        st.success("✅ Logged experiment to outputs/reports")

# ==========================================================
# COMPARE MODELS MODE
# ==========================================================
elif evaluation_mode == "Compare Models":  # Compare Models mode
    if st.button("Compare Models"):
        # Validation
        if comparison_type == "Built-in vs Built-in":
            if baseline_key == quantized_key:
                st.error("Please select different models for comparison.")
                st.stop()
        else:  # Baseline vs Uploaded
            if not user_model_path:
                st.error("Please upload a model file to compare.")
                st.stop()
        
        results = {}
        
        # ========== LOAD AND EVALUATE BASELINE ==========
        st.info(f"Loading baseline: {baseline_variant}...")
        try:
            baseline_model = load_model(baseline_key, device)
            st.success(f"✅ Baseline model loaded: {baseline_variant}")
            baseline_profile = None
        except Exception as e:
            st.error(f"Error loading baseline model: {e}")
            st.stop()
        
        st.info("Evaluating baseline model...")
        if model_type == "ResNet18":
            baseline_metrics = evaluate_cifar10_model(baseline_model, eval_samples, latency_runs, latency_warmup, device)
        elif model_type == "DistilBERT":
            baseline_metrics = evaluate_sst2_model(baseline_model, eval_samples, latency_runs, latency_warmup, device)
        else:
            st.error("Unsupported model architecture.")
            st.stop()
        
        results["baseline"] = baseline_metrics
        
        # ========== LOAD AND EVALUATE COMPARISON MODEL ==========
        if comparison_type == "Built-in vs Built-in":
            # Built-in model
            st.info(f"Loading comparison model: {quantized_variant}...")
            try:
                quantized_model = load_model(quantized_key, device)
                st.success(f"✅ Comparison model loaded: {quantized_variant}")
                quantized_profile = None
            except Exception as e:
                st.error(f"Error loading comparison model: {e}")
                st.stop()
            
            st.info("Evaluating comparison model...")
            if model_type == "ResNet18":
                quantized_metrics = evaluate_cifar10_model(quantized_model, eval_samples, latency_runs, latency_warmup, device)
            elif model_type == "DistilBERT":
                quantized_metrics = evaluate_sst2_model(quantized_model, eval_samples, latency_runs, latency_warmup, device)
            else:
                st.error("Unsupported model architecture.")
                st.stop()
            
            comparison_name = quantized_variant
            comparison_display_name = quantized_variant
            
        else:  # Baseline vs Uploaded
            # Uploaded model
            if not uploaded_file or not user_model_path:
                st.error("Please upload a model file to compare.")
                st.stop()
            
            st.info(f"Loading uploaded model: {uploaded_file.name}...")
            model, load_info = load_generic_pytorch_model(user_model_path, str(device))
            
            if not load_info["success"]:
                st.error(f"❌ Failed to load uploaded model: {load_info['error']}")
                if load_info["warnings"]:
                    for warning in load_info["warnings"]:
                        st.warning(warning)
                st.stop()
            
            st.success("✅ Uploaded model loaded successfully!")
            
            # Profile the uploaded model
            st.info("Profiling uploaded model...")
            quantized_profile = ModelProfile(model, str(device))
            display_model_profile(quantized_profile)
            
            quantized_model = model
            comparison_name = uploaded_file.name
            comparison_display_name = f"Uploaded ({quantized_profile.arch_type})"
            
            st.info("Evaluating uploaded model...")
            
            # Auto-detect evaluation pipeline based on architecture
            detected_type = quantized_profile.arch_type
            if "ResNet" in str(type(model)) or detected_type == "CNN":
                try:
                    quantized_metrics = evaluate_cifar10_model(quantized_model, eval_samples, latency_runs, latency_warmup, device)
                except Exception as e:
                    st.warning(f"CIFAR-10 evaluation failed: {e}. Using generic evaluation.")
                    quantized_metrics = evaluate_generic_model(quantized_model, quantized_profile, eval_samples, latency_runs, latency_warmup, device)
            elif "DistilBert" in str(type(model)) or detected_type == "Transformer":
                try:
                    quantized_metrics = evaluate_sst2_model(quantized_model, eval_samples, latency_runs, latency_warmup, device)
                except Exception as e:
                    st.warning(f"SST-2 evaluation failed: {e}. Using generic evaluation.")
                    quantized_metrics = evaluate_generic_model(quantized_model, quantized_profile, eval_samples, latency_runs, latency_warmup, device)
            else:
                quantized_metrics = evaluate_generic_model(quantized_model, quantized_profile, eval_samples, latency_runs, latency_warmup, device)
        
        results["quantized"] = quantized_metrics
        
        # ========== DISPLAY COMPARISON ==========
        baseline_display_name = f"{baseline_variant} (Baseline)"
        st.subheader(f"📊 Comparison: {baseline_display_name} vs {comparison_display_name}")
        
        # Side-by-side metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### {baseline_display_name}")
            if baseline_metrics['Accuracy'] is not None:
                st.metric("Accuracy", f"{baseline_metrics['Accuracy']*100:.2f}%")
            else:
                st.metric("Accuracy", "N/A")
            st.metric("Latency", f"{baseline_metrics['Latency (ms)']:.3f} ms")
            st.metric("Model Size", f"{baseline_metrics['param_MB']:.2f} MB")
        
        with col2:
            st.markdown(f"### {comparison_display_name}")
            if quantized_metrics['Accuracy'] is not None:
                st.metric("Accuracy", f"{quantized_metrics['Accuracy']*100:.2f}%")
            else:
                st.metric("Accuracy", "N/A")
            st.metric("Latency", f"{quantized_metrics['Latency (ms)']:.3f} ms")
            st.metric("Model Size", f"{quantized_metrics['param_MB']:.2f} MB")
        
        # Improvement metrics (only if both have accuracy)
        if baseline_metrics.get('Accuracy') is not None and quantized_metrics.get('Accuracy') is not None:
            improvements = calculate_improvements(baseline_metrics, quantized_metrics)
            
            st.subheader("📈 Improvements")
            imp_col1, imp_col2, imp_col3 = st.columns(3)
            
            if "Accuracy" in improvements:
                with imp_col1:
                    acc_imp = improvements["Accuracy"]
                    delta = f"{acc_imp['relative']:+.2f}%"
                    st.metric("Accuracy Change", f"{acc_imp['absolute']*100:+.2f}%", delta=delta)
            
            if "Latency" in improvements:
                with imp_col2:
                    lat_imp = improvements["Latency"]
                    delta = f"{lat_imp['relative']:+.2f}%"
                    st.metric("Latency Change", f"{lat_imp['absolute']:+.3f} ms", delta=delta)
            
            if "Model Size" in improvements:
                with imp_col3:
                    size_imp = improvements["Model Size"]
                    delta = f"{size_imp['relative']:+.2f}%"
                    st.metric("Size Reduction", f"{size_imp['absolute']:.2f} MB", delta=delta)
        else:
            st.info("⚠️ Accuracy comparison not available (one or both models don't have accuracy metrics)")
        
        # Comparison charts
        charts = create_comparison_chart(baseline_metrics, quantized_metrics, baseline_display_name, comparison_display_name)
        if charts is not None:
            for chart in charts:
                st.plotly_chart(chart, width='stretch')
        
        # Comparison table
        st.subheader("Detailed Comparison")
        comparison_df = pd.DataFrame({
            baseline_display_name: baseline_metrics,
            comparison_display_name: quantized_metrics
        })
        st.dataframe(comparison_df, width='stretch')
        
        # Log both experiments
        log_experiment(
            baseline_metrics,
            model_name=f"{model_type}-{baseline_variant}",
            method="ui-eval-comparison",
            out_dir="outputs/reports"
        )
        log_experiment(
            quantized_metrics,
            model_name=f"{comparison_name}",
            method="ui-eval-comparison",
            out_dir="outputs/reports"
        )
        st.success("✅ Both experiments logged to outputs/reports")

# ==========================================================
# INTELLIGENT BENCHMARKING MODE
# ==========================================================
elif evaluation_mode == "Intelligent Benchmarking":
    st.header("🧠 Intelligent Benchmarking")
    st.markdown("""
    <div style='background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
    <h4 style='margin-top: 0;'>✨ Perfect for Beginners</h4>
    <p>This mode automatically detects your model type and suggests the best benchmarking workflow. 
    No need to understand quantization methods or manually find baselines!</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    **How it works:**
    - **Workflow A**: Upload a quantized model → We automatically find and compare it to the baseline
    - **Workflow B**: Upload a baseline model → We find all quantized variants and rank them by performance
    
    Just upload your model and let the system do the rest! 🚀
    """)
    
    # Initialize saved models
    init_saved_models()
    
    # Saved Models Section
    st.sidebar.header("💾 Saved Models")
    st.sidebar.caption("Quickly switch between previously analyzed models")
    saved_models = get_saved_models()
    selected_saved = "-- Select a saved model --"
    
    if saved_models:
        saved_model_names = list(saved_models.keys())
        selected_saved = st.sidebar.selectbox(
            "Load saved model",
            ["-- Select a saved model --"] + saved_model_names,
            key="intelligent_saved_select",
            help="Select a previously saved model to analyze"
        )
        
        if selected_saved != "-- Select a saved model --":
            # Display saved model info
            saved_info = saved_models[selected_saved]
            with st.sidebar.expander(f"📋 {selected_saved} Details"):
                st.write(f"**Architecture**: {saved_info['architecture']}")
                st.write(f"**Task**: {saved_info['task'].replace('_', ' ').title()}")
                st.write(f"**Dataset**: {saved_info['dataset'].upper() if saved_info['dataset'] != 'unknown' else 'Unknown'}")
                st.write(f"**Precision**: {saved_info['precision'].upper()}")
                st.write(f"**Quantization**: {saved_info['quantization_method'].upper()}")
            
            # Delete button
            if st.sidebar.button(f"🗑️ Delete", key=f"delete_{selected_saved}", help=f"Remove {selected_saved} from saved models"):
                delete_saved_model(selected_saved)
                st.sidebar.success(f"✅ Deleted {selected_saved}")
                st.rerun()
    else:
        st.sidebar.info("No saved models yet. Upload and analyze a model to save it here.")
    
    st.sidebar.write("---")
    
    # Model upload
    st.sidebar.header("📤 Upload Model")
    uploaded_file = st.sidebar.file_uploader(
        "Upload model file", 
        type=["pt", "pth"],
        key="intelligent_upload"
    )
    
    # Settings
    st.sidebar.header("⚙️ Evaluation Settings")
    device_choice = st.sidebar.selectbox(
        "Run on", 
        ["cpu", "cuda (if available)"], 
        key="intelligent_device",
        help="Device to run evaluation on"
    )
    device = torch.device("cuda" if (device_choice.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    
    eval_samples = st.sidebar.number_input(
        "Num eval samples", 
        min_value=32, 
        max_value=5000, 
        value=512, 
        step=32, 
        key="intelligent_samples",
        help="Number of samples to use for accuracy evaluation"
    )
    latency_runs = st.sidebar.number_input(
        "Latency runs", 
        min_value=5, 
        max_value=200, 
        value=20, 
        key="intelligent_latency_runs",
        help="Number of inference runs to average for latency measurement"
    )
    latency_warmup = st.sidebar.number_input(
        "Latency warmup", 
        min_value=1, 
        max_value=50, 
        value=5, 
        key="intelligent_warmup",
        help="Number of warmup runs before measuring latency"
    )
    
    # Determine which model to use (saved or uploaded)
    model_to_use = None
    model_path_to_use = None
    use_saved_model = False
    
    if selected_saved != "-- Select a saved model --" and selected_saved in saved_models:
        model_path_to_use = saved_models[selected_saved]["path"]
        use_saved_model = True
    elif uploaded_file:
        # Save uploaded file
        os.makedirs("uploads", exist_ok=True)
        model_path_to_use = os.path.join("uploads", uploaded_file.name)
        with open(model_path_to_use, "wb") as f:
            f.write(uploaded_file.read())
    
    # Initialize session state for IB analysis results
    if "ib_analysis_complete" not in st.session_state:
        st.session_state.ib_analysis_complete = False
    if "ib_model" not in st.session_state:
        st.session_state.ib_model = None
    if "ib_metadata" not in st.session_state:
        st.session_state.ib_metadata = None
    if "ib_initial_metrics" not in st.session_state:
        st.session_state.ib_initial_metrics = None
    if "ib_model_path" not in st.session_state:
        st.session_state.ib_model_path = None
    if "ib_use_saved_model" not in st.session_state:
        st.session_state.ib_use_saved_model = False
    if "ib_uploaded_filename" not in st.session_state:
        st.session_state.ib_uploaded_filename = None
    if "ib_comparison_results" not in st.session_state:
        st.session_state.ib_comparison_results = None
    if "ib_comparison_type" not in st.session_state:
        st.session_state.ib_comparison_type = None
    
    if st.button("🔍 Analyze Model", key="intelligent_analyze"):
        if not model_path_to_use:
            st.error("Please upload a model file or select a saved model first.")
            st.stop()
        
        # Load model
        if use_saved_model:
            st.info(f"Loading saved model: {selected_saved}...")
        else:
            st.info(f"Loading model: {uploaded_file.name}...")
        
        model, load_info = load_generic_pytorch_model(model_path_to_use, str(device))
        
        if not load_info["success"]:
            st.error(f"❌ Failed to load model: {load_info['error']}")
            if load_info["warnings"]:
                for warning in load_info["warnings"]:
                    st.warning(warning)
            st.stop()
        
        st.success("✅ Model loaded successfully!")
        
        # Inspect model
        st.info("🔍 Inspecting model...")
        metadata = inspect_model(
            model,
            source=ModelSource.USER,
            file_path=model_path_to_use,
            device=str(device)
        )
        
        # Display metadata
        display_model_metadata(metadata)
        
        # First, evaluate the model
        st.info("Running evaluation...")
        initial_metrics = evaluate_model_with_metadata(
            model, metadata, eval_samples, latency_runs, latency_warmup, device
        )
        
        # Store in session state
        st.session_state.ib_analysis_complete = True
        st.session_state.ib_model = model
        st.session_state.ib_metadata = metadata
        st.session_state.ib_initial_metrics = initial_metrics
        st.session_state.ib_model_path = model_path_to_use
        st.session_state.ib_use_saved_model = use_saved_model
        if uploaded_file:
            st.session_state.ib_uploaded_filename = uploaded_file.name
        else:
            st.session_state.ib_uploaded_filename = selected_saved if use_saved_model else None
        
        st.rerun()
    
    # Display saved model option (outside analyze block, but only if analysis is complete)
    if st.session_state.ib_analysis_complete and not st.session_state.ib_use_saved_model and st.session_state.ib_uploaded_filename:
        st.sidebar.write("---")
        st.sidebar.subheader("💾 Save This Model")
        st.sidebar.caption("Save this model for quick access later")
        model_save_name = st.sidebar.text_input(
            "Model name",
            value=st.session_state.ib_uploaded_filename.replace(".pt", "").replace(".pth", "") if st.session_state.ib_uploaded_filename else "",
            key="intelligent_save_name",
            help="Give this model a memorable name"
        )
        if st.sidebar.button("💾 Save to Saved Models", key="intelligent_save", use_container_width=True):
            if model_save_name:
                save_model_to_session(model_save_name, st.session_state.ib_model_path, st.session_state.ib_metadata)
                st.sidebar.success(f"✅ Saved as '{model_save_name}'")
                st.rerun()
            else:
                st.sidebar.error("⚠️ Please enter a model name")
    
    # Display results if analysis is complete
    if st.session_state.ib_analysis_complete:
        model = st.session_state.ib_model
        metadata = st.session_state.ib_metadata
        initial_metrics = st.session_state.ib_initial_metrics
        
        # Display initial evaluation results
        st.subheader("📊 Initial Evaluation Results")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{initial_metrics['Accuracy']*100:.2f}%" if initial_metrics.get('Accuracy') else "N/A")
        with col2:
            st.metric("Latency", f"{initial_metrics['Latency (ms)']:.3f} ms")
        with col3:
            st.metric("Model Size", f"{initial_metrics['param_MB']:.2f} MB")
        with col4:
            st.metric("Peak Memory", f"{initial_metrics.get('peak_mem_MB', 0):.2f} MB" if initial_metrics.get('peak_mem_MB', 0) > 0 else "N/A")
        
        st.write("---")
        
        # NOW determine workflow and show comparison options AFTER evaluation
        orchestrator = get_comparison_orchestrator()
        workflow = orchestrator.determine_workflow(metadata)
        
        if workflow == OrchestratorWorkflowType.QUANTIZED_TO_BASELINE:
            st.subheader("📊 Workflow A: Quantized Model → Compare to Baseline")
            st.markdown("""
            <div style='background-color: #e3f2fd; padding: 15px; border-radius: 5px; border-left: 4px solid #2196F3; margin-bottom: 20px;'>
            <p style='color: #1565C0; margin: 0;'><strong>Detected:</strong> You uploaded a quantized model. We'll find the matching baseline 
            and compare them side-by-side.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Find baseline
            baseline_result = orchestrator.find_baseline_for_quantized(metadata, str(device))
            
            if not baseline_result:
                st.warning("⚠️ No baseline found for this model.")
                st.write("""
                **Options:**
                1. Upload a baseline model manually
                2. Register a baseline in the system
                3. Continue with single model evaluation
                """)
                
                # Option to compare with baseline manually
                # Check if comparison already done
                if st.session_state.ib_comparison_results and st.session_state.ib_comparison_type == "baseline_comparison":
                    # Display stored comparison
                    baseline_metrics = st.session_state.ib_comparison_results["baseline_metrics"]
                    quantized_metrics = st.session_state.ib_comparison_results["quantized_metrics"]
                    baseline_metadata = st.session_state.ib_comparison_results.get("baseline_metadata")
                    
                    st.subheader("📊 Comparison: Baseline vs Quantized")
                else:
                    st.markdown("**Would you like to compare with a baseline?**")
                    if st.button("🔍 Find and Compare with Baseline", key="find_baseline_manual"):
                        # Try to find baseline again
                        baseline_result = orchestrator.find_baseline_for_quantized(metadata, str(device))
                        if baseline_result:
                            baseline_metadata, baseline_model = baseline_result
                            st.success(f"✅ Found baseline: {baseline_metadata.architecture}")
                            
                            # Evaluate baseline
                            st.info("Evaluating baseline model...")
                            baseline_metrics = evaluate_model_with_metadata(
                                baseline_model, baseline_metadata, eval_samples, latency_runs, latency_warmup, device
                            )
                            
                            # Store in session state
                            st.session_state.ib_comparison_results = {
                                "baseline_metrics": baseline_metrics,
                                "quantized_metrics": initial_metrics,
                                "baseline_metadata": baseline_metadata,
                                "quantized_metadata": metadata
                            }
                            st.session_state.ib_comparison_type = "baseline_comparison"
                            st.rerun()
                        else:
                            st.error("❌ Could not find a baseline for comparison.")
                            st.stop()
                    
                    # If no comparison yet, stop here
                    if not st.session_state.ib_comparison_results:
                        st.stop()
                    
                    # Get stored results
                    baseline_metrics = st.session_state.ib_comparison_results["baseline_metrics"]
                    quantized_metrics = st.session_state.ib_comparison_results["quantized_metrics"]
                    baseline_metadata = st.session_state.ib_comparison_results.get("baseline_metadata")
                    
                    st.subheader("📊 Comparison: Baseline vs Quantized")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("### Baseline (FP32)")
                        if baseline_metrics.get('Accuracy'):
                            st.metric("Accuracy", f"{baseline_metrics['Accuracy']*100:.2f}%")
                        st.metric("Latency", f"{baseline_metrics['Latency (ms)']:.3f} ms")
                        st.metric("Model Size", f"{baseline_metrics['param_MB']:.2f} MB")
                    
                    with col2:
                        st.markdown("### Quantized Model")
                        if quantized_metrics.get('Accuracy'):
                            st.metric("Accuracy", f"{quantized_metrics['Accuracy']*100:.2f}%")
                        st.metric("Latency", f"{quantized_metrics['Latency (ms)']:.3f} ms")
                        st.metric("Model Size", f"{quantized_metrics['param_MB']:.2f} MB")
                    
                    # Improvements
                    if baseline_metrics.get('Accuracy') and quantized_metrics.get('Accuracy'):
                        improvements = calculate_improvements(baseline_metrics, quantized_metrics)
                        st.subheader("📈 Improvements")
                        imp_col1, imp_col2, imp_col3 = st.columns(3)
                        
                        if "Accuracy" in improvements:
                            with imp_col1:
                                acc_imp = improvements["Accuracy"]
                                delta = f"{acc_imp['relative']:+.2f}%"
                                st.metric("Accuracy Change", f"{acc_imp['absolute']*100:+.2f}%", delta=delta)
                        
                        if "Latency" in improvements:
                            with imp_col2:
                                lat_imp = improvements["Latency"]
                                delta = f"{lat_imp['relative']:+.2f}%"
                                st.metric("Latency Change", f"{lat_imp['absolute']:+.3f} ms", delta=delta)
                        
                        if "Model Size" in improvements:
                            with imp_col3:
                                size_imp = improvements["Model Size"]
                                delta = f"{size_imp['relative']:+.2f}%"
                                st.metric("Size Reduction", f"{size_imp['absolute']:.2f} MB", delta=delta)
                    
                    # Export/Download buttons
                    st.subheader("💾 Export Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Export as CSV
                        comparison_df = pd.DataFrame({
                            "Baseline (FP32)": baseline_metrics,
                            "Quantized Model": quantized_metrics
                        })
                        csv = comparison_df.to_csv(index=True)
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv,
                            file_name=f"comparison_baseline_vs_quantized_{metadata.architecture}.csv",
                            mime="text/csv",
                            key="download_csv_baseline_manual"
                        )
                    
                    with col2:
                        # Export as JSON
                        export_data = {
                            "baseline": baseline_metrics,
                            "quantized": quantized_metrics,
                            "improvements": calculate_improvements(baseline_metrics, quantized_metrics) if baseline_metrics.get('Accuracy') and quantized_metrics.get('Accuracy') else {}
                        }
                        json_str = json.dumps(export_data, indent=2)
                        st.download_button(
                            label="📥 Download JSON",
                            data=json_str,
                            file_name=f"comparison_baseline_vs_quantized_{metadata.architecture}.json",
                            mime="application/json",
                            key="download_json_baseline_manual"
                        )
            else:
                baseline_metadata, baseline_model = baseline_result
                st.success(f"✅ Found baseline: {baseline_metadata.architecture} ({baseline_metadata.model_key})")
                
                # Show comparison option AFTER initial evaluation
                # Check if comparison already done
                if st.session_state.ib_comparison_results and st.session_state.ib_comparison_type == "baseline_comparison":
                    # Display stored comparison
                    baseline_metrics = st.session_state.ib_comparison_results["baseline_metrics"]
                    quantized_metrics = st.session_state.ib_comparison_results["quantized_metrics"]
                    baseline_metadata = st.session_state.ib_comparison_results.get("baseline_metadata")
                    
                    st.subheader("📊 Comparison: Baseline vs Quantized")
                else:
                    st.markdown("**Would you like to compare with the baseline?**")
                    if st.button("📊 Compare with Baseline", key="compare_with_baseline_a"):
                        # Evaluate baseline
                        st.info("Evaluating baseline model...")
                        baseline_metrics = evaluate_model_with_metadata(
                            baseline_model, baseline_metadata, eval_samples, latency_runs, latency_warmup, device
                        )
                        
                        # Use already evaluated quantized metrics
                        quantized_metrics = initial_metrics
                        
                        # Store in session state
                        st.session_state.ib_comparison_results = {
                            "baseline_metrics": baseline_metrics,
                            "quantized_metrics": quantized_metrics,
                            "baseline_metadata": baseline_metadata,
                            "quantized_metadata": metadata
                        }
                        st.session_state.ib_comparison_type = "baseline_comparison"
                        st.rerun()
                    
                    # If no comparison yet, stop here
                    if not st.session_state.ib_comparison_results:
                        st.stop()
                    
                    # Get stored results
                    baseline_metrics = st.session_state.ib_comparison_results["baseline_metrics"]
                    quantized_metrics = st.session_state.ib_comparison_results["quantized_metrics"]
                    baseline_metadata = st.session_state.ib_comparison_results.get("baseline_metadata")
                    
                    st.subheader("📊 Comparison: Baseline vs Quantized")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("### Baseline (FP32)")
                        if baseline_metrics.get('Accuracy'):
                            st.metric("Accuracy", f"{baseline_metrics['Accuracy']*100:.2f}%")
                        st.metric("Latency", f"{baseline_metrics['Latency (ms)']:.3f} ms")
                        st.metric("Model Size", f"{baseline_metrics['param_MB']:.2f} MB")
                    
                    with col2:
                        st.markdown("### Quantized Model")
                        if quantized_metrics.get('Accuracy'):
                            st.metric("Accuracy", f"{quantized_metrics['Accuracy']*100:.2f}%")
                        st.metric("Latency", f"{quantized_metrics['Latency (ms)']:.3f} ms")
                        st.metric("Model Size", f"{quantized_metrics['param_MB']:.2f} MB")
                    
                    # Improvements
                    if baseline_metrics.get('Accuracy') and quantized_metrics.get('Accuracy'):
                        improvements = calculate_improvements(baseline_metrics, quantized_metrics)
                        st.subheader("📈 Improvements")
                        imp_col1, imp_col2, imp_col3 = st.columns(3)
                        
                        if "Accuracy" in improvements:
                            with imp_col1:
                                acc_imp = improvements["Accuracy"]
                                delta = f"{acc_imp['relative']:+.2f}%"
                                st.metric("Accuracy Change", f"{acc_imp['absolute']*100:+.2f}%", delta=delta)
                        
                        if "Latency" in improvements:
                            with imp_col2:
                                lat_imp = improvements["Latency"]
                                delta = f"{lat_imp['relative']:+.2f}%"
                                st.metric("Latency Change", f"{lat_imp['absolute']:+.3f} ms", delta=delta)
                        
                        if "Model Size" in improvements:
                            with imp_col3:
                                size_imp = improvements["Model Size"]
                                delta = f"{size_imp['relative']:+.2f}%"
                                st.metric("Size Reduction", f"{size_imp['absolute']:.2f} MB", delta=delta)
                    
                    # Comparison charts
                    charts = create_comparison_chart(
                        baseline_metrics, quantized_metrics, 
                        "Baseline (FP32)", "Quantized Model"
                    )
                    if charts:
                        for chart in charts:
                            st.plotly_chart(chart, width='stretch')
                    
                    # Comparison table
                    st.subheader("Detailed Comparison")
                    comparison_df = pd.DataFrame({
                        "Baseline (FP32)": baseline_metrics,
                        "Quantized Model": quantized_metrics
                    })
                    st.dataframe(comparison_df, width='stretch')
                    
                    # Export/Download buttons
                    st.subheader("💾 Export Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Export as CSV
                        csv = comparison_df.to_csv(index=True)
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv,
                            file_name=f"comparison_baseline_vs_quantized_{metadata.architecture}.csv",
                            mime="text/csv",
                            key="download_csv_baseline"
                        )
                    
                    with col2:
                        # Export as JSON
                        export_data = {
                            "baseline": baseline_metrics,
                            "quantized": quantized_metrics,
                            "improvements": calculate_improvements(baseline_metrics, quantized_metrics) if baseline_metrics.get('Accuracy') and quantized_metrics.get('Accuracy') else {}
                        }
                        json_str = json.dumps(export_data, indent=2)
                        st.download_button(
                            label="📥 Download JSON",
                            data=json_str,
                            file_name=f"comparison_baseline_vs_quantized_{metadata.architecture}.json",
                            mime="application/json",
                            key="download_json_baseline"
                        )
        
        elif workflow == OrchestratorWorkflowType.BASELINE_TO_QUANTIZED:
            st.subheader("📊 Workflow B: Baseline Model → Find Best Quantization")
            st.success("✅ **Detected**: You uploaded a baseline FP32 model!")
            st.markdown("""
            <div style='background-color: #e8f5e9; padding: 15px; border-radius: 5px; border-left: 4px solid #4CAF50; margin-bottom: 20px;'>
            <p style='color: #2e7d32; margin: 0;'>We'll automatically find all available quantized variants and rank them by performance.
            This will help you choose the best quantization strategy for your model.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Find quantized variants
            variants = orchestrator.find_quantized_variants(metadata, str(device))
            
            if not variants:
                st.warning("⚠️ No quantized variants found for this baseline.")
                st.write("""
                **Options:**
                1. Generate quantized variants using the quantization scripts
                2. Continue with single model evaluation
                """)
                
                # Option to find quantized variants
                # Check if comparison already done
                if st.session_state.ib_comparison_results and st.session_state.ib_comparison_type == "variants_comparison":
                    # Display stored comparison
                    baseline_metrics = st.session_state.ib_comparison_results["baseline_metrics"]
                    variant_results = st.session_state.ib_comparison_results["variant_results"]
                    
                    st.subheader("📊 Quantization Strategy Comparison")
                else:
                    st.markdown("**Would you like to find and compare quantized variants?**")
                    if st.button("🔍 Find Quantized Variants", key="find_variants_manual"):
                        # Try to find variants again
                        variants = orchestrator.find_quantized_variants(metadata, str(device))
                        if variants:
                            st.success(f"✅ Found {len(variants)} quantized variant(s)")
                            
                            # Evaluate all variants
                            variant_results = []
                            progress_bar = st.progress(0)
                            
                            for i, (variant_metadata, variant_model) in enumerate(variants):
                                progress_bar.progress((i + 1) / (len(variants) + 1))
                                st.info(f"Evaluating variant {i+1}/{len(variants)}: {variant_metadata.quantization_method.value}")
                                
                                variant_metrics = evaluate_model_with_metadata(
                                    variant_model, variant_metadata, eval_samples, latency_runs, latency_warmup, device
                                )
                                
                                variant_results.append({
                                    "metadata": variant_metadata,
                                    "metrics": variant_metrics,
                                    "model": variant_model
                                })
                            
                            progress_bar.empty()
                            
                            # Use already evaluated baseline metrics
                            baseline_metrics = initial_metrics
                            
                            # Store in session state
                            st.session_state.ib_comparison_results = {
                                "baseline_metrics": baseline_metrics,
                                "variant_results": variant_results,
                                "baseline_metadata": metadata
                            }
                            st.session_state.ib_comparison_type = "variants_comparison"
                            st.rerun()
                        else:
                            st.error("❌ Could not find quantized variants for comparison.")
                            st.stop()
                    
                    # If no comparison yet, stop here
                    if not st.session_state.ib_comparison_results:
                        st.stop()
                    
                    # Get stored results
                    baseline_metrics = st.session_state.ib_comparison_results["baseline_metrics"]
                    variant_results = st.session_state.ib_comparison_results["variant_results"]
                    
                    st.subheader("📊 Quantization Strategy Comparison")
                    
                    # Create comparison table
                    comparison_data = {
                        "Model": ["Baseline (FP32)"] + [
                            f"{v['metadata'].quantization_method.value.upper()}" 
                            for v in variant_results
                        ]
                    }
                    
                    if baseline_metrics.get('Accuracy'):
                        comparison_data["Accuracy (%)"] = [baseline_metrics['Accuracy']*100] + [
                            v['metrics'].get('Accuracy', 0)*100 if v['metrics'].get('Accuracy') else None
                            for v in variant_results
                        ]
                    
                    comparison_data["Latency (ms)"] = [baseline_metrics['Latency (ms)']] + [
                        v['metrics']['Latency (ms)'] for v in variant_results
                    ]
                    
                    comparison_data["Size (MB)"] = [baseline_metrics['param_MB']] + [
                        v['metrics']['param_MB'] for v in variant_results
                    ]
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, width='stretch')
                    
                    # Rankings
                    st.subheader("🏆 Rankings")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    # Fastest inference
                    with col1:
                        st.markdown("### ⚡ Fastest Inference")
                        sorted_by_latency = sorted(
                            variant_results,
                            key=lambda x: x['metrics']['Latency (ms)']
                        )
                        for i, v in enumerate(sorted_by_latency[:3], 1):
                            st.write(f"{i}. {v['metadata'].quantization_method.value.upper()}: {v['metrics']['Latency (ms)']:.3f} ms")
                    
                    # Lowest memory
                    with col2:
                        st.markdown("### 💾 Lowest Memory")
                        sorted_by_size = sorted(
                            variant_results,
                            key=lambda x: x['metrics']['param_MB']
                        )
                        for i, v in enumerate(sorted_by_size[:3], 1):
                            st.write(f"{i}. {v['metadata'].quantization_method.value.upper()}: {v['metrics']['param_MB']:.2f} MB")
                    
                    # Best accuracy retention
                    with col3:
                        st.markdown("### 🎯 Best Accuracy Retention")
                        if baseline_metrics.get('Accuracy'):
                            sorted_by_accuracy = sorted(
                                variant_results,
                                key=lambda x: x['metrics'].get('Accuracy', 0) if x['metrics'].get('Accuracy') else 0,
                                reverse=True
                            )
                            for i, v in enumerate(sorted_by_accuracy[:3], 1):
                                acc = v['metrics'].get('Accuracy', 0)
                                if acc:
                                    acc_retention = (acc / baseline_metrics['Accuracy']) * 100
                                    st.write(f"{i}. {v['metadata'].quantization_method.value.upper()}: {acc_retention:.1f}%")
                    
                    # Export/Download buttons
                    st.subheader("💾 Export Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Export comparison table as CSV
                        comparison_data = {
                            "Model": ["Baseline (FP32)"] + [
                                f"{v['metadata'].quantization_method.value.upper()}" 
                                for v in variant_results
                            ]
                        }
                        
                        if baseline_metrics.get('Accuracy'):
                            comparison_data["Accuracy (%)"] = [baseline_metrics['Accuracy']*100] + [
                                v['metrics'].get('Accuracy', 0)*100 if v['metrics'].get('Accuracy') else None
                                for v in variant_results
                            ]
                        
                        comparison_data["Latency (ms)"] = [baseline_metrics['Latency (ms)']] + [
                            v['metrics']['Latency (ms)'] for v in variant_results
                        ]
                        
                        comparison_data["Size (MB)"] = [baseline_metrics['param_MB']] + [
                            v['metrics']['param_MB'] for v in variant_results
                        ]
                        
                        export_df = pd.DataFrame(comparison_data)
                        csv = export_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv,
                            file_name=f"quantization_comparison_{metadata.architecture}.csv",
                            mime="text/csv",
                            key="download_csv_variants_manual"
                        )
                    
                    with col2:
                        # Export as JSON
                        export_data = {
                            "baseline": baseline_metrics,
                            "variants": [
                                {
                                    "method": v['metadata'].quantization_method.value,
                                    "metrics": v['metrics']
                                }
                                for v in variant_results
                            ]
                        }
                        json_str = json.dumps(export_data, indent=2)
                        st.download_button(
                            label="📥 Download JSON",
                            data=json_str,
                            file_name=f"quantization_comparison_{metadata.architecture}.json",
                            mime="application/json",
                            key="download_json_variants_manual"
                        )
            else:
                st.success(f"✅ Found {len(variants)} quantized variant(s)")
                
                # Show comparison option AFTER initial evaluation
                st.markdown("**Would you like to compare with quantized variants?**")
                if st.button("📊 Compare with Variants", key="compare_with_variants_b"):
                    # Use already evaluated baseline metrics
                    baseline_metrics = initial_metrics
                    
                    # Evaluate all variants
                    variant_results = []
                    progress_bar = st.progress(0)
                    
                    for i, (variant_metadata, variant_model) in enumerate(variants):
                        progress_bar.progress((i + 1) / (len(variants) + 1))
                        st.info(f"Evaluating variant {i+1}/{len(variants)}: {variant_metadata.quantization_method.value}")
                        
                        variant_metrics = evaluate_model_with_metadata(
                            variant_model, variant_metadata, eval_samples, latency_runs, latency_warmup, device
                        )
                        
                        variant_results.append({
                            "metadata": variant_metadata,
                            "metrics": variant_metrics,
                            "model": variant_model
                        })
                    
                    progress_bar.empty()
                    
                    # Rank variants
                    st.subheader("📊 Quantization Strategy Comparison")
                    
                    # Create comparison table
                    comparison_data = {
                        "Model": ["Baseline (FP32)"] + [
                            f"{v['metadata'].quantization_method.value.upper()}" 
                            for v in variant_results
                        ]
                    }
                    
                    if baseline_metrics.get('Accuracy'):
                        comparison_data["Accuracy (%)"] = [baseline_metrics['Accuracy']*100] + [
                            v['metrics'].get('Accuracy', 0)*100 if v['metrics'].get('Accuracy') else None
                            for v in variant_results
                        ]
                    
                    comparison_data["Latency (ms)"] = [baseline_metrics['Latency (ms)']] + [
                        v['metrics']['Latency (ms)'] for v in variant_results
                    ]
                    
                    comparison_data["Size (MB)"] = [baseline_metrics['param_MB']] + [
                        v['metrics']['param_MB'] for v in variant_results
                    ]
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, width='stretch')
                    
                    # Rankings
                    st.subheader("🏆 Rankings")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    # Fastest inference
                    with col1:
                        st.markdown("### ⚡ Fastest Inference")
                        sorted_by_latency = sorted(
                            variant_results,
                            key=lambda x: x['metrics']['Latency (ms)']
                        )
                        for i, v in enumerate(sorted_by_latency[:3], 1):
                            st.write(f"{i}. {v['metadata'].quantization_method.value.upper()}: {v['metrics']['Latency (ms)']:.3f} ms")
                    
                    # Lowest memory
                    with col2:
                        st.markdown("### 💾 Lowest Memory")
                        sorted_by_size = sorted(
                            variant_results,
                            key=lambda x: x['metrics']['param_MB']
                        )
                        for i, v in enumerate(sorted_by_size[:3], 1):
                            st.write(f"{i}. {v['metadata'].quantization_method.value.upper()}: {v['metrics']['param_MB']:.2f} MB")
                    
                    # Best accuracy retention
                    with col3:
                        st.markdown("### 🎯 Best Accuracy Retention")
                        if baseline_metrics.get('Accuracy'):
                            sorted_by_accuracy = sorted(
                                variant_results,
                                key=lambda x: x['metrics'].get('Accuracy', 0) if x['metrics'].get('Accuracy') else 0,
                                reverse=True
                            )
                            for i, v in enumerate(sorted_by_accuracy[:3], 1):
                                acc = v['metrics'].get('Accuracy', 0)
                                if acc:
                                    acc_retention = (acc / baseline_metrics['Accuracy']) * 100
                                    st.write(f"{i}. {v['metadata'].quantization_method.value.upper()}: {acc_retention:.1f}%")
                    
                    # Recommendation
                    st.subheader("💡 Recommendation")
                    if variant_results:
                        # Simple recommendation: best balance of speed and accuracy
                        best_variant = max(
                            variant_results,
                            key=lambda x: (
                                (x['metrics'].get('Accuracy', 0) / baseline_metrics.get('Accuracy', 1)) * 0.5 +
                                (baseline_metrics['Latency (ms)'] / x['metrics']['Latency (ms)']) * 0.5
                                if baseline_metrics.get('Accuracy') and x['metrics'].get('Accuracy')
                                else baseline_metrics['Latency (ms)'] / x['metrics']['Latency (ms)']
                            )
                        )
                        st.success(
                            f"**Recommended**: {best_variant['metadata'].quantization_method.value.upper()} "
                            f"for best balance of accuracy retention and speedup"
                        )
                    
                    # Export/Download buttons
                    st.subheader("💾 Export Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Export comparison table as CSV
                        comparison_data = {
                            "Model": ["Baseline (FP32)"] + [
                                f"{v['metadata'].quantization_method.value.upper()}" 
                                for v in variant_results
                            ]
                        }
                        
                        if baseline_metrics.get('Accuracy'):
                            comparison_data["Accuracy (%)"] = [baseline_metrics['Accuracy']*100] + [
                                v['metrics'].get('Accuracy', 0)*100 if v['metrics'].get('Accuracy') else None
                                for v in variant_results
                            ]
                        
                        comparison_data["Latency (ms)"] = [baseline_metrics['Latency (ms)']] + [
                            v['metrics']['Latency (ms)'] for v in variant_results
                        ]
                        
                        comparison_data["Size (MB)"] = [baseline_metrics['param_MB']] + [
                            v['metrics']['param_MB'] for v in variant_results
                        ]
                        
                        export_df = pd.DataFrame(comparison_data)
                        csv = export_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv,
                            file_name=f"quantization_comparison_{metadata.architecture}.csv",
                            mime="text/csv",
                            key="download_csv_variants"
                        )
                    
                    with col2:
                        # Export as JSON
                        export_data = {
                            "baseline": baseline_metrics,
                            "variants": [
                                {
                                    "method": v['metadata'].quantization_method.value,
                                    "metrics": v['metrics']
                                }
                                for v in variant_results
                            ]
                        }
                        json_str = json.dumps(export_data, indent=2)
                        st.download_button(
                            label="📥 Download JSON",
                            data=json_str,
                            file_name=f"quantization_comparison_{metadata.architecture}.json",
                            mime="application/json",
                            key="download_json_variants"
                        )
        
        else:
            st.warning("⚠️ Could not determine workflow. Model type may be unsupported.")
            st.markdown("""
            <div style='background-color: #fff3e0; padding: 15px; border-radius: 5px; border-left: 4px solid #FF9800; margin-bottom: 20px;'>
            <p style='color: #E65100; margin: 0;'><strong>Note:</strong> The model has been evaluated above. If you'd like to compare it with other models,
            you can use the "Compare Models" mode or manually upload comparison models.</p>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    st.write("Run with: streamlit run ui/app.py")
