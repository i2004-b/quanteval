# ui/app.py
# Run: streamlit run ui/app.py
import streamlit as st
import warnings
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path so `ui` package imports resolve when
# running `streamlit run ui/app.py` or executing the file directly.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import os
import torch
import pandas as pd
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from transformers import DistilBertTokenizer
from datasets import load_dataset
from ui.model_detection import (
    load_generic_pytorch_model,
    ModelProfile,
    detect_architecture_type,
)
from ui.model_loader import load_model, load_hf_model_by_name
import json
import time
import traceback

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False
from ui.model_inspection import (
    inspect_model,
    inspect_model_from_registry,
    ModelMetadata,
    ModelSource
)
from ui.baseline_registry import get_baseline_registry
from ui.comparison_orchestrator import get_comparison_orchestrator, ComparisonGroup
from ui.explanations import beginner_conclusion
from ui.recommendation import recommend_comparison_target
from ui.model_analysis import ModelAnalysisResult, analyze_uploaded_model as analyze_uploaded_model_analysis, analyze_builtin_model
from ui.comparison_planner import (
    EntryPath,
    ComparisonIntent,
    ComparisonPlan,
    plan_learn,
    plan_upload,
    get_builtin_architectures,
)
from eval.metrics import top1, f1
from eval.latency import measure_latency_s
from eval.memory import param_bytes, peak_gpu_mem_once, on_disk_bytes_state_dict
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
        #"int8 (static)": "distilbert_static_int8", #doesnt exist yet
        "QAT int8": "distilbert_qat_int8",
    },
}

# ==========================================================
# STREAMLIT
# ==========================================================
st.set_page_config(page_title="Quanteval — Model Evaluation", layout="wide", page_icon="🧠")

# Session state for entry path (Learn vs Upload)
if "entry_path" not in st.session_state:
    st.session_state.entry_path = None  # None | "learn" | "upload"
if "comparison_plan" not in st.session_state:
    st.session_state.comparison_plan = None  # ComparisonPlan when set
if "presentation_mode" not in st.session_state:
    st.session_state.presentation_mode = "Beginner"  # for Path B only

# ==========================================================
# ENTRY SCREEN — two choices drive all subsequent logic
# ==========================================================
if st.session_state.entry_path is None:
    st.title("🧠 Quanteval — Model Evaluation")
    st.markdown("""
    **A comprehensive quantization benchmarking framework for PyTorch models.**

    Choose how you want to explore:
    """)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📚 **Learn about quantization**", use_container_width=True, key="entry_learn"):
            st.session_state.entry_path = "learn"
            st.rerun()
        st.caption("Use built-in models to see how quantization affects accuracy, latency, and size. Guided, no setup.")
    with col2:
        if st.button("📤 **Upload your own model to explore quantization**", use_container_width=True, key="entry_upload"):
            st.session_state.entry_path = "upload"
            st.rerun()
        st.caption("Upload a model; we analyze it and suggest comparisons to baselines or quantized variants.")
    st.markdown("---")
    st.info("💡 The **uploaded or selected model** determines what comparisons are performed automatically.")
    st.stop()

# Past entry screen: we have entry_path in ["learn", "upload"]
entry_path = st.session_state.entry_path
st.sidebar.header("⚙️ Evaluation Options")
if st.sidebar.button("🏠 Back to start", key="back_to_start"):
    st.session_state.entry_path = None
    st.session_state.comparison_plan = None
    for key in list(st.session_state.keys()):
        if key.startswith("ib_") or key.startswith("entry_") or key == "comparison_plan" or key == "presentation_mode":
            try:
                del st.session_state[key]
            except Exception:
                pass
    st.rerun()

# Utility: full session reset (helps clear stale widget IDs between code edits)
if st.sidebar.button("⚠️ Clear UI session state (full reset)", key="clear_full_session"):
    for k in list(st.session_state.keys()):
        try:
            del st.session_state[k]
        except Exception:
            pass
    # Try the public rerun API, otherwise fall back to raising the internal RerunException.
    try:
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
        elif hasattr(st, "rerun"):
            st.rerun()
        else:
            raise AttributeError("no rerun API on st")
    except Exception:
        try:
            # Internal API used by Streamlit to trigger a rerun
            from streamlit.runtime.scriptrunner.script_runner import RerunException
            raise RerunException()
        except Exception:
            # Last resort: stop execution so the UI refreshes on the next interaction
            st.stop()

# Path B only: presentation mode (Beginner vs Advanced)
if entry_path == "upload":
    user_mode = st.sidebar.radio(
        "👤 Presentation:",
        ["Beginner Mode", "Advanced Mode"],
        help="Beginner: guided demo with explanations. Advanced: manual selection, raw metrics, multi-model comparison.",
        key="presentation_radio",
    )
    st.session_state.presentation_mode = "Beginner" if user_mode == "Beginner Mode" else "Advanced"
else:
    user_mode = "Beginner Mode"  # Path A is beginner-only

# Routing: Path A (learn) vs Path B (upload). Advanced workflows only for Path B.
internal_mode = None
if entry_path == "learn":
    internal_mode = "Path A: Learn"
elif entry_path == "upload":
    # Path B: user can choose Advanced workflows (Single/Compare/Multi) or stay in planner-driven flow
    if user_mode == "Advanced Mode":
        internal_mode = st.sidebar.radio(
            "📊 Advanced Workflow:",
            ["Planner-driven (upload)", "Compare to Built-in", "Multi-Model (Advanced)"],
            help="Planner-driven uses analysis to pick comparisons; others are manual.",
            key="advanced_workflow",
        )
        if internal_mode == "Planner-driven (upload)":
            internal_mode = "Path B: Upload"
        # else: Compare to Built-in, Multi-Model
    else:
        internal_mode = "Path B: Upload"

# Defensive defaults so static analysers (Pylance) don't report undefined variables
# These values are overwritten by specific UI branches when needed.
device = torch.device("cpu")
eval_samples = 200
latency_runs = 20
latency_warmup = 5
model_type = None
variant = None

# ==========================================================
# MODEL SELECTION / USER UPLOAD (Advanced workflows only)
# Path A (Learn) and Path B (Upload) use planner; no selection here.
# ==========================================================
if internal_mode == "Path A: Learn":
    # No model selection in sidebar; plan_learn drives architecture + baseline + variants
    use_generic_loader = False
    model_key = None
    user_model_path = None
    baseline_key = None
    quantized_key = None
elif internal_mode == "Path B: Upload":
    # Upload + planner; model selection is in Path B block
    use_generic_loader = False
    model_key = None
    user_model_path = None
    baseline_key = None
    quantized_key = None
elif internal_mode == "Single Model":
    # Single Model option removed — planner-driven covers this flow
    # Keep variables initialized to avoid NameError in other branches
    use_generic_loader = False
    model_key = None
    user_model_path = None
    baseline_key = None
    quantized_key = None



# ==========================================================
# IMPORTED MODEL PROFILING DISPLAY
# ==========================================================
def display_model_profile(profile: ModelProfile):
    """Display model profile in Streamlit UI."""
    st.subheader("🔍 Model Profile")
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    # Support passing either a ModelProfile instance or a plain dict (from session)
    if hasattr(profile, 'to_dict'):
        p_dict = profile.to_dict()
        arch = getattr(profile, 'arch_type', p_dict.get('architecture_type', 'Unknown'))
        params = getattr(profile, 'param_counts', {}).get('total', p_dict.get('total_parameters', 0))
        size_mb = getattr(profile, 'size_fp32', {}).get('size_mb', p_dict.get('size_estimates', {}).get('fp32_mb', None))
        depth = getattr(profile, 'depth', p_dict.get('model_depth', None))
    elif isinstance(profile, dict):
        p_dict = profile
        arch = p_dict.get('architecture_type', 'Unknown')
        params = p_dict.get('total_parameters', 0)
        size_mb = p_dict.get('size_estimates', {}).get('fp32_mb', None)
        depth = p_dict.get('model_depth', None)
    else:
        # Unknown profile type
        arch = 'Unknown'
        params = 0
        size_mb = None
        depth = None

    with col1:
        st.metric("Architecture", arch)
    with col2:
        st.metric("Parameters", f"{params:,}")
    with col3:
        st.metric("FP32 Size", f"{size_mb:.1f} MB" if size_mb is not None else "N/A")
    with col4:
        st.metric("Model Depth", depth if depth is not None else "N/A")
    
    # Expandable detailed info
    with st.expander("📊 Detailed Profile Information"):
        # Prefer the dict representation when available (p_dict set earlier), else fall back
        profile_dict_local = None
        try:
            if 'p_dict' in locals() and p_dict is not None:
                profile_dict_local = p_dict
            elif hasattr(profile, 'to_dict'):
                profile_dict_local = profile.to_dict()
        except Exception:
            profile_dict_local = None

        # Size estimates
        st.write("**Size Estimates:**")
        size_estimates = None
        if isinstance(profile_dict_local, dict):
            size_estimates = profile_dict_local.get('size_estimates')
        else:
            size_estimates = getattr(profile, 'size_estimates', None)

        if size_estimates:
            try:
                size_df = pd.DataFrame([size_estimates]).T
                size_df.columns = ["Size (MB)"]
                st.dataframe(size_df, width='stretch')
            except Exception:
                st.write(size_estimates)
        else:
            st.write("N/A")

        # Layer composition (top 10)
        st.write("**Layer Composition (Top 10):**")
        layer_comp = None
        if isinstance(profile_dict_local, dict):
            layer_comp = profile_dict_local.get('layer_composition')
        else:
            layer_comp = getattr(profile, 'layer_composition', None)

        if layer_comp:
            try:
                lc = dict(list(layer_comp.items())[:10])
                layer_df = pd.DataFrame(list(lc.items()), columns=["Layer Type", "Count"])
                st.dataframe(layer_df, width='stretch')
            except Exception:
                st.write(layer_comp)
        else:
            st.write("N/A")

        # I/O shapes
        input_shapes = profile_dict_local.get('input_shapes') if isinstance(profile_dict_local, dict) else getattr(profile, 'input_shapes', None)
        output_shapes = profile_dict_local.get('output_shapes') if isinstance(profile_dict_local, dict) else getattr(profile, 'output_shapes', None)

        if input_shapes:
            st.write("**Input Shapes:**")
            st.json(input_shapes)
        if output_shapes:
            st.write("**Output Shapes:**")
            st.json(output_shapes)

        # FLOPs
        flops_val = profile_dict_local.get('flops') if isinstance(profile_dict_local, dict) else getattr(profile, 'flops', None)
        if flops_val:
            try:
                st.write(f"**FLOPs:** {flops_val:,.0f}")
            except Exception:
                st.write(f"**FLOPs:** {flops_val}")

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
    # Fallback: some quantized models report zero-sized parameters; use state_dict on-disk size
    if param_mb == 0:
        try:
            param_mb = on_disk_bytes_state_dict(model) / 1e6
        except Exception:
            pass

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
    # Fallback for quantized models
    if param_mb == 0:
        try:
            param_mb = on_disk_bytes_state_dict(model) / 1e6
        except Exception:
            pass

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

    # Accept either a ModelMetadata instance or a plain dict (from session)
    if hasattr(metadata, 'to_dict'):
        m_dict = metadata.to_dict()
        arch = metadata.architecture
        task = metadata.task.value.replace("_", " ").title()
        dataset = metadata.dataset.value.upper() if metadata.dataset.value != "unknown" else "Unknown"
        precision = metadata.precision.value.upper()
    elif isinstance(metadata, dict):
        m_dict = metadata
        arch = m_dict.get('architecture', 'Unknown')
        task = m_dict.get('task', 'unknown').replace("_", " ").title()
        dataset = m_dict.get('dataset', 'unknown').upper() if m_dict.get('dataset', 'unknown') != 'unknown' else 'Unknown'
        precision = m_dict.get('precision', 'UNKNOWN').upper()
    else:
        m_dict = {}
        arch = 'Unknown'
        task = 'Unknown'
        dataset = 'Unknown'
        precision = 'UNKNOWN'

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Architecture", arch)
    with col2:
        st.metric("Task", task)
    with col3:
        st.metric("Dataset", dataset)
    with col4:
        st.metric("Precision", precision)

    # Quantization info
    with st.expander("📋 Detailed Metadata"):
        st.json(m_dict)
        try:
            # metadata may be a dict without helper methods
            is_quant = False
            is_base = False
            if hasattr(metadata, 'is_quantized'):
                is_quant = metadata.is_quantized()
            elif isinstance(metadata, dict):
                is_quant = metadata.get('quantization_method') not in (None, 'none', 'unknown')

            if hasattr(metadata, 'is_baseline'):
                is_base = metadata.is_baseline()
            elif isinstance(metadata, dict):
                is_base = metadata.get('precision') == 'fp32' and metadata.get('quantization_method') in (None, 'none')

            if is_quant:
                qname = m_dict.get('quantization_method', 'unknown').upper()
                st.info(f"✅ **Quantized Model Detected**: {qname}")
            elif is_base:
                st.info("✅ **Baseline FP32 Model Detected**")
        except Exception:
            pass


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


# Single Model evaluation removed — planner-driven and Multi-Model cover needed workflows.
# This block was intentionally removed to simplify the Advanced UX and avoid duplicate paths.

    
# Setup UI for compare-to-built-in: select baseline architecture/variant and upload one model
if internal_mode == "Compare to Built-in":
    model_type = st.sidebar.selectbox("Baseline Model Architecture", list(MODEL_REGISTRY.keys()), key="compare_upload_baseline_arch")
    variants = list(MODEL_REGISTRY[model_type].keys())
    baseline_variant = st.sidebar.selectbox("Baseline Model", variants, index=0, key="compare_upload_baseline_variant")
    baseline_key = MODEL_REGISTRY[model_type][baseline_variant]

    st.sidebar.write("---")
    st.sidebar.write("Upload model to compare:")
    uploaded_file = st.sidebar.file_uploader("Upload model file", type=["pt", "pth"], key="compare_upload")
    user_model_path = None
    if uploaded_file:
        os.makedirs("uploads", exist_ok=True)
        temp_path = os.path.join("uploads", uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        user_model_path = temp_path
        # Immediately profile uploaded model
        try:
            m, load_info = load_generic_pytorch_model(user_model_path, str(device))
            if load_info.get("success"):
                st.info("Profiling uploaded model...")
                p = ModelProfile(m, str(device))
                display_model_profile(p)
                # keep in session for reuse
                st.session_state.compare_uploaded_model = m
                st.session_state.compare_uploaded_path = user_model_path
                # persist profile so we can show it across reruns without recomputing
                try:
                    st.session_state.compare_uploaded_profile = p.to_dict()
                except Exception:
                    st.session_state.compare_uploaded_profile = None
            elif load_info.get("model_type") == "state_dict_only":
                st.warning("Uploaded file appears to be a state-dict-only checkpoint (no architecture).")
                st.info(f"Parameters (from checkpoint): {load_info.get('state_dict_param_count')}\nSize (MB): {load_info.get('state_dict_size_mb'):.2f}")
                # Save load_info so we can explain to the user later; we cannot run inference.
                st.session_state.compare_uploaded_path = user_model_path
                st.session_state.compare_uploaded_load_info = load_info
                # persist a minimal profile-like dict so UI can show the size/params
                st.session_state.compare_uploaded_profile = {
                    'architecture_type': 'Unknown',
                    'total_parameters': load_info.get('state_dict_param_count', 0),
                    'size_estimates': {'fp32_mb': load_info.get('state_dict_size_mb', 0.0)},
                }
            else:
                st.warning(f"Upload loaded with warnings: {load_info.get('warnings')}")
        except Exception as e:
            st.warning(f"Failed to load uploaded model for profiling: {e}")

    if st.button("Compare Models"):
        # Upload vs Built-in only
        if not user_model_path and not st.session_state.get('compare_uploaded_path'):
            st.error("Please upload a model file to compare.")
            st.stop()
        # Load baseline
        st.info(f"Loading baseline: {baseline_variant}...")
        try:
            baseline_model = load_model(baseline_key, device)
            st.success(f"✅ Baseline model loaded: {baseline_variant}")
        except Exception as e:
            st.error(f"Error loading baseline model: {e}")
            st.stop()

        st.info("Evaluating baseline model...")
        baseline_md = inspect_model_from_registry(baseline_key, baseline_model, str(device))
        if baseline_md.architecture == "ResNet18":
            baseline_metrics = evaluate_cifar10_model(baseline_model, eval_samples, latency_runs, latency_warmup, device)
        elif baseline_md.architecture == "DistilBERT":
            baseline_metrics = evaluate_sst2_model(baseline_model, eval_samples, latency_runs, latency_warmup, device)
        else:
            st.error("Unsupported baseline architecture.")
            st.stop()

        # Load uploaded model (reuse profiled if available)
        if st.session_state.get('compare_uploaded_model'):
            uploaded_model = st.session_state.compare_uploaded_model
            uploaded_path = st.session_state.compare_uploaded_path
            # if we have a stored profile dict, display it so it persists
            if st.session_state.get('compare_uploaded_profile'):
                display_model_profile(st.session_state.compare_uploaded_profile)
        else:
            # If earlier upload was a state-dict-only checkpoint we cannot run full benchmarking
            if st.session_state.get('compare_uploaded_load_info') and st.session_state['compare_uploaded_load_info'].get('model_type') == 'state_dict_only':
                st.error("Uploaded file is a state-dict-only checkpoint (no architecture). Full benchmarking (inference/latency) is not possible.")
                st.info("Options: upload a scripted/traced model (.pt), provide the model code/architecture, or export the model with the architecture included.")
                st.stop()

            uploaded_model, load_info = load_generic_pytorch_model(user_model_path, str(device))
            if not load_info.get('success'):
                st.error(f"Failed to load uploaded model: {load_info.get('error')}")
                st.stop()

        st.info("Evaluating uploaded model...")
        uploaded_profile = ModelProfile(uploaded_model, str(device))
        # Inspect uploaded model metadata so we can display detected info persistently
        try:
            uploaded_metadata = inspect_model(uploaded_model, source=ModelSource.USER, file_path=uploaded_path, device=str(device))
            # persist metadata for reuse
            st.session_state.compare_uploaded_metadata = uploaded_metadata.to_dict()
        except Exception:
            uploaded_metadata = None
            st.session_state.compare_uploaded_metadata = None

        display_model_profile(uploaded_profile)
        if "ResNet" in uploaded_profile.arch_type:
            uploaded_metrics = evaluate_cifar10_model(uploaded_model, eval_samples, latency_runs, latency_warmup, device)
        elif "DistilBERT" in uploaded_profile.arch_type:
            uploaded_metrics = evaluate_sst2_model(uploaded_model, eval_samples, latency_runs, latency_warmup, device)
        else:
            uploaded_metrics = evaluate_generic_model(uploaded_model, uploaded_profile, eval_samples, latency_runs, latency_warmup, device)

        # Display results (reuse existing display code)
        baseline_display_name = f"{baseline_variant} (Baseline)"
        comparison_display_name = f"Uploaded ({uploaded_profile.arch_type})"
        st.subheader(f"📊 Comparison: {baseline_display_name} vs {comparison_display_name}")
        # Show detected metadata for baseline and uploaded models (persisted)
        try:
            col_md_1, col_md_2 = st.columns(2)
            with col_md_1:
                st.markdown("**Baseline detected metadata**")
                display_model_metadata(baseline_md)
            with col_md_2:
                st.markdown("**Uploaded detected metadata**")
                if uploaded_metadata:
                    display_model_metadata(uploaded_metadata)
                elif st.session_state.get('compare_uploaded_profile'):
                    # show minimal persisted profile as metadata-like dict
                    display_model_metadata(st.session_state.get('compare_uploaded_profile'))
        except Exception:
            pass

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"### {baseline_display_name}")
            if baseline_metrics.get('Accuracy') is not None:
                st.metric("Accuracy", f"{baseline_metrics['Accuracy']*100:.2f}%")
            else:
                st.metric("Accuracy", "N/A")
            st.metric("Latency", f"{baseline_metrics['Latency (ms)']:.3f} ms")
            st.metric("Model Size", f"{baseline_metrics['param_MB']:.2f} MB")
        with col2:
            st.markdown(f"### {comparison_display_name}")
            if uploaded_metrics.get('Accuracy') is not None:
                st.metric("Accuracy", f"{uploaded_metrics['Accuracy']*100:.2f}%")
            else:
                st.metric("Accuracy", "N/A")
            st.metric("Latency", f"{uploaded_metrics['Latency (ms)']:.3f} ms")
            st.metric("Model Size", f"{uploaded_metrics['param_MB']:.2f} MB")

        # Log
        log_experiment(baseline_metrics, model_name=f"{model_type}-{baseline_variant}", method="ui-eval-comparison", out_dir="outputs/reports")
        log_experiment(uploaded_metrics, model_name=f"uploaded-{os.path.basename(st.session_state.get('compare_uploaded_path', user_model_path))}", method="ui-eval-comparison", out_dir="outputs/reports")
        st.success("✅ Both experiments logged to outputs/reports")
        

# ==========================================================
# MULTI-MODEL (ADVANCED) MODE
# ==========================================================
elif internal_mode == "Multi-Model (Advanced)":
    st.header("🧪 Multi-Model Benchmarking (Advanced)")
    st.caption("Upload and benchmark multiple models, then compare them manually.")
    # Persistent error list for this multi-model workflow so messages survive reruns
    if "advanced_multi_errors" not in st.session_state:
        st.session_state.advanced_multi_errors = []  # list of dicts: {name,timestamp,error,trace}

    # If there are previously captured errors, show them prominently and persistently
    if st.session_state.advanced_multi_errors:
        with st.expander("⚠️ Recent errors (click to expand). These are persisted across reruns."):
            for err in st.session_state.advanced_multi_errors:
                ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(err.get('timestamp', 0)))
                st.markdown(f"**{err.get('name','(unknown)')}** — {ts}")
                st.error(err.get('error'))
                st.code(err.get('trace',''), language='text')
            if st.button("Clear displayed errors", key="advanced_multi_clear_errors"):
                st.session_state.advanced_multi_errors = []
                # don't rerun automatically; leave it to user
    init_saved_models()
    saved_models = get_saved_models()

    st.sidebar.header("📦 Model Set")
    uploaded_files = st.sidebar.file_uploader(
        "Upload one or more `.pt` / `.pth` model files",
        type=["pt", "pth"],
        accept_multiple_files=True,
        key="advanced_multi_upload",
    )

    # Persist a model set separate from the Beginner 'saved_models' list.
    if "advanced_model_set" not in st.session_state:
        st.session_state.advanced_model_set = {}  # name -> {"path":..., "metadata":..., "metrics":...}

    if uploaded_files:
        if st.sidebar.button("➕ Add uploaded models to set", key="advanced_add_to_set"):
            os.makedirs("uploads", exist_ok=True)
            for f in uploaded_files:
                path = os.path.join("uploads", f.name)
                with open(path, "wb") as out:
                    out.write(f.read())
                st.session_state.advanced_model_set[f.name] = {"path": path, "metadata": None, "metrics": None}
            st.sidebar.success(f"Added {len(uploaded_files)} model(s) to the set.")
            st.rerun()

    # Also allow selecting from already-saved models
    if saved_models:
        chosen_saved = st.sidebar.multiselect(
            "Add from Saved Models",
            options=list(saved_models.keys()),
            default=[],
            key="advanced_add_saved_models",
        )
        if st.sidebar.button("➕ Add selected saved models to set", key="advanced_add_saved_to_set"):
            for name in chosen_saved:
                st.session_state.advanced_model_set[name] = {
                    "path": saved_models[name]["path"],
                    "metadata": saved_models[name].get("metadata"),
                    "metrics": None,
                }
            st.sidebar.success(f"Added {len(chosen_saved)} saved model(s) to the set.")
            st.rerun()

    st.sidebar.write("---")
    st.sidebar.header("⚙️ Evaluation Settings")
    device_choice = st.sidebar.selectbox("Run on", ["cpu", "cuda (if available)"], key="advanced_multi_device")
    device = torch.device("cuda" if (device_choice.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    eval_samples = st.sidebar.number_input("Num eval samples", min_value=32, max_value=5000, value=512, step=32, key="advanced_multi_samples")
    latency_runs = st.sidebar.number_input("Latency runs", min_value=5, max_value=200, value=20, key="advanced_multi_latency_runs")
    latency_warmup = st.sidebar.number_input("Latency warmup", min_value=1, max_value=50, value=5, key="advanced_multi_warmup")

    model_set = st.session_state.advanced_model_set
    if not model_set:
        st.info("Add models to your model set from the sidebar to begin.")
        st.stop()

    st.subheader("📋 Current Model Set")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "name": name,
                    "path": info.get("path"),
                    "has_metadata": info.get("metadata") is not None,
                    "has_metrics": info.get("metrics") is not None,
                }
                for name, info in model_set.items()
            ]
        ),
        width='stretch',
    )

    if st.button("▶️ Run evaluation for model set", key="advanced_multi_eval"):
        from ui.model_detection import load_generic_pytorch_model
        from ui.model_inspection import inspect_model, ModelSource
        for name, info in model_set.items():
            try:
                st.info(f"Loading {name}...")
                model, load_info = load_generic_pytorch_model(info["path"], str(device))
                # Handle state-dict-only checkpoints specially: we can record size/params but cannot run inference
                if not load_info.get("success"):
                    if load_info.get("model_type") == "state_dict_only":
                        st.info(f"{name} appears to be a state-dict-only checkpoint. Recording size/param info.")
                        info["metadata"] = {"architecture": "unknown", "note": "state_dict_only"}
                        info["metrics"] = {
                            "Accuracy": None,
                            "Latency (ms)": None,
                            "param_MB": load_info.get("state_dict_size_mb"),
                            "param_count": load_info.get("state_dict_param_count"),
                        }
                        # Do not attempt to evaluate this model
                        continue
                    else:
                        err_msg = load_info.get('error') or 'unknown load error'
                        st.error(f"Failed to load {name}: {err_msg}")
                        # persist the error across reruns
                        st.session_state.advanced_multi_errors.append({
                            "name": name,
                            "timestamp": time.time(),
                            "error": f"Load failed: {err_msg}",
                            "trace": load_info.get('traceback') or ''
                        })
                        continue

                try:
                    metadata = inspect_model(model, source=ModelSource.USER, file_path=info["path"], device=str(device))
                except Exception as e:
                    tb = traceback.format_exc()
                    st.warning(f"Warning: inspection failed for {name}")
                    st.session_state.advanced_multi_errors.append({
                        "name": name,
                        "timestamp": time.time(),
                        "error": str(e),
                        "trace": tb,
                    })
                    # keep going to evaluation attempt (some inspect failures are non-fatal)
                    metadata = None

                try:
                    metrics = evaluate_model_with_metadata(model, metadata, eval_samples, latency_runs, latency_warmup, device)
                except Exception as e:
                    tb = traceback.format_exc()
                    st.error(f"Evaluation failed for {name}")
                    # persist error details
                    st.session_state.advanced_multi_errors.append({
                        "name": name,
                        "timestamp": time.time(),
                        "error": str(e),
                        "trace": tb,
                    })
                    # mark that this model attempted evaluation but failed
                    info["metadata"] = (metadata.to_dict() if metadata is not None else None)
                    info["metrics"] = None
                    continue

                # Success path for this model
                info["metadata"] = (metadata.to_dict() if metadata is not None else None)
                info["metrics"] = metrics
                st.success(f"✅ {name} evaluation complete")

            except Exception as e:
                tb = traceback.format_exc()
                st.error(f"Unexpected error evaluating {name}")
                st.session_state.advanced_multi_errors.append({
                    "name": name,
                    "timestamp": time.time(),
                    "error": str(e),
                    "trace": tb,
                })
                # continue with other models
                continue

        st.success("Finished evaluating available models in the set.")
        # Refresh UI so metrics/metadata show, but keep errors persisted above
        try:
            st.experimental_rerun()
        except Exception:
            try:
                raise st.experimental_rerun
            except Exception:
                # Last resort: do not crash; simply continue and let the user refresh
                pass

    # Comparison UI (pairwise)
    evaluated = {n: i for n, i in model_set.items() if i.get("metrics")}
    if len(evaluated) < 2:
        st.info("Evaluate at least two models to compare them.")
        st.stop()

    st.subheader("⚖️ Pairwise Comparison")
    names = list(evaluated.keys())
    left = st.selectbox("Model A", names, index=0, key="advanced_pair_left")
    right = st.selectbox("Model B", names, index=1 if len(names) > 1 else 0, key="advanced_pair_right")

    m1 = evaluated[left]["metrics"]
    m2 = evaluated[right]["metrics"]
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### {left}")
        st.json(evaluated[left].get("metadata") or {})
        st.dataframe(pd.DataFrame([m1]).T, width='stretch')
    with col2:
        st.markdown(f"### {right}")
        st.json(evaluated[right].get("metadata") or {})
        st.dataframe(pd.DataFrame([m2]).T, width='stretch')

    if st.toggle("Generate explanation (optional)", value=False, key="advanced_pair_explain"):
        headline, explanation = beginner_conclusion(left, right, m1, m2)
        st.subheader("🧾 Explanation")
        st.success(headline)
        st.write(explanation)

# ==========================================================
# PATH A: LEARN ABOUT QUANTIZATION (built-in only, planner-driven)
# ==========================================================
elif internal_mode == "Path A: Learn":
    st.header("📚 Learn about quantization")
    st.markdown("""
    **System selects:** an FP32 baseline and quantized variants.
    Compare them and inspect tradeoffs.
    """)
    
    # Define evaluation settings for this mode
    device = torch.device("cpu")
    eval_samples = 200
    latency_runs = 20
    latency_warmup = 5
    
    architectures = get_builtin_architectures(MODEL_REGISTRY)
    if not architectures:
        st.warning("No built-in architectures available.")
        st.stop()
    
    selected_arch = st.selectbox("Architecture:", architectures, key="learn_arch")
    plan = plan_learn(MODEL_REGISTRY, selected_arch)
    
    st.info(plan.explanation)
    
    if not plan.can_compare:
        st.warning(plan.analysis_only_message)
        st.stop()
    
    if st.button("▶️ Run comparison (baseline + quantized)", key="learn_run"):
        # Load baseline model
        st.info("Loading baseline model...")
        try:
            baseline_model = load_model(plan.baseline_key, device)
            st.success(f"✅ Loaded: {plan.baseline_key}")
        except FileNotFoundError as e:
            st.error(f"Could not load baseline model: {plan.baseline_key}")
            st.error(str(e))
            st.info("💡 **Solution**: Run the training script to generate this model file.")
            if "resnet18" in plan.baseline_key.lower():
                st.code("python scripts/train_cifar10.py", language="bash")
            elif "distilbert" in plan.baseline_key.lower():
                st.code("python scripts/train_sst2.py", language="bash")
            with st.expander("📋 Full Error Details"):
                st.exception(e)
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error loading baseline: {plan.baseline_key}")
            with st.expander("📋 Error Details"):
                st.exception(e)
            st.stop()
        
        # Inspect baseline
        baseline_md = inspect_model_from_registry(plan.baseline_key, baseline_model, str(device))
        
        # Evaluate baseline
        st.info("Evaluating baseline model...")
        try:
            baseline_metrics = evaluate_model_with_metadata(
                baseline_model, baseline_md, eval_samples, latency_runs, latency_warmup, device
            )
            st.success("✅ Baseline evaluation complete")
        except Exception as e:
            st.error("Failed to evaluate baseline model")
            with st.expander("📋 Error Details"):
                st.exception(e)
            st.stop()
        
        # Load and evaluate variants
        variant_results = []
        progress_bar = st.progress(0)
        
        for idx, vk in enumerate(plan.variant_keys):
            progress_bar.progress((idx + 1) / (len(plan.variant_keys) + 1))
            
            st.info(f"Loading variant {idx + 1}/{len(plan.variant_keys)}: {vk}")
            try:
                vm = load_model(vk, device)
                vmd = inspect_model_from_registry(vk, vm, str(device))

                # If the variant metadata lacks dataset/task info, inherit from baseline metadata
                try:
                    if getattr(vmd, 'dataset', None) and vmd.dataset.value == 'unknown':
                        vmd.dataset = baseline_md.dataset
                        vmd.task = baseline_md.task
                        vmd.num_classes = baseline_md.num_classes
                except Exception:
                    pass

                st.info(f"Evaluating {vk}...")
                vmetrics = evaluate_model_with_metadata(
                    vm, vmd, eval_samples, latency_runs, latency_warmup, device
                )
                
                variant_results.append({
                    "key": vk,
                    "metadata": vmd,
                    "metrics": vmetrics,
                    "model": vm
                })
                st.success(f"✅ {vk} complete")
                
            except FileNotFoundError as e:
                st.warning(f"⚠️ Could not load variant: {vk}")
                st.info(f"This model file doesn't exist yet. You can generate it by running the appropriate quantization script.")
                with st.expander(f"Error details for {vk}"):
                    st.exception(e)
                # Continue with other variants
                continue
            except Exception as e:
                st.warning(f"⚠️ Error with variant: {vk}")
                with st.expander(f"Error details for {vk}"):
                    st.exception(e)
                # Continue with other variants
                continue
        
        progress_bar.empty()
        
        #
        # Check if we have any variants to compare
        if not variant_results:
            st.warning("⚠️ No local quantized variants found for this architecture.")

            # Suggest generating local quantized models
            st.info("You can generate quantized variants locally:")
            if "resnet18" in selected_arch.lower():
                st.code("python scripts/quantize_cifar10_ptq.py", language="bash")
                st.code("python scripts/train_cifar10_qat.py", language="bash")
            elif "distilbert" in selected_arch.lower():
                st.code("python scripts/quantize_sst2_dynamic.py", language="bash")

            # 🌐 NEW: Online model fallback
            try:
                from ui.hf_search import search_hf_models
                from ui.model_loader import load_hf_model_by_name
            except Exception as e:
                search_hf_models = None
                load_hf_model_by_name = None
                st.warning("Hugging Face search/load utilities unavailable (missing dependencies).")

            st.divider()
            st.subheader("🌐 Find similar models online (Hugging Face)")

            candidates = []
            if search_hf_models is not None:
                try:
                    candidates = search_hf_models(selected_arch)
                except Exception as e:
                    st.warning(f"Hugging Face search failed: {e}")

            # If no candidates found, run debug search and show diagnostics
            if not candidates and search_hf_models is not None:
                try:
                    from ui.hf_search import search_hf_models_debug
                    dbg_results, dbg_info = search_hf_models_debug(selected_arch)
                    # show debug info to the user to help diagnose
                    with st.expander("Debug: Hugging Face search diagnostics"):
                        st.write({"queries_tried": dbg_info})
                    # offer debug results if any
                    if dbg_results:
                        candidates = dbg_results
                except Exception:
                    pass

            if candidates:
                # Ensure we have metadata to evaluate against (fall back to session state)
                metadata = locals().get('metadata') or st.session_state.get('ib_metadata') or st.session_state.get('compare_uploaded_metadata') or None
                if metadata is None:
                    st.error("Model metadata not available for comparison. Run the initial analysis first.")
                    st.stop()
                chosen_model = st.selectbox("Select a model to compare:", candidates)

                if st.button("Load online model and compare"):
                    if load_hf_model_by_name is None:
                        st.error("Cannot load Hugging Face models: loader not available.")
                    else:
                        with st.spinner(f"Loading {chosen_model}..."):
                            try:
                                hf_model = load_hf_model_by_name(chosen_model, device)
                                st.success(f"Loaded online model: {chosen_model}")
                                # Add it as a variant (simple representation)
                                variant_results.append({
                                    "metadata": metadata,
                                    "metrics": evaluate_model_with_metadata(hf_model, metadata, eval_samples, latency_runs, latency_warmup, device),
                                    "model": hf_model,
                                    "source": "huggingface",
                                    "name": chosen_model,
                                })
                            except Exception as e:
                                st.error(f"Failed to load Hugging Face model: {e}")
            else:
                st.info("No similar models found online.")

            # --- Help / Options & Next Steps (placed under HF search) ---
            st.markdown("---")
            st.subheader("Options to continue")
            st.write("You can: 1) Upload a baseline/variant manually, 2) Generate quantized variants locally, or 3) Re-run Hugging Face search with diagnostics.")

            with st.expander("1) Upload a baseline or quantized variant manually", expanded=False):
                st.write("Upload a PyTorch checkpoint (.pt/.pth/.safetensors) or an ONNX file (.onnx) to compare.")
                uploaded_main = st.file_uploader("Upload model file", type=["pt", "pth", "safetensors", "onnx"], key="upload_variant_bottom")
                if uploaded_main is not None:
                    tmpdir = os.path.join(".", "uploads")
                    os.makedirs(tmpdir, exist_ok=True)
                    out_path = os.path.join(tmpdir, uploaded_main.name)
                    with open(out_path, "wb") as f:
                        f.write(uploaded_main.getbuffer())
                    st.success(f"Saved uploaded model to {out_path}. You can now use the Compare flow to load it as a variant.")

            with st.expander("2) Generate quantized variants locally", expanded=False):
                st.write("Run these scripts to create PTQ/QAT variants locally:")
                try:
                    if "resnet18" in selected_arch.lower():
                        st.code("python scripts/quantize_cifar10_ptq.py", language="bash")
                        st.code("python scripts/train_cifar10_qat.py", language="bash")
                    elif "distilbert" in selected_arch.lower():
                        st.code("python scripts/quantize_sst2_dynamic.py", language="bash")
                except Exception:
                    st.write("Select an architecture to see suggested scripts.")
                st.info("After generating variants, re-run this comparison to see them listed.")

            with st.expander("3) Re-run Hugging Face search with diagnostics", expanded=False):
                st.write("Run the HF debug search to see per-query diagnostics and samples.")
                if st.button("Run HF debug search", key="hf_debug_bottom"):
                    try:
                        from ui.hf_search import search_hf_models_debug
                        dbg_results, dbg_info = search_hf_models_debug(selected_arch)
                        st.write({"queries_tried": dbg_info})
                        if dbg_results:
                            st.success("Debug search found candidates — re-run the comparison to surface them.")
                            st.write(dbg_results)
                    except Exception as e:
                        st.error(f"HF debug search failed: {e}")

            # If still nothing to compare, stop
            if not variant_results:
                st.stop()
        #
        
        # Display results
        st.subheader("📊 Results")
        
        # Create comparison table
        comparison_data = {
            "Model": ["Baseline (FP32)"] + [
                v["metadata"].quantization_method.value.upper() for v in variant_results
            ]
        }
        
        if baseline_metrics.get('Accuracy') is not None:
            comparison_data["Accuracy (%)"] = [baseline_metrics['Accuracy'] * 100] + [
                v['metrics'].get('Accuracy', 0) * 100 if v['metrics'].get('Accuracy') else None
                for v in variant_results
            ]
        
        comparison_data["Latency (ms)"] = [baseline_metrics['Latency (ms)']] + [
            v['metrics']['Latency (ms)'] for v in variant_results
        ]
        
        comparison_data["Size (MB)"] = [baseline_metrics['param_MB']] + [
            v['metrics']['param_MB'] for v in variant_results
        ]
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Explanations (required for Beginner Mode)
        st.subheader("🧾 Explanation")
        
        for v in variant_results:
            variant_name = v["metadata"].quantization_method.value.upper()
            headline, expl = beginner_conclusion(
                "Baseline (FP32)",
                variant_name,
                baseline_metrics,
                v["metrics"]
            )
            st.success(f"**{variant_name}**: {headline}")
            st.write(expl)
            st.markdown("---")
        
        # Summary recommendation
        st.subheader("💡 Overall Recommendation")
        if variant_results:
            # Find best variant using weighted score across accuracy, latency, and size.
            # We compute normalized ratios (higher is better):
            #  - accuracy_ratio = variant_acc / baseline_acc (if available)
            #  - latency_ratio = baseline_latency / variant_latency
            #  - size_ratio = baseline_size_MB / variant_size_MB
            # Final score = w_acc*accuracy_ratio + w_lat*latency_ratio + w_size*size_ratio
            # Default weights: accuracy 0.5, latency 0.3, size 0.2
            def variant_score(v, w_acc=0.5, w_lat=0.3, w_size=0.2):
                vm = v['metrics']
                # Accuracy ratio
                try:
                    if baseline_metrics.get('Accuracy') and vm.get('Accuracy') is not None:
                        acc_ratio = vm['Accuracy'] / baseline_metrics['Accuracy']
                    else:
                        acc_ratio = 0.5
                except Exception:
                    acc_ratio = 0.5

                # Latency ratio (higher means faster vs baseline)
                try:
                    lat_ratio = baseline_metrics['Latency (ms)'] / vm['Latency (ms)']
                except Exception:
                    lat_ratio = 1.0

                # Size ratio (higher means smaller variant)
                try:
                    size_ratio = baseline_metrics['param_MB'] / vm['param_MB']
                except Exception:
                    size_ratio = 1.0

                # Guard against zero/inf
                if not (isinstance(acc_ratio, (int, float)) and acc_ratio == acc_ratio):
                    acc_ratio = 0.5
                if not (isinstance(lat_ratio, (int, float)) and lat_ratio == lat_ratio):
                    lat_ratio = 1.0
                if not (isinstance(size_ratio, (int, float)) and size_ratio == size_ratio):
                    size_ratio = 1.0

                return w_acc * acc_ratio + w_lat * lat_ratio + w_size * size_ratio

            best_variant = max(variant_results, key=lambda x: variant_score(x))
            score_val = variant_score(best_variant)
            st.success(
                f"**Recommended for deployment**: {best_variant['metadata'].quantization_method.value.upper()} "
                f"(score={score_val:.3f}) — best combined accuracy/latency/size tradeoff."
            )

# ==========================================================
# PATH B: UPLOAD YOUR OWN MODEL (planner-driven comparison)
# ==========================================================
elif internal_mode == "Path B: Upload":
    st.header("📤 Upload your own model to explore quantization")
    st.markdown("""
    We **analyze** your model (architecture, task, precision) and **infer comparison intent**:  
    - If quantized → compare to FP32 baseline if available  
    - If FP32 → compare to known quantized variants  
    - If no match → analysis-only with explanation  
    Then choose **Beginner** (guided + explanations) or **Advanced** (raw metrics, multi-model).
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

    # Beginner Mode: hide knobs by default (but keep them accessible)
    if user_mode == "Beginner Mode":
        eval_samples = 200
        latency_runs = 50
        latency_warmup = 10
        with st.sidebar.expander("Show advanced settings"):
            eval_samples = st.number_input(
                "Num eval samples",
                min_value=32,
                max_value=5000,
                value=int(eval_samples),
                step=32,
                key="intelligent_samples",
                help="Number of samples to use for accuracy evaluation"
            )
            latency_runs = st.number_input(
                "Latency runs",
                min_value=5,
                max_value=200,
                value=int(latency_runs),
                key="intelligent_latency_runs",
                help="Number of inference runs to average for latency measurement"
            )
            latency_warmup = st.number_input(
                "Latency warmup",
                min_value=1,
                max_value=50,
                value=int(latency_warmup),
                key="intelligent_warmup",
                help="Number of warmup runs before measuring latency"
            )
    else:
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
        # Build analysis result and run comparison planner (shared logic)
        try:
            profile = ModelProfile(model, str(device))
            param_count = profile.param_counts["total"]
            size_mb = profile.size_fp32["size_mb"]
        except Exception:
            profile = None
            param_count = sum(p.numel() for p in model.parameters())
            size_mb = (param_count * 4) / (1024 * 1024)
        analysis = ModelAnalysisResult(
            metadata=metadata,
            profile=profile,
            param_count=param_count,
            size_mb=size_mb,
            load_success=True,
            model=model,
        )
        plan = plan_upload(analysis)
        st.session_state.comparison_plan = plan
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
        orchestrator = get_comparison_orchestrator()  # for manual fallback buttons
        # Always show detected metadata so it remains visible after evaluation
        try:
            display_model_metadata(metadata)
        except Exception:
            pass
        
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

        

        # Show comparison plan (system-driven; same logic for Beginner and Advanced)
        plan = st.session_state.get("comparison_plan")
        if not plan:
            st.warning("Comparison plan not set. Re-run Analyze Model.")
            st.stop()
        st.subheader("📋 Comparison plan")
        st.info(plan.explanation)
        if not plan.can_compare:
            st.warning(plan.analysis_only_reason or "No comparison available.")
            st.subheader("🧾 What these results mean")
            if initial_metrics.get("Accuracy") is None:
                st.info(
                    "Accuracy is not available for this model in the current pipeline. "
                    "Latency, memory/size, and architecture detection are still useful."
                )
            else:
                st.write(
                    "Accuracy tells you how often the model is correct. "
                    "Latency (ms) is inference time per example; Model Size is from parameter storage."
                )
            st.success("**Conclusion:** Your model has been analyzed. No built-in baseline/variants for comparison.")

            # Offer alternatives including Hugging Face search for online variants
            st.write("---")
            st.subheader("Options to continue")
            st.write("You can: 1) Upload a baseline/variant manually, 2) Generate quantized variants locally, or 3) Search Hugging Face for similar models.")

            # Local generation helper
            with st.expander("🛠️ Generate quantized variants locally"):
                if "resnet" in metadata.architecture.lower():
                    st.code("python scripts/quantize_cifar10_ptq.py", language="bash")
                    st.code("python scripts/train_cifar10_qat.py", language="bash")
                elif "distilbert" in metadata.architecture.lower():
                    st.code("python scripts/quantize_sst2_dynamic.py", language="bash")

            # Hugging Face fallback
            try:
                from ui.hf_search import search_hf_models
                from ui.model_loader import load_hf_model_by_name
            except Exception:
                search_hf_models = None
                load_hf_model_by_name = None

            if search_hf_models is None:
                st.info("Hugging Face search unavailable (missing dependencies).")
                st.stop()

            st.subheader("🌐 Search Hugging Face for similar models")
            try:
                hf_candidates = search_hf_models(metadata.architecture)
            except Exception as e:
                st.error(f"Hugging Face search failed: {e}")
                st.stop()

            # --- Help / Options & Next Steps (always visible under HF search) ---
            st.markdown("---")
            st.subheader("Options to continue")
            st.write("You can: 1) Upload a baseline/variant manually, 2) Generate quantized variants locally, or 3) Re-run Hugging Face search with diagnostics.")

            with st.expander("1) Upload a baseline or quantized variant manually", expanded=False):
                st.write("Upload a PyTorch checkpoint (.pt/.pth/.safetensors) or an ONNX file (.onnx) to compare.")
                uploaded_hf_help = st.file_uploader("Upload model file", type=["pt", "pth", "safetensors", "onnx"], key="upload_variant_hf_help")
                if uploaded_hf_help is not None:
                    tmpdir = os.path.join(".", "uploads")
                    os.makedirs(tmpdir, exist_ok=True)
                    out_path = os.path.join(tmpdir, uploaded_hf_help.name)
                    with open(out_path, "wb") as f:
                        f.write(uploaded_hf_help.getbuffer())
                    st.success(f"Saved uploaded model to {out_path}. You can now use the Compare flow to load it as a variant.")

            with st.expander("2) Generate quantized variants locally", expanded=False):
                st.write("Run these scripts to create PTQ/QAT variants locally:")
                try:
                    if "resnet" in metadata.architecture.lower():
                        st.code("python scripts/quantize_cifar10_ptq.py", language="bash")
                        st.code("python scripts/train_cifar10_qat.py", language="bash")
                    elif "distilbert" in metadata.architecture.lower():
                        st.code("python scripts/quantize_sst2_dynamic.py", language="bash")
                except Exception:
                    st.write("Select an architecture to see suggested scripts.")
                st.info("After generating variants, re-run this comparison to see them listed.")

            with st.expander("3) Re-run Hugging Face search with diagnostics", expanded=False):
                st.write("Run the HF debug search to see per-query diagnostics and samples.")
                if st.button("Run HF debug search", key="hf_debug_under_search"):
                    try:
                        from ui.hf_search import search_hf_models_debug
                        dbg_results, dbg_info = search_hf_models_debug(metadata.architecture)
                        st.write({"queries_tried": dbg_info})
                        if dbg_results:
                            st.success("Debug search found candidates — refresh to surface them.")
                            st.write(dbg_results)
                    except Exception as e:
                        st.error(f"HF debug search failed: {e}")

            if not hf_candidates:
                st.info("No similar models found on Hugging Face.")
                st.stop()

            selected_hf = st.selectbox("Select an online model to compare:", hf_candidates, key="hf_search_fallback_select")
            if st.button("Load and evaluate online model", key="hf_search_fallback_load"):
                if load_hf_model_by_name is None:
                    st.error("Cannot load Hugging Face models: loader not available.")
                else:
                    with st.spinner(f"Loading {selected_hf} from Hugging Face..."):
                        try:
                            hf_model = load_hf_model_by_name(selected_hf, device)
                            st.success(f"Loaded online model: {selected_hf}")
                            hf_metrics = evaluate_model_with_metadata(hf_model, metadata, eval_samples, latency_runs, latency_warmup, device)
                            st.subheader("📊 Online model results")
                            st.metric("Latency", f"{hf_metrics.get('Latency (ms)', 'N/A')}")
                            st.metric("Model Size", f"{hf_metrics.get('param_MB', 'N/A')}")
                            if hf_metrics.get('Accuracy') is not None:
                                st.metric("Accuracy", f"{hf_metrics['Accuracy']*100:.2f}%")
                        except Exception as e:
                            st.error(f"Failed to load/evaluate Hugging Face model: {e}")
            # After offering these options, stop further automatic plan handling
            st.stop()
        
        # Use plan intent (shared backend logic)
        from ui.comparison_planner import ComparisonIntent
        if plan.intent == ComparisonIntent.COMPARE_QUANTIZED_TO_BASELINE:
            st.subheader("📊 Quantized model → Compare to baseline")
            # Load baseline from plan (no duplicate lookup)
            baseline_result = None
            if plan.baseline_key:
                try:
                    baseline_model = load_model(plan.baseline_key, device)
                    baseline_metadata = inspect_model_from_registry(plan.baseline_key, baseline_model, str(device))
                    baseline_result = (baseline_metadata, baseline_model)
                except Exception as e:
                    st.warning(f"Could not load baseline: {e}")
            
            if not baseline_result:
                st.warning("⚠️ No baseline found for this model.")
                st.write("""
                **Options:**
                1. Upload a baseline model manually
                2. Register a baseline in the system
                3. Continue with single model evaluation
                """)

                # Beginner Mode: try to recommend a similar uploaded model if available
                if user_mode == "Beginner Mode":
                    rec = recommend_comparison_target(metadata, saved_models=saved_models)
                    if rec.mode == "uploaded_similar" and rec.uploaded_model_name:
                        st.info(f"Suggested comparison: **{rec.uploaded_model_name}** (similar uploaded model).")
                        if st.button("⚖️ Compare with suggested uploaded model", key="compare_with_suggested_uploaded"):
                            from ui.model_detection import load_generic_pytorch_model
                            from ui.model_inspection import inspect_model, ModelSource

                            other_path = saved_models[rec.uploaded_model_name]["path"]
                            other_model, other_load = load_generic_pytorch_model(other_path, str(device))
                            if not other_load["success"]:
                                st.error(f"Failed to load suggested model: {other_load['error']}")
                                st.stop()

                            other_md = inspect_model(other_model, source=ModelSource.USER, file_path=other_path, device=str(device))
                            other_metrics = evaluate_model_with_metadata(
                                other_model, other_md, eval_samples, latency_runs, latency_warmup, device
                            )

                            headline, explanation = beginner_conclusion(
                                rec.uploaded_model_name,
                                "Your uploaded model",
                                other_metrics,
                                initial_metrics,
                            )
                            st.subheader("🧾 Explanation")
                            st.success(headline)
                            st.write(explanation)

                            st.dataframe(
                                pd.DataFrame(
                                    {
                                        rec.uploaded_model_name: other_metrics,
                                        "Your uploaded model": initial_metrics,
                                    }
                                ),
                                width='stretch',
                            )
                            st.stop()
                    else:
                        st.info("Tip: Save or upload a baseline/similar model to enable comparisons, or continue in analysis-only mode.")
                
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

                    # Beginner Mode: required human-readable explanation
                    if user_mode == "Beginner Mode":
                        st.subheader("🧾 Explanation")
                        headline, explanation = beginner_conclusion(
                            "Baseline (FP32)",
                            "Quantized Model",
                            baseline_metrics,
                            quantized_metrics,
                        )
                        st.success(headline)
                        st.write(explanation)
                    
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

                    # Beginner Mode: required human-readable explanation
                    if user_mode == "Beginner Mode":
                        st.subheader("🧾 Explanation")
                        headline, explanation = beginner_conclusion(
                            "Baseline (FP32)",
                            "Quantized Model",
                            baseline_metrics,
                            quantized_metrics,
                        )
                        st.success(headline)
                        st.write(explanation)
                    
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
        
        elif plan.intent == ComparisonIntent.COMPARE_BASELINE_TO_VARIANTS:
            st.subheader("📊 Baseline model → Compare to quantized variants")
            st.success("✅ **Detected**: You uploaded a baseline FP32 model.")
            # Load variants from plan (no duplicate lookup)
            variants = []
            for vk in plan.variant_keys:
                try:
                    vm = load_model(vk, device)
                    vmd = inspect_model_from_registry(vk, vm, str(device))
                    if vmd.is_quantized() and vmd.is_compatible_with(metadata):
                        variants.append((vmd, vm))
                except Exception:
                    continue
                
            #check if variants found
            
            if not variants:
                st.warning("⚠️ No quantized variants found for this baseline.")
                st.write("""
                **Options:**
                1. Generate quantized variants locally
                2. Search for similar quantized models online (Hugging Face)
                3. Continue with single model evaluation
                """)

                # ----------------------------
                # Option A — local scripts
                # ----------------------------
                with st.expander("🛠️ Generate quantized variants locally"):
                    if "resnet" in metadata.architecture.lower():
                        st.code("python scripts/quantize_cifar10_ptq.py", language="bash")
                        st.code("python scripts/train_cifar10_qat.py", language="bash")
                    elif "distilbert" in metadata.architecture.lower():
                        st.code("python scripts/quantize_sst2_dynamic.py", language="bash")

                # ----------------------------
                # Option B — Hugging Face search (NEW)
                # ----------------------------
                st.subheader("🌐 Find quantized variants online")

                try:
                    from ui.hf_search import search_hf_models
                    from ui.model_loader import load_hf_model_by_name
                except Exception:
                    search_hf_models = None
                    load_hf_model_by_name = None
                    st.warning("Hugging Face integration unavailable (missing dependencies).")

                hf_candidates = []
                if search_hf_models is not None:
                    try:
                        hf_candidates = search_hf_models(metadata.architecture)
                    except Exception as e:
                        st.warning(f"Hugging Face search failed: {e}")

                if hf_candidates:
                    selected_hf = st.selectbox(
                        "Select an online model to compare:",
                        hf_candidates,
                        key="hf_variant_select"
                    )

                    if st.button("⚡ Load and Compare Online Model", key="load_hf_variant"):
                        if load_hf_model_by_name is None:
                            st.error("Cannot load Hugging Face models: loader not available.")
                        else:
                            with st.spinner(f"Loading {selected_hf} from Hugging Face..."):
                                try:
                                    hf_model = load_hf_model_by_name(selected_hf, device)
                                    st.success(f"✅ Loaded online model: {selected_hf}")

                                    # Evaluate HF model
                                    hf_metrics = evaluate_model_with_metadata(
                                        hf_model, metadata, eval_samples, latency_runs, latency_warmup, device
                                    )

                                    variant_results = [{
                                        "metadata": metadata,
                                        "metrics": hf_metrics,
                                        "model": hf_model,
                                        "source": "huggingface",
                                        "name": selected_hf
                                    }]

                                    baseline_metrics = initial_metrics

                                    st.session_state.ib_comparison_results = {
                                        "baseline_metrics": baseline_metrics,
                                        "variant_results": variant_results,
                                        "baseline_metadata": metadata
                                    }
                                except Exception as e:
                                    st.error(f"Failed to load Hugging Face model: {e}")
                        st.session_state.ib_comparison_type = "variants_comparison"
                        st.rerun()

                else:
                    st.info("No similar models found online.")

                # ----------------------------
                # Option C — fallback stop
                # ----------------------------
                st.stop()
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
                        # Recommendation using weighted score across accuracy, latency, and size
                        def variant_score(v, w_acc=0.5, w_lat=0.3, w_size=0.2):
                            vm = v['metrics']
                            try:
                                if baseline_metrics.get('Accuracy') and vm.get('Accuracy') is not None:
                                    acc_ratio = vm['Accuracy'] / baseline_metrics['Accuracy']
                                else:
                                    acc_ratio = 0.5
                            except Exception:
                                acc_ratio = 0.5
                            try:
                                lat_ratio = baseline_metrics['Latency (ms)'] / vm['Latency (ms)']
                            except Exception:
                                lat_ratio = 1.0
                            try:
                                size_ratio = baseline_metrics['param_MB'] / vm['param_MB']
                            except Exception:
                                size_ratio = 1.0
                            if not (isinstance(acc_ratio, (int, float)) and acc_ratio == acc_ratio):
                                acc_ratio = 0.5
                            if not (isinstance(lat_ratio, (int, float)) and lat_ratio == lat_ratio):
                                lat_ratio = 1.0
                            if not (isinstance(size_ratio, (int, float)) and size_ratio == size_ratio):
                                size_ratio = 1.0
                            return w_acc * acc_ratio + w_lat * lat_ratio + w_size * size_ratio

                        best_variant = max(variant_results, key=lambda x: variant_score(x))
                        score_val = variant_score(best_variant)
                        st.success(
                            f"**Recommended**: {best_variant['metadata'].quantization_method.value.upper()} "
                            f"(score={score_val:.3f}) — best combined accuracy/latency/size tradeoff"
                        )

                        if user_mode == "Beginner Mode":
                            st.subheader("🧾 Explanation (baseline vs recommended)")
                            headline, explanation = beginner_conclusion(
                                "Baseline (FP32)",
                                f"{best_variant['metadata'].quantization_method.value.upper()}",
                                baseline_metrics,
                                best_variant["metrics"],
                            )
                            st.success(headline)
                            st.write(explanation)
                    
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

    # Bottom-of-page: Options & Next Steps (shown after initial evaluation completes)
    try:
        if st.session_state.get("ib_analysis_complete"):
            metadata = st.session_state.get("ib_metadata")
            arch = getattr(metadata, "architecture", "your model") if metadata is not None else "your model"
            st.markdown("---")
            with st.expander("Options & Next Steps", expanded=False):
                st.write("If you don't have local quantized variants, you can:")
                st.markdown(f"- **Upload** a baseline/variant (PyTorch `.pt/.pth/.safetensors` or ONNX `.onnx`) to compare with {arch}.")
                st.markdown("- **Generate** quantized variants locally using the provided scripts.")
                st.markdown("- **Run a Hugging Face debug search** to see per-query diagnostics when automatic search returns nothing.")
                st.write("---")
                uploaded_main = st.file_uploader("Upload a baseline or quantized variant to compare", type=["pt", "pth", "safetensors", "onnx"], key="upload_variant_bottom")
                if uploaded_main is not None:
                    tmpdir = os.path.join(".", "uploads")
                    os.makedirs(tmpdir, exist_ok=True)
                    out_path = os.path.join(tmpdir, uploaded_main.name)
                    with open(out_path, "wb") as f:
                        f.write(uploaded_main.getbuffer())
                    st.success(f"Saved uploaded model to {out_path}. You can now use the Compare flow to load it as a variant.")
                with st.expander("Generate quantized variants locally"):
                    if metadata is not None and "resnet" in getattr(metadata, "architecture", "").lower():
                        st.code("python scripts/quantize_cifar10_ptq.py", language="bash")
                        st.code("python scripts/train_cifar10_qat.py", language="bash")
                    elif metadata is not None and "distilbert" in getattr(metadata, "architecture", "").lower():
                        st.code("python scripts/quantize_sst2_dynamic.py", language="bash")
                    st.info("After generating variants, re-run this comparison to see them listed.")
                with st.expander("Hugging Face debug search"):
                    if st.button("Run HF debug search", key="hf_debug_bottom"):
                        try:
                            from ui.hf_search import search_hf_models_debug
                            dbg_results, dbg_info = search_hf_models_debug(getattr(metadata, "architecture", ""))
                            st.write({"queries_tried": dbg_info})
                            if dbg_results:
                                st.success("Debug search found candidates — re-run the comparison to surface them.")
                                st.write(dbg_results)
                        except Exception as e:
                            st.error(f"HF debug search failed: {e}")
    except Exception:
        pass
