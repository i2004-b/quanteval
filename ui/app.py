# ui/app.py
# Run: streamlit run ui/app.py
import streamlit as st
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os, sys
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
st.set_page_config(page_title="Quanteval — Model Evaluation", layout="wide")
st.title("🧠 Quanteval — Model Evaluation UI")

st.sidebar.header("Evaluation Options")

# ==========================================================
# EVALUATION MODE SELECTION
# ==========================================================
evaluation_mode = st.sidebar.radio(
    "Evaluation Mode:",
    ["Single Model", "Compare Models"]
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

else:  # Compare Models mode
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
    use_generic_loader = False

# ----------
# Common settings
# ----------
device_choice = st.sidebar.selectbox("Run on", ["cpu", "cuda (if available)"])
device = torch.device("cuda" if (device_choice.startswith("cuda") and torch.cuda.is_available()) else "cpu")

eval_samples = st.sidebar.number_input("Num eval samples", min_value=32, max_value=5000, value=512, step=32)
latency_runs = st.sidebar.number_input("Latency runs", min_value=5, max_value=200, value=20)
latency_warmup = st.sidebar.number_input("Latency warmup", min_value=1, max_value=50, value=5)

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
# COMPARE MODELS MODE - built in only
# ==========================================================
else:  # Compare Models mode
    if st.button("Compare Models"):
        if baseline_key == quantized_key:
            st.error("Please select different models for comparison.")
            st.stop()
        
        results = {}
        
        # Evaluate baseline
        st.info(f"Loading baseline: {baseline_variant}...")
        try:
            baseline_model = load_model(baseline_key, device)
            st.success(f"✅ Baseline model loaded: {baseline_variant}")
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
        
        # Evaluate quantized
        st.info(f"Loading quantized: {quantized_variant}...")
        try:
            quantized_model = load_model(quantized_key, device)
            st.success(f"✅ Quantized model loaded: {quantized_variant}")
        except Exception as e:
            st.error(f"Error loading quantized model: {e}")
            st.stop()
        
        st.info("Evaluating quantized model...")
        if model_type == "ResNet18":
            quantized_metrics = evaluate_cifar10_model(quantized_model, eval_samples, latency_runs, latency_warmup, device)
        elif model_type == "DistilBERT":
            quantized_metrics = evaluate_sst2_model(quantized_model, eval_samples, latency_runs, latency_warmup, device)
        else:
            st.error("Unsupported model architecture.")
            st.stop()
        
        results["quantized"] = quantized_metrics
        
        # Display comparison
        st.subheader(f"📊 Comparison: {baseline_variant} vs {quantized_variant}")
        
        # Side-by-side metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### {baseline_variant} (Baseline)")
            st.metric("Accuracy", f"{baseline_metrics['Accuracy']*100:.2f}%")
            st.metric("Latency", f"{baseline_metrics['Latency (ms)']:.3f} ms")
            st.metric("Model Size", f"{baseline_metrics['param_MB']:.2f} MB")
        
        with col2:
            st.markdown(f"### {quantized_variant} (Quantized)")
            st.metric("Accuracy", f"{quantized_metrics['Accuracy']*100:.2f}%")
            st.metric("Latency", f"{quantized_metrics['Latency (ms)']:.3f} ms")
            st.metric("Model Size", f"{quantized_metrics['param_MB']:.2f} MB")
        
        # Improvement metrics
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
        charts = create_comparison_chart(baseline_metrics, quantized_metrics, baseline_variant, quantized_variant)
        if charts is not None:  # Only display if charts were created
            for chart in charts:
                st.plotly_chart(chart, width='stretch')
        
        # Comparison table
        st.subheader("Detailed Comparison")
        comparison_df = pd.DataFrame({
            baseline_variant: baseline_metrics,
            quantized_variant: quantized_metrics
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
            model_name=f"{model_type}-{quantized_variant}",
            method="ui-eval-comparison",
            out_dir="outputs/reports"
        )
        st.success("✅ Both experiments logged to outputs/reports")


if __name__ == "__main__":
    st.write("Run with: streamlit run ui/app.py")
