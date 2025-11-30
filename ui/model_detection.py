# ui/model_detection.py
"""
For users importing their own model
Generic model loading and introspection system.
Works with any PyTorch model regardless of architecture.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from collections import defaultdict
import warnings

# Try to import optional dependencies
try:
    from fvcore.nn import FlopCounterMode
    FLOPS_AVAILABLE = True
except ImportError:
    FLOPS_AVAILABLE = False

# ==========================================================
# MODEL TYPE DETECTION
# ==========================================================

class ModelArchitectureType:
    """Enum for detected model types"""
    CNN = "CNN"
    TRANSFORMER = "Transformer"
    RECURRENT = "RNN/LSTM"
    MLP = "MLP"
    HYBRID = "Hybrid"
    UNKNOWN = "Unknown"


def detect_architecture_type(model: nn.Module) -> str:
    """
    Detect the primary architecture type by inspecting model layers.
    Returns one of: CNN, Transformer, RNN/LSTM, MLP, Hybrid, Unknown
    """
    layer_types = defaultdict(int)
    
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv3d):
            layer_types["conv"] += 1
        elif isinstance(module, (nn.LSTM, nn.GRU, nn.RNN)):
            layer_types["recurrent"] += 1
        elif isinstance(module, nn.MultiheadAttention):
            layer_types["attention"] += 1
        elif isinstance(module, nn.TransformerEncoderLayer) or isinstance(module, nn.TransformerDecoderLayer):
            layer_types["transformer"] += 1
        elif isinstance(module, nn.Linear):
            layer_types["linear"] += 1
    
    # Decision logic
    if layer_types["transformer"] > 0 or layer_types["attention"] > 5:
        if layer_types["conv"] > 0:
            return ModelArchitectureType.HYBRID
        return ModelArchitectureType.TRANSFORMER
    
    if layer_types["recurrent"] > 0:
        return ModelArchitectureType.RECURRENT
    
    if layer_types["conv"] > 0:
        if layer_types["linear"] > 2:
            return ModelArchitectureType.HYBRID
        return ModelArchitectureType.CNN
    
    if layer_types["linear"] > 0:
        return ModelArchitectureType.MLP
    
    return ModelArchitectureType.UNKNOWN


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count total, trainable, and non-trainable parameters.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
    }


def estimate_model_size(model: nn.Module, precision: str = "fp32") -> Dict[str, float]:
    """
    Estimate model size in MB based on parameter count and precision.
    precision: "fp32", "fp16", "int8", "int4"
    """
    params = count_parameters(model)["total"]
    
    # Bytes per parameter
    precision_bytes = {
        "fp32": 4,
        "fp16": 2,
        "int8": 1,
        "int4": 0.5,
    }
    
    bytes_per_param = precision_bytes.get(precision, 4)
    size_mb = (params * bytes_per_param) / (1024 * 1024)
    
    return {
        "size_mb": size_mb,
        "precision": precision,
        "params": params,
    }


# ==========================================================
# INPUT/OUTPUT SHAPE INFERENCE
# ==========================================================

def infer_model_io_shapes(
    model: nn.Module,
    device: str = "cpu",
    sample_inputs: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Attempt to infer input and output shapes by running the model with synthetic data.
    
    Args:
        model: PyTorch model to analyze
        device: Device to run on ("cpu" or "cuda")
        sample_inputs: Optional sample inputs. If None, we'll try common shapes.
    
    Returns:
        Dict with input_shapes, output_shapes, and inference_info
    """
    model.eval()
    device = torch.device(device)
    
    # If no sample inputs provided, try common shapes
    if sample_inputs is None:
        sample_inputs = _generate_sample_inputs(model, device)
    
    try:
        with torch.no_grad():
            if isinstance(sample_inputs, dict):
                outputs = model(**sample_inputs)
            elif isinstance(sample_inputs, (list, tuple)):
                outputs = model(*sample_inputs)
            else:
                outputs = model(sample_inputs)
        
        # Extract shapes
        input_shapes = _extract_tensor_shapes(sample_inputs)
        output_shapes = _extract_tensor_shapes(outputs)
        
        return {
            "input_shapes": input_shapes,
            "output_shapes": output_shapes,
            "inference_success": True,
            "error": None,
        }
    except Exception as e:
        return {
            "input_shapes": None,
            "output_shapes": None,
            "inference_success": False,
            "error": str(e),
        }


def _generate_sample_inputs(model: nn.Module, device: torch.device) -> torch.Tensor:
    """
    Generate plausible sample inputs by inspecting the model's first layer.
    """
    # Try to find the first Conv or Linear layer to infer input shape
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            in_channels = module.in_channels
            return torch.randn(1, in_channels, 224, 224).to(device)
        elif isinstance(module, nn.Conv1d):
            in_channels = module.in_channels
            return torch.randn(1, in_channels, 128).to(device)
        elif isinstance(module, nn.Linear):
            in_features = module.in_features
            return torch.randn(1, in_features).to(device)
    
    # Default: assume image-like input
    return torch.randn(1, 3, 224, 224).to(device)


def _extract_tensor_shapes(obj: Any) -> Any:
    """
    Recursively extract tensor shapes from nested structures.
    Handles: Tensor, dict, list, tuple
    """
    if isinstance(obj, torch.Tensor):
        return {
            "shape": tuple(obj.shape),
            "dtype": str(obj.dtype),
        }
    elif isinstance(obj, dict):
        return {k: _extract_tensor_shapes(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_extract_tensor_shapes(item) for item in obj]
    else:
        return type(obj).__name__


# ==========================================================
# LAYER ANALYSIS
# ==========================================================

def analyze_layer_composition(model: nn.Module) -> Dict[str, int]:
    """
    Count occurrences of each layer type in the model.
    """
    layer_counts = defaultdict(int)
    
    for module in model.modules():
        layer_name = module.__class__.__name__
        layer_counts[layer_name] += 1
    
    return dict(sorted(layer_counts.items(), key=lambda x: x[1], reverse=True))


def get_model_depth(model: nn.Module) -> int:
    """
    Estimate model depth (nesting level of modules).
    """
    def _get_max_depth(module):
        if len(list(module.children())) == 0:
            return 1
        return 1 + max(_get_max_depth(child) for child in module.children())
    
    return _get_max_depth(model)


# ==========================================================
# FLOPS & COMPLEXITY ESTIMATION
# ==========================================================

def estimate_flops(
    model: nn.Module,
    sample_input: Any,
    device: str = "cpu",
) -> Optional[int]:
    """
    Estimate FLOPs using fvcore if available.
    Returns FLOPs count or None if estimation fails.
    """
    if not FLOPS_AVAILABLE:
        return None
    
    model.eval()
    device = torch.device(device)
    model = model.to(device)
    
    # Handle dict inputs
    if isinstance(sample_input, dict):
        sample_input = {k: v.to(device) for k, v in sample_input.items()}
    else:
        sample_input = sample_input.to(device)
    
    try:
        with torch.no_grad():
            with FlopCounterMode(model, ignore_list=[]) as flops_counter:
                if isinstance(sample_input, dict):
                    model(**sample_input)
                else:
                    model(sample_input)
            return flops_counter.total()
    except Exception as e:
        warnings.warn(f"FLOPs estimation failed: {e}")
        return None


# ==========================================================
# COMPREHENSIVE MODEL PROFILING
# ==========================================================

class ModelProfile:
    """Container for comprehensive model information."""
    
    def __init__(self, model: nn.Module, device: str = "cpu", sample_input: Optional[Any] = None):
        self.model = model
        self.device = device
        
        # Basic info
        self.arch_type = detect_architecture_type(model)
        self.param_counts = count_parameters(model)
        self.layer_composition = analyze_layer_composition(model)
        self.depth = get_model_depth(model)
        
        # I/O shapes
        io_info = infer_model_io_shapes(model, device, sample_input)
        self.input_shapes = io_info.get("input_shapes")
        self.output_shapes = io_info.get("output_shapes")
        self.io_inference_success = io_info.get("inference_success")
        
        # Size estimates
        self.size_fp32 = estimate_model_size(model, "fp32")
        self.size_fp16 = estimate_model_size(model, "fp16")
        self.size_int8 = estimate_model_size(model, "int8")
        self.size_int4 = estimate_model_size(model, "int4")
        
        # If sample input provided, compute FLOPs
        self.flops = None
        if sample_input is not None:
            self.flops = estimate_flops(model, sample_input, device)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary for easy serialization."""
        return {
            "architecture_type": self.arch_type,
            "total_parameters": self.param_counts["total"],
            "trainable_parameters": self.param_counts["trainable"],
            "frozen_parameters": self.param_counts["frozen"],
            "model_depth": self.depth,
            "layer_composition": self.layer_composition,
            "input_shapes": self.input_shapes,
            "output_shapes": self.output_shapes,
            "size_estimates": {
                "fp32_mb": self.size_fp32["size_mb"],
                "fp16_mb": self.size_fp16["size_mb"],
                "int8_mb": self.size_int8["size_mb"],
                "int4_mb": self.size_int4["size_mb"],
            },
            "flops": self.flops,
        }


# ==========================================================
# GENERIC MODEL LOADER
# ==========================================================

def load_generic_pytorch_model(
    model_path: str,
    device: str = "cpu",
    map_location: Optional[str] = None,
) -> Tuple[Optional[nn.Module], Dict[str, Any]]:
    """
    Generic loader for any PyTorch model.
    
    Handles:
    - .pt / .pth files (state_dict or full model)
    - Automatic device mapping
    - Error handling and recovery
    
    Returns:
        (model, load_info_dict)
    """
    device = torch.device(device)
    if map_location is None:
        map_location = device
    
    load_info = {
        "success": False,
        "model_type": None,
        "error": None,
        "warnings": [],
    }
    
    try:
        # Try to load
        checkpoint = torch.load(model_path, map_location=map_location, weights_only=False)
        
        # Case 1: Direct model object
        if isinstance(checkpoint, nn.Module):
            model = checkpoint
            load_info["model_type"] = "full_model"
        
        # Case 2: State dict only
        elif isinstance(checkpoint, dict):
            # Check if it looks like a state dict
            if "state_dict" in checkpoint:
                load_info["warnings"].append(
                    "Found 'state_dict' key in checkpoint. You may need to specify the model architecture separately."
                )
                load_info["error"] = "State dict with checkpoint wrapper detected. Cannot load without architecture."
                return None, load_info
            
            # Otherwise, assume it's a direct state_dict
            load_info["model_type"] = "state_dict_only"
            load_info["error"] = "Loaded state_dict without architecture. Please provide model architecture or use built-in models."
            return None, load_info
        
        else:
            load_info["error"] = f"Unknown checkpoint type: {type(checkpoint)}"
            return None, load_info
        
        model.to(device)
        model.eval()
        
        load_info["success"] = True
        return model, load_info
    
    except Exception as e:
        load_info["error"] = str(e)
        return None, load_info