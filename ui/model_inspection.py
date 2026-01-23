"""
Model Inspection Module

Provides comprehensive model introspection to extract metadata including:
- Architecture (ResNet18, DistilBERT, etc.)
- Task (image classification, text classification, etc.)
- Dataset (CIFAR-10, SST-2, etc.)
- Precision (FP32, INT8, etc.)
- Quantization method (none, dynamic, static, QAT)
- Source (internal, user)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import re
from collections import defaultdict

from ui.model_detection import detect_architecture_type, ModelProfile


class TaskType(Enum):
    """Supported task types"""
    IMAGE_CLASSIFICATION = "image_classification"
    TEXT_CLASSIFICATION = "text_classification"
    UNKNOWN = "unknown"


class DatasetType(Enum):
    """Supported dataset types"""
    CIFAR10 = "cifar10"
    SST2 = "sst2"
    IMAGENET = "imagenet"
    UNKNOWN = "unknown"


class PrecisionType(Enum):
    """Model precision types"""
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"
    UNKNOWN = "unknown"


class QuantizationMethod(Enum):
    """Quantization methods"""
    NONE = "none"  # Baseline FP32
    DYNAMIC = "dynamic"  # Dynamic quantization (PTQ)
    STATIC = "static"  # Static quantization (PTQ with calibration)
    QAT = "qat"  # Quantization-Aware Training
    UNKNOWN = "unknown"


class ModelSource(Enum):
    """Model source"""
    INTERNAL = "internal"  # Built-in registered model
    USER = "user"  # User-uploaded model


@dataclass
class ModelMetadata:
    """
    Comprehensive model metadata extracted through inspection.
    
    This is the core data structure that enables intelligent model comparison
    and quantization workflow decisions.
    """
    # Core identification
    architecture: str  # e.g., "ResNet18", "DistilBERT", "CNN", "Transformer"
    task: TaskType
    dataset: DatasetType
    
    # Quantization info
    precision: PrecisionType
    quantization_method: QuantizationMethod
    
    # Source tracking
    source: ModelSource
    model_key: Optional[str] = None  # Registry key if internal
    file_path: Optional[str] = None  # File path if user-uploaded
    
    # Additional metadata
    num_classes: Optional[int] = None
    confidence: float = 1.0  # Confidence in detection (0.0-1.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization/display"""
        return {
            "architecture": self.architecture,
            "task": self.task.value,
            "dataset": self.dataset.value,
            "precision": self.precision.value,
            "quantization_method": self.quantization_method.value,
            "source": self.source.value,
            "model_key": self.model_key,
            "file_path": self.file_path,
            "num_classes": self.num_classes,
            "confidence": self.confidence,
        }
    
    def get_comparison_key(self) -> Tuple[str, str, str]:
        """
        Returns a tuple (architecture, task, dataset) that identifies
        models that should be compared together.
        """
        return (self.architecture, self.task.value, self.dataset.value)
    
    def is_baseline(self) -> bool:
        """Check if this is a baseline FP32 model"""
        return (self.precision == PrecisionType.FP32 and 
                self.quantization_method == QuantizationMethod.NONE)
    
    def is_quantized(self) -> bool:
        """Check if this is a quantized model"""
        return self.quantization_method != QuantizationMethod.NONE
    
    def is_compatible_with(self, other: 'ModelMetadata') -> bool:
        """
        Check if two models are compatible for comparison.
        They must have the same architecture, task, and dataset.
        """
        return self.get_comparison_key() == other.get_comparison_key()


def detect_quantization_method(model: nn.Module) -> Tuple[QuantizationMethod, float]:
    """
    Detect quantization method by inspecting model layers.
    
    Returns:
        (method, confidence) tuple
    """
    # Check for quantized layers
    has_quantized_linear = False
    has_quantized_conv = False
    has_dynamic_quantized = False
    
    for name, module in model.named_modules():
        module_type = type(module).__name__
        
        # PyTorch quantized layers
        if "QuantizedLinear" in module_type or "QuantizedConv" in module_type:
            has_quantized_linear = True
            if "Dynamic" in module_type:
                has_dynamic_quantized = True
        
        # Check for quantization stubs (QAT indicator)
        if "QuantStub" in module_type or "DeQuantStub" in module_type:
            # This suggests QAT architecture, but model might not be quantized yet
            pass
    
    # Check state dict for quantized parameters
    state_dict = model.state_dict()
    has_quantized_params = any("scale" in k or "zero_point" in k for k in state_dict.keys())
    
    # Decision logic
    if has_dynamic_quantized:
        return QuantizationMethod.DYNAMIC, 0.9
    elif has_quantized_linear or has_quantized_conv or has_quantized_params:
        # Could be static PTQ or QAT - need more context
        # Check if model has quantization stubs (QAT architecture)
        has_stubs = any("quant" in name.lower() or "dequant" in name.lower() 
                       for name in model.named_modules())
        if has_stubs:
            return QuantizationMethod.QAT, 0.8
        else:
            return QuantizationMethod.STATIC, 0.8
    else:
        return QuantizationMethod.NONE, 1.0


def detect_precision(model: nn.Module) -> Tuple[PrecisionType, float]:
    """
    Detect model precision by inspecting parameter dtypes.
    
    Returns:
        (precision, confidence) tuple
    """
    param_dtypes = set()
    for param in model.parameters():
        param_dtypes.add(param.dtype)
    
    # Check for quantized types
    if torch.qint8 in param_dtypes or torch.quint8 in param_dtypes:
        return PrecisionType.INT8, 0.95
    elif torch.float16 in param_dtypes or torch.half in param_dtypes:
        return PrecisionType.FP16, 0.95
    elif torch.float32 in param_dtypes:
        # Could be FP32 baseline or quantized model with FP32 params
        # Check quantization method for more info
        quant_method, _ = detect_quantization_method(model)
        if quant_method != QuantizationMethod.NONE:
            # Quantized model but params are still FP32 (normal for some quantization schemes)
            return PrecisionType.INT8, 0.7  # Infer from quantization method
        return PrecisionType.FP32, 0.9
    else:
        return PrecisionType.UNKNOWN, 0.5


def detect_architecture_name(model: nn.Module, arch_type: str) -> str:
    """
    Detect specific architecture name (e.g., "ResNet18", "DistilBERT")
    from model structure and class name.
    
    Args:
        model: PyTorch model
        arch_type: Generic architecture type from model_detection (CNN, Transformer, etc.)
    """
    model_class_name = model.__class__.__name__
    
    # Check class name for known architectures
    if "ResNet" in model_class_name:
        # Try to extract ResNet variant
        match = re.search(r'ResNet(\d+)', model_class_name)
        if match:
            return f"ResNet{match.group(1)}"
        return "ResNet"
    elif "DistilBert" in model_class_name or "DistilBERT" in model_class_name:
        return "DistilBERT"
    elif "BERT" in model_class_name:
        return "BERT"
    elif "EfficientNet" in model_class_name:
        match = re.search(r'EfficientNet[_-]?([b\d]+)', model_class_name, re.I)
        if match:
            return f"EfficientNet-{match.group(1)}"
        return "EfficientNet"
    elif "MobileNet" in model_class_name:
        match = re.search(r'MobileNet[_-]?V?(\d+)', model_class_name, re.I)
        if match:
            return f"MobileNetV{match.group(1)}"
        return "MobileNet"
    elif "SqueezeNet" in model_class_name:
        return "SqueezeNet"
    
    # Fallback to generic type
    return arch_type


def detect_task_and_dataset(
    model: nn.Module, 
    architecture: str,
    profile: Optional[ModelProfile] = None
) -> Tuple[TaskType, DatasetType, int, float]:
    """
    Detect task type and dataset from model architecture and structure.
    
    Returns:
        (task, dataset, num_classes, confidence) tuple
    """
    # Architecture-based heuristics
    if "ResNet" in architecture or "EfficientNet" in architecture or "MobileNet" in architecture:
        # Vision models - likely image classification
        task = TaskType.IMAGE_CLASSIFICATION
        
        # Try to detect dataset from final layer
        num_classes = None
        for module in reversed(list(model.modules())):
            if isinstance(module, nn.Linear):
                num_classes = module.out_features
                break
        
        # Heuristic: CIFAR-10 has 10 classes, ImageNet has 1000
        if num_classes == 10:
            dataset = DatasetType.CIFAR10
            confidence = 0.8
        elif num_classes == 1000:
            dataset = DatasetType.IMAGENET
            confidence = 0.8
        else:
            dataset = DatasetType.UNKNOWN
            confidence = 0.5
        
        return task, dataset, num_classes, confidence
    
    elif "DistilBERT" in architecture or "BERT" in architecture:
        # Transformer models - likely text classification
        task = TaskType.TEXT_CLASSIFICATION
        
        # Try to detect num_classes from final layer
        num_classes = None
        for module in model.modules():
            if isinstance(module, nn.Linear) and hasattr(module, 'out_features'):
                # Usually the classifier head
                if module.out_features in [2, 3, 4, 5]:  # Common for binary/multi-class
                    num_classes = module.out_features
                    break
        
        # Heuristic: SST-2 is binary classification (2 classes)
        if num_classes == 2:
            dataset = DatasetType.SST2
            confidence = 0.8
        else:
            dataset = DatasetType.UNKNOWN
            confidence = 0.5
        
        return task, dataset, num_classes, confidence
    
    # Fallback
    return TaskType.UNKNOWN, DatasetType.UNKNOWN, None, 0.3


def inspect_model(
    model: nn.Module,
    source: ModelSource = ModelSource.USER,
    model_key: Optional[str] = None,
    file_path: Optional[str] = None,
    device: str = "cpu"
) -> ModelMetadata:
    """
    Main inspection function that extracts comprehensive metadata from a model.
    
    This is the core function that enables intelligent workflow decisions.
    
    Args:
        model: PyTorch model to inspect
        source: Whether model is internal (registered) or user-uploaded
        model_key: Registry key if internal model
        file_path: File path if user-uploaded
        device: Device to run inspection on
    
    Returns:
        ModelMetadata object with all detected information
    """
    # Get basic architecture type
    arch_type = detect_architecture_type(model)
    
    # Create profile for additional info
    try:
        profile = ModelProfile(model, device)
    except Exception:
        profile = None
    
    # Detect specific architecture name
    architecture = detect_architecture_name(model, arch_type)
    
    # Detect quantization
    quantization_method, quant_confidence = detect_quantization_method(model)
    precision, prec_confidence = detect_precision(model)
    
    # Detect task and dataset
    task, dataset, num_classes, task_confidence = detect_task_and_dataset(
        model, architecture, profile
    )
    
    # Overall confidence (minimum of all detections)
    overall_confidence = min(quant_confidence, prec_confidence, task_confidence)
    
    return ModelMetadata(
        architecture=architecture,
        task=task,
        dataset=dataset,
        precision=precision,
        quantization_method=quantization_method,
        source=source,
        model_key=model_key,
        file_path=file_path,
        num_classes=num_classes,
        confidence=overall_confidence,
    )


def inspect_model_from_registry(
    model_key: str,
    model: nn.Module,
    device: str = "cpu"
) -> ModelMetadata:
    """
    Inspect a model that was loaded from the registry.
    Uses registry key to enhance detection accuracy.
    """
    # Parse registry key for hints
    architecture_hint = None
    dataset_hint = None
    
    if "resnet18" in model_key.lower():
        architecture_hint = "ResNet18"
        if "cifar" in model_key.lower() or "cifar10" in model_key.lower():
            dataset_hint = DatasetType.CIFAR10
    elif "distilbert" in model_key.lower():
        architecture_hint = "DistilBERT"
        if "sst" in model_key.lower() or "sst2" in model_key.lower():
            dataset_hint = DatasetType.SST2
    
    # Run standard inspection
    metadata = inspect_model(
        model,
        source=ModelSource.INTERNAL,
        model_key=model_key,
        device=device
    )
    
    # Override with registry hints if available
    if architecture_hint:
        metadata.architecture = architecture_hint
        metadata.confidence = min(metadata.confidence + 0.1, 1.0)
    
    if dataset_hint:
        metadata.dataset = dataset_hint
        metadata.confidence = min(metadata.confidence + 0.1, 1.0)
    
    # Detect quantization from key
    if "ptq" in model_key.lower() or "quantized" in model_key.lower():
        if metadata.quantization_method == QuantizationMethod.NONE:
            metadata.quantization_method = QuantizationMethod.STATIC
            metadata.precision = PrecisionType.INT8
    elif "qat" in model_key.lower():
        if metadata.quantization_method == QuantizationMethod.NONE:
            metadata.quantization_method = QuantizationMethod.QAT
            metadata.precision = PrecisionType.INT8
    elif "dynamic" in model_key.lower():
        if metadata.quantization_method == QuantizationMethod.NONE:
            metadata.quantization_method = QuantizationMethod.DYNAMIC
            metadata.precision = PrecisionType.INT8
    elif "baseline" in model_key.lower():
        metadata.quantization_method = QuantizationMethod.NONE
        metadata.precision = PrecisionType.FP32
    
    return metadata
