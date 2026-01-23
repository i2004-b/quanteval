# Quantization Benchmarking Framework Architecture

## Overview

This document describes the architecture of the intelligent quantization benchmarking framework. The system has been refactored to support two main workflows:

1. **Workflow A**: Quantized Model → Compare to Baseline
2. **Workflow B**: Baseline Model → Find Best Quantization Strategy

## Core Components

### 1. Model Inspection Layer (`ui/model_inspection.py`)

The model inspection layer provides comprehensive model introspection to extract metadata.

**Key Classes:**
- `ModelMetadata`: Core data structure containing:
  - Architecture (e.g., "ResNet18", "DistilBERT")
  - Task (image_classification, text_classification)
  - Dataset (CIFAR-10, SST-2, ImageNet)
  - Precision (FP32, INT8, FP16, INT4)
  - Quantization method (none, dynamic, static, QAT)
  - Source (internal, user)

**Key Functions:**
- `inspect_model()`: Main inspection function that extracts all metadata
- `detect_quantization_method()`: Detects quantization by inspecting layers
- `detect_precision()`: Detects precision from parameter dtypes
- `detect_architecture_name()`: Identifies specific architecture names
- `detect_task_and_dataset()`: Infers task and dataset from model structure

### 2. Baseline Registry (`ui/baseline_registry.py`)

Extensible registry for baseline FP32 models indexed by (architecture, task, dataset).

**Key Classes:**
- `BaselineEntry`: Represents a baseline model entry
- `BaselineRegistry`: Main registry class with methods:
  - `register()`: Register a new baseline
  - `find_baseline()`: Find baseline for given metadata
  - `has_baseline()`: Check if baseline exists

**Usage:**
```python
registry = get_baseline_registry()
registry.register("ResNet18", TaskType.IMAGE_CLASSIFICATION, 
                 DatasetType.CIFAR10, "resnet18_baseline")
```

### 3. Comparison Orchestrator (`ui/comparison_orchestrator.py`)

Intelligently orchestrates model comparisons by:
- Deciding which models belong in the same comparison group
- Preventing incompatible comparisons
- Automatically assembling baseline + quantized variants

**Key Classes:**
- `ComparisonGroup`: Container for models that should be compared together
- `ComparisonOrchestrator`: Main orchestrator class

**Key Methods:**
- `determine_workflow()`: Determines workflow type from metadata
- `find_baseline_for_quantized()`: Workflow A - find baseline for quantized model
- `find_quantized_variants()`: Workflow B - find all quantized variants
- `create_comparison_group()`: Automatically create comparison group

### 4. Updated UI (`ui/app.py`)

The Streamlit UI has been enhanced with a new "Intelligent Benchmarking" mode that:

1. **Automatically detects** uploaded model type
2. **Suggests appropriate workflow** based on detection
3. **Finds compatible models** automatically
4. **Ranks and compares** models intelligently

## Workflows

### Workflow A: Quantized Model → Baseline Comparison

**Flow:**
1. User uploads a quantized model
2. System inspects model and detects it's quantized
3. System finds matching baseline using registry
4. Both models are evaluated
5. Side-by-side comparison is displayed with improvements

**Use Case:** User has a quantized model and wants to see how it compares to the baseline.

### Workflow B: Baseline Model → Best Quantization Strategy

**Flow:**
1. User uploads a baseline FP32 model
2. System inspects model and detects it's a baseline
3. System searches for all compatible quantized variants
4. All variants are evaluated
5. Variants are ranked by:
   - Fastest inference
   - Lowest memory
   - Best accuracy retention
6. Recommendation is provided

**Use Case:** User has a baseline model and wants to find the best quantization strategy.

## Design Decisions

### 1. Metadata Schema

The `ModelMetadata` dataclass uses enums for type safety and extensibility:
- `TaskType`: Extensible enum for different tasks
- `DatasetType`: Extensible enum for different datasets
- `PrecisionType`: Standard precision types
- `QuantizationMethod`: Standard quantization methods

### 2. Registry Design

Baselines are indexed by `(architecture, task, dataset)` tuple, not hardcoded to specific models. This allows:
- Easy addition of new baselines
- Automatic matching based on metadata
- No architecture-specific logic in UI

### 3. Comparison Key

Models are compared based on their `comparison_key` which is `(architecture, task, dataset)`. This ensures:
- Only compatible models are compared
- Prevents invalid comparisons
- Clear error messages when models are incompatible

### 4. Extensibility

The system is designed to be easily extended:
- New architectures: Add detection logic in `model_inspection.py`
- New tasks/datasets: Add to enums and detection logic
- New baselines: Register in `baseline_registry.py`
- New quantization methods: Add detection logic

## Integration Points

### Model Loader Integration

The system integrates with `model_loader.py`:
- Uses `MODEL_REGISTRY` to find models
- Uses `load_model()` to load models
- Maintains backward compatibility with existing registry

### Evaluation Integration

The system uses existing evaluation functions:
- `evaluate_cifar10_model()` for CIFAR-10 models
- `evaluate_sst2_model()` for SST-2 models
- `evaluate_generic_model()` as fallback

### UI Integration

The new "Intelligent Benchmarking" mode is added alongside existing modes:
- "Single Model": Original single model evaluation
- "Compare Models": Original manual comparison
- "Intelligent Benchmarking": New automatic workflow

## Usage Examples

### Adding a New Baseline

```python
from ui.baseline_registry import register_baseline
from ui.model_inspection import TaskType, DatasetType

register_baseline(
    architecture="EfficientNet-B0",
    task=TaskType.IMAGE_CLASSIFICATION,
    dataset=DatasetType.IMAGENET,
    model_key="efficientnet_b0_baseline",
    description="EfficientNet-B0 baseline for ImageNet"
)
```

### Inspecting a Model

```python
from ui.model_inspection import inspect_model, ModelSource

metadata = inspect_model(
    model,
    source=ModelSource.USER,
    file_path="path/to/model.pt",
    device="cpu"
)

print(f"Architecture: {metadata.architecture}")
print(f"Task: {metadata.task.value}")
print(f"Is Quantized: {metadata.is_quantized()}")
```

### Creating a Comparison Group

```python
from ui.comparison_orchestrator import get_comparison_orchestrator

orchestrator = get_comparison_orchestrator()
group = orchestrator.create_comparison_group(metadata, model, device="cpu")

if group:
    for metadata, model, name in group.get_all_models():
        print(f"{name}: {metadata.architecture}")
```

## Future Enhancements

1. **Automatic Quantization Generation**: Generate quantized variants on-the-fly
2. **Multi-Model Comparison**: Compare more than 2 models at once
3. **Custom Metrics**: Allow users to define custom comparison metrics
4. **Export Results**: Export comparison results to CSV/JSON
5. **Model Recommendations**: ML-based recommendations for quantization strategies

## Testing

To test the system:

1. **Test Workflow A**: Upload a quantized model (e.g., `resnet18_quantized_ptq.pt`)
2. **Test Workflow B**: Upload a baseline model (e.g., `resnet18_baseline.pt`)
3. **Test Detection**: Upload various model types and verify detection accuracy

## Notes

- The system maintains backward compatibility with existing functionality
- All existing evaluation modes continue to work
- New modules are modular and can be used independently
- Error handling is comprehensive with clear user messages
