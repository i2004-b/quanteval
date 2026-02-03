"""
Comparison Orchestrator Module

Intelligently orchestrates model comparisons by:
1. Deciding which models belong in the same comparison group
2. Preventing incompatible comparisons
3. Automatically assembling baseline + quantized variants
4. Supporting both workflows:
   - Workflow A: Quantized model → Compare to baseline
   - Workflow B: Baseline model → Find best quantization strategy
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from ui.model_inspection import (
    ModelMetadata, 
    ModelSource, 
    QuantizationMethod,
    PrecisionType
)
from ui.baseline_registry import get_baseline_registry
from ui.model_loader import MODEL_REGISTRY, load_model


class WorkflowType(Enum):
    """Supported workflow types"""
    QUANTIZED_TO_BASELINE = "quantized_to_baseline"  # Workflow A
    BASELINE_TO_QUANTIZED = "baseline_to_quantized"  # Workflow B
    UNKNOWN = "unknown"


@dataclass
class ComparisonGroup:
    """
    A group of models that should be compared together.
    
    Contains:
    - Baseline model (FP32, no quantization)
    - One or more quantized variants
    - Metadata for all models
    """
    baseline_metadata: ModelMetadata
    baseline_model: Any  # PyTorch model
    
    quantized_variants: List[Tuple[ModelMetadata, Any]]  # List of (metadata, model) tuples
    
    def get_all_models(self) -> List[Tuple[ModelMetadata, Any, str]]:
        """
        Get all models in the comparison group.
        
        Returns:
            List of (metadata, model, display_name) tuples
        """
        models = [
            (self.baseline_metadata, self.baseline_model, "Baseline (FP32)")
        ]
        
        for metadata, model in self.quantized_variants:
            # Generate display name
            if metadata.quantization_method == QuantizationMethod.DYNAMIC:
                name = f"Dynamic INT8"
            elif metadata.quantization_method == QuantizationMethod.PTQ:
                name = f"PTQ INT8"
            elif metadata.quantization_method == QuantizationMethod.STATIC:
                name = f"Static INT8"
            elif metadata.quantization_method == QuantizationMethod.QAT:
                name = f"QAT INT8"
            else:
                name = f"Quantized ({metadata.precision.value})"
            
            models.append((metadata, model, name))
        
        return models
    
    def is_valid(self) -> bool:
        """Check if comparison group is valid (all models compatible)"""
        if not self.baseline_metadata.is_baseline():
            return False
        
        baseline_key = self.baseline_metadata.get_comparison_key()
        
        for metadata, _ in self.quantized_variants:
            if not metadata.is_quantized():
                return False
            if metadata.get_comparison_key() != baseline_key:
                return False
        
        return True


class ComparisonOrchestrator:
    """
    Orchestrates intelligent model comparisons.
    
    Handles:
    - Determining which workflow to use
    - Finding compatible baselines
    - Assembling comparison groups
    - Validating comparisons
    """
    
    def __init__(self):
        self.baseline_registry = get_baseline_registry()
    
    def determine_workflow(self, metadata: ModelMetadata) -> WorkflowType:
        """
        Determine which workflow to use based on model metadata.
        
        Args:
            metadata: Model metadata to analyze
        
        Returns:
            WorkflowType enum
        """
        if metadata.is_quantized():
            return WorkflowType.QUANTIZED_TO_BASELINE
        elif metadata.is_baseline():
            return WorkflowType.BASELINE_TO_QUANTIZED
        else:
            return WorkflowType.UNKNOWN
    
    def find_baseline_for_quantized(
        self,
        quantized_metadata: ModelMetadata,
        device: str = "cpu"
    ) -> Optional[Tuple[ModelMetadata, Any]]:
        """
        Find and load baseline model for a quantized model (Workflow A).
        
        Args:
            quantized_metadata: Metadata of the quantized model
            device: Device to load model on
        
        Returns:
            (baseline_metadata, baseline_model) tuple if found, None otherwise
        """
        # Find baseline key
        baseline_key = self.baseline_registry.find_baseline(quantized_metadata)
        
        if not baseline_key:
            return None
        
        # Load baseline model
        try:
            baseline_model = load_model(baseline_key, device)
        except Exception as e:
            # Baseline model file might not exist
            return None
        
        # Inspect baseline to get metadata
        from ui.model_inspection import inspect_model_from_registry
        baseline_metadata = inspect_model_from_registry(baseline_key, baseline_model, device)
        
        # Verify compatibility
        if not quantized_metadata.is_compatible_with(baseline_metadata):
            return None
        
        return (baseline_metadata, baseline_model)
    
    def find_quantized_variants(
        self,
        baseline_metadata: ModelMetadata,
        device: str = "cpu"
    ) -> List[Tuple[ModelMetadata, Any]]:
        """
        Find all available quantized variants for a baseline (Workflow B).
        
        Args:
            baseline_metadata: Metadata of the baseline model
            device: Device to load models on
        
        Returns:
            List of (metadata, model) tuples for quantized variants
        """
        variants = []
        
        # Search registry for compatible quantized models
        comparison_key = baseline_metadata.get_comparison_key()
        architecture, task, dataset = comparison_key
        
        # Search through registry for matching models
        for model_key, config in MODEL_REGISTRY.items():
            # Skip baseline itself
            if model_key == baseline_metadata.model_key:
                continue
            
            # Check if key suggests it's a variant of this baseline
            # This is a heuristic - could be improved
            if architecture.lower() in model_key.lower():
                # Try to load and inspect
                try:
                    model = load_model(model_key, device)
                    from ui.model_inspection import inspect_model_from_registry
                    variant_metadata = inspect_model_from_registry(model_key, model, device)
                    
                    # Check if compatible
                    if variant_metadata.is_compatible_with(baseline_metadata):
                        if variant_metadata.is_quantized():
                            variants.append((variant_metadata, model))
                except Exception:
                    # Model might not exist or fail to load
                    continue
        
        return variants
    
    def create_comparison_group_workflow_a(
        self,
        quantized_metadata: ModelMetadata,
        quantized_model: Any,
        device: str = "cpu"
    ) -> Optional[ComparisonGroup]:
        """
        Create comparison group for Workflow A (quantized → baseline).
        
        Args:
            quantized_metadata: Metadata of quantized model
            quantized_model: Quantized model object
            device: Device to load models on
        
        Returns:
            ComparisonGroup if successful, None otherwise
        """
        # Find baseline
        baseline_result = self.find_baseline_for_quantized(quantized_metadata, device)
        
        if not baseline_result:
            return None
        
        baseline_metadata, baseline_model = baseline_result
        
        # Create comparison group
        group = ComparisonGroup(
            baseline_metadata=baseline_metadata,
            baseline_model=baseline_model,
            quantized_variants=[(quantized_metadata, quantized_model)]
        )
        
        if not group.is_valid():
            return None
        
        return group
    
    def create_comparison_group_workflow_b(
        self,
        baseline_metadata: ModelMetadata,
        baseline_model: Any,
        device: str = "cpu"
    ) -> Optional[ComparisonGroup]:
        """
        Create comparison group for Workflow B (baseline → quantized variants).
        
        Args:
            baseline_metadata: Metadata of baseline model
            baseline_model: Baseline model object
            device: Device to load models on
        
        Returns:
            ComparisonGroup if successful, None otherwise
        """
        # Find quantized variants
        variants = self.find_quantized_variants(baseline_metadata, device)
        
        if not variants:
            return None
        
        # Create comparison group
        group = ComparisonGroup(
            baseline_metadata=baseline_metadata,
            baseline_model=baseline_model,
            quantized_variants=variants
        )
        
        if not group.is_valid():
            return None
        
        return group
    
    def create_comparison_group(
        self,
        metadata: ModelMetadata,
        model: Any,
        device: str = "cpu"
    ) -> Optional[ComparisonGroup]:
        """
        Automatically create comparison group based on model type.
        
        Args:
            metadata: Model metadata
            model: Model object
            device: Device to load models on
        
        Returns:
            ComparisonGroup if successful, None otherwise
        """
        workflow = self.determine_workflow(metadata)
        
        if workflow == WorkflowType.QUANTIZED_TO_BASELINE:
            return self.create_comparison_group_workflow_a(metadata, model, device)
        elif workflow == WorkflowType.BASELINE_TO_QUANTIZED:
            return self.create_comparison_group_workflow_b(metadata, model, device)
        else:
            return None
    
    def validate_comparison(
        self,
        metadata1: ModelMetadata,
        metadata2: ModelMetadata
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate that two models can be compared.
        
        Args:
            metadata1: First model metadata
            metadata2: Second model metadata
        
        Returns:
            (is_valid, error_message) tuple
        """
        # Check compatibility
        if not metadata1.is_compatible_with(metadata2):
            return (
                False,
                f"Models are incompatible: {metadata1.get_comparison_key()} != {metadata2.get_comparison_key()}"
            )
        
        # Check that one is baseline and one is quantized
        if metadata1.is_baseline() == metadata2.is_baseline():
            if metadata1.is_baseline():
                return (False, "Both models are baselines. Cannot compare baseline to baseline.")
            else:
                return (False, "Both models are quantized. Compare each to baseline separately.")
        
        return (True, None)


# Global singleton instance
_default_orchestrator = None


def get_comparison_orchestrator() -> ComparisonOrchestrator:
    """Get the global comparison orchestrator instance"""
    global _default_orchestrator
    if _default_orchestrator is None:
        _default_orchestrator = ComparisonOrchestrator()
    return _default_orchestrator
