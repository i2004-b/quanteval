"""
Baseline Registry Module

Provides an extensible registry for baseline FP32 models indexed by:
(architecture, task, dataset)

This allows the system to automatically find matching baselines for
quantized models without hardcoding architecture-specific logic.
"""

from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

from ui.model_inspection import TaskType, DatasetType, ModelMetadata


@dataclass
class BaselineEntry:
    """
    Entry in the baseline registry.
    
    Each entry maps a (architecture, task, dataset) combination to a
    baseline model key in the model_loader registry.
    """
    architecture: str
    task: TaskType
    dataset: DatasetType
    model_key: str  # Key in model_loader.MODEL_REGISTRY
    description: Optional[str] = None
    
    def get_key(self) -> Tuple[str, str, str]:
        """Get the registry key tuple"""
        return (self.architecture, self.task.value, self.dataset.value)


class BaselineRegistry:
    """
    Extensible registry for baseline models.
    
    Usage:
        registry = BaselineRegistry()
        registry.register("ResNet18", TaskType.IMAGE_CLASSIFICATION, 
                         DatasetType.CIFAR10, "resnet18_baseline")
        
        baseline_key = registry.find_baseline(metadata)
    """
    
    def __init__(self):
        self._registry: Dict[Tuple[str, str, str], BaselineEntry] = {}
        self._initialize_default_baselines()
    
    def _initialize_default_baselines(self):
        """Initialize with default baselines from the existing registry"""
        # ResNet18 CIFAR-10 baseline
        self.register(
            architecture="ResNet18",
            task=TaskType.IMAGE_CLASSIFICATION,
            dataset=DatasetType.CIFAR10,
            model_key="resnet18_baseline",
            description="ResNet18 baseline for CIFAR-10 image classification"
        )
        
        # DistilBERT SST-2 baseline
        self.register(
            architecture="DistilBERT",
            task=TaskType.TEXT_CLASSIFICATION,
            dataset=DatasetType.SST2,
            model_key="distilbert_baseline",
            description="DistilBERT baseline for SST-2 text classification"
        )
    
    def register(
        self,
        architecture: str,
        task: TaskType,
        dataset: DatasetType,
        model_key: str,
        description: Optional[str] = None
    ):
        """
        Register a baseline model.
        
        Args:
            architecture: Model architecture name (e.g., "ResNet18")
            task: Task type (e.g., TaskType.IMAGE_CLASSIFICATION)
            dataset: Dataset type (e.g., DatasetType.CIFAR10)
            model_key: Key in model_loader.MODEL_REGISTRY
            description: Optional description
        """
        entry = BaselineEntry(
            architecture=architecture,
            task=task,
            dataset=dataset,
            model_key=model_key,
            description=description
        )
        key = entry.get_key()
        
        if key in self._registry:
            raise ValueError(
                f"Baseline already registered for {key}. "
                f"Existing: {self._registry[key].model_key}, "
                f"New: {model_key}"
            )
        
        self._registry[key] = entry
    
    def find_baseline(self, metadata: ModelMetadata) -> Optional[str]:
        """
        Find a baseline model key for the given model metadata.
        
        Args:
            metadata: ModelMetadata to find baseline for
        
        Returns:
            Model key if baseline found, None otherwise
        """
        key = metadata.get_comparison_key()
        entry = self._registry.get(key)
        
        if entry:
            return entry.model_key
        
        return None
    
    def find_baseline_by_key(
        self,
        architecture: str,
        task: TaskType,
        dataset: DatasetType
    ) -> Optional[str]:
        """
        Find baseline by explicit key components.
        
        Args:
            architecture: Model architecture
            task: Task type
            dataset: Dataset type
        
        Returns:
            Model key if baseline found, None otherwise
        """
        key = (architecture, task.value, dataset.value)
        entry = self._registry.get(key)
        
        if entry:
            return entry.model_key
        
        return None
    
    def list_baselines(self) -> List[BaselineEntry]:
        """List all registered baselines"""
        return list(self._registry.values())
    
    def has_baseline(self, metadata: ModelMetadata) -> bool:
        """Check if a baseline exists for the given metadata"""
        return self.find_baseline(metadata) is not None
    
    def get_entry(self, metadata: ModelMetadata) -> Optional[BaselineEntry]:
        """Get the full baseline entry for metadata"""
        key = metadata.get_comparison_key()
        return self._registry.get(key)


# Global singleton instance
_default_registry = None


def get_baseline_registry() -> BaselineRegistry:
    """Get the global baseline registry instance"""
    global _default_registry
    if _default_registry is None:
        _default_registry = BaselineRegistry()
    return _default_registry


def register_baseline(
    architecture: str,
    task: TaskType,
    dataset: DatasetType,
    model_key: str,
    description: Optional[str] = None
):
    """
    Convenience function to register a baseline in the global registry.
    
    Example:
        register_baseline(
            "EfficientNet-B0",
            TaskType.IMAGE_CLASSIFICATION,
            DatasetType.IMAGENET,
            "efficientnet_b0_baseline"
        )
    """
    registry = get_baseline_registry()
    registry.register(architecture, task, dataset, model_key, description)
