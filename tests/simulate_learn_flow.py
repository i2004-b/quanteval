import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from ui.comparison_planner import plan_learn
from ui.app import evaluate_model_with_metadata
from ui.baseline_registry import get_baseline_registry
from ui.model_loader import load_model
from ui.model_inspection import inspect_model_from_registry

# Simulate for ResNet18
registry = get_baseline_registry()
baselines = registry.list_baselines()
archs = [b.architecture for b in baselines]
print('Available architectures:', archs)

# find ResNet18 baseline key
for b in baselines:
    if b.architecture == 'ResNet18':
        baseline_key = b.model_key
        break
else:
    raise SystemExit('ResNet18 baseline not registered')

print('Baseline key:', baseline_key)
device = torch.device('cpu')

baseline_model = load_model(baseline_key, device)
baseline_md = inspect_model_from_registry(baseline_key, baseline_model, str(device))
print('baseline metadata:', baseline_md.to_dict())
baseline_metrics = evaluate_model_with_metadata(baseline_model, baseline_md, 200, 5, 1, device)
print('baseline metrics:', baseline_metrics)

# find variant keys from app.MODEL_REGISTRY
from ui.app import MODEL_REGISTRY as UI_REG
variant_keys = [UI_REG['ResNet18'][k] for k in UI_REG['ResNet18'] if UI_REG['ResNet18'][k] != baseline_key]
print('variant keys:', variant_keys)

for vk in variant_keys:
    vm = load_model(vk, device)
    vmd = inspect_model_from_registry(vk, vm, str(device))
    print('\nraw variant metadata for',vk, vmd.to_dict())
    # inherit baseline dataset if unknown
    if getattr(vmd,'dataset',None) and vmd.dataset.value == 'unknown':
        vmd.dataset = baseline_md.dataset
        vmd.task = baseline_md.task
        vmd.num_classes = baseline_md.num_classes
    metrics = evaluate_model_with_metadata(vm, vmd, 200, 5, 1, device)
    print('metrics for',vk, metrics)
