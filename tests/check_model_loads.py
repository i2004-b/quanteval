import sys
import os
import traceback
import torch

# Ensure project root is on sys.path so `ui` package can be imported
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from ui.model_loader import MODEL_REGISTRY, load_model, _get_project_path
except ModuleNotFoundError as e:
    print(f"Dependency missing when importing ui.model_loader: {e.name}")
    print("Please install required packages, e.g. `pip install -r requirements.txt`.")
    raise

print('Using device: cpu')
device = torch.device('cpu')

results = {}
for key, cfg in MODEL_REGISTRY.items():
    if cfg.get('type') == 'user':
        print(f"Skipping user upload entry: {key}")
        continue
    print(f"\n=== Trying to load {key} ({cfg.get('type')}) ===")
    try:
        m = load_model(key, device)
        # Quick sanity checks
        params = sum(p.numel() for p in m.parameters()) if any(True for _ in m.parameters()) else 0
        print(f"Loaded {key}: params={params}")
        results[key] = ('ok', params)
    except Exception as e:
        print(f"Failed to load {key}: {e}")
        traceback.print_exc()
        results[key] = ('error', str(e))

print('\nSummary:')
for k, v in results.items():
    print(f"{k}: {v[0]} - {v[1]}")
