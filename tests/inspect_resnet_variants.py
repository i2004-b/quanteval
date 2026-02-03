import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from ui.model_loader import load_model
from ui.model_inspection import inspect_model_from_registry

device = torch.device('cpu')
keys = ['resnet18_baseline','resnet18_ptq','resnet18_qat']
for k in keys:
    try:
        m = load_model(k, device)
        md = inspect_model_from_registry(k, m, str(device))
        print(f"Model: {k}")
        print('  quantization_method:', md.quantization_method)
        print('  precision:', md.precision)
        print('  architecture:', md.architecture)
        print('  dataset:', md.dataset)
        print('  num_classes:', md.num_classes)
        print('  confidence:', md.confidence)
        print('  state_dict_keys sample:', list(m.state_dict().keys())[:10])
        # Print module names containing 'quant' or 'dequant'
        quant_names = [name for name, module in m.named_modules() if 'quant' in name.lower() or 'dequant' in name.lower()]
        print('  modules with quant/dequant in name:', quant_names[:20])
    except Exception as e:
        print(f"Failed for {k}: {e}")
        import traceback; traceback.print_exc()
