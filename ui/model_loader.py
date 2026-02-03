# model_loader.py
import os
import torch
import glob
import tempfile
import torchvision.models as models
from torchvision.models import resnet18
from transformers import AutoModelForSequenceClassification, AutoModel

# Get project root directory (parent of ui/)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def _get_project_path(relative_path):
    """Convert relative path to absolute path based on project root."""
    return os.path.join(_PROJECT_ROOT, relative_path)


def _torch_load(path, map_location="cpu"):
    """Load checkpoint; use weights_only=False on PyTorch 2+ for state dicts."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


###############################################################################
# 1) --- HANDLERS -------------------------------------------------------------
###############################################################################

def load_hf_model(model_dir, device):
    """Loads an entire HuggingFace folder."""
    # Convert to absolute path if relative
    if not os.path.isabs(model_dir):
        model_dir = _get_project_path(model_dir)
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"HuggingFace model directory not found: {model_dir}")
    try:
        from transformers import DistilBertForSequenceClassification
    except Exception as e:
        raise ImportError("Missing dependency 'transformers'. Install it with `pip install transformers`.") from e
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return model

def load_hf_model_by_name(model_name, device):
    # Robust loader: try several AutoModel variants in order of likelihood
    # for common model types (image classification, sequence classification, masked LM, generic).
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoModelForImageClassification,
        AutoModelForSequenceClassification,
        AutoModelForMaskedLM,
    )

    # Try to inspect config first
    try:
        cfg = AutoConfig.from_pretrained(model_name)
    except Exception:
        cfg = None

    # Attempt image classification loader first if config hints at vision model
    tried = []
    if cfg is not None and getattr(cfg, "model_type", "").lower() in (
        "mobilenet_v2", "efficientnet", "resnet", "convnext", "beit", "swin", "vit"
    ):
        try:
            model = AutoModelForImageClassification.from_pretrained(model_name)
            model.to(device)
            model.eval()
            return model
        except Exception as e:
            tried.append(("AutoModelForImageClassification", str(e)))

    # Try sequence classification
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        tried.append(("AutoModelForSequenceClassification", str(e)))

    # Try masked LM / encoder models
    try:
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        tried.append(("AutoModelForMaskedLM", str(e)))

    # Fallback to generic AutoModel
    try:
        model = AutoModel.from_pretrained(model_name)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        tried.append(("AutoModel", str(e)))

    # If we reach here, aggregate attempts and try a secondary fallback.
    err_msgs = "; ".join([f"{k}: {v}" for k, v in tried])

    # Secondary fallback: some HF repos provide nonstandard checkpoint filenames
    # (e.g., model.pt, *.pth, or safetensors) and are not consumable via
    # transformers' from_pretrained. Try downloading the repo and loading any
    # candidate checkpoint into a torchvision image model (Mobilenet/EfficientNet).
    def _try_load_hf_checkpoint_as_torchvision(repo_id, device):
        try:
            from huggingface_hub import snapshot_download
        except Exception as e:
            return None, f"huggingface_hub unavailable: {e}"

        try:
            repo_dir = snapshot_download(repo_id=repo_id, repo_type="model")
        except Exception as e:
            return None, f"snapshot_download failed: {e}"

        # Search for common checkpoint file extensions and other runtime artifacts
        candidates = []
        for pat in ("*.pt", "*.pth", "*.bin", "*.safetensors", "*.onnx", "*.tflite", "*.pb", "*.n2x"):
            candidates.extend(glob.glob(os.path.join(repo_dir, pat)))

        # If no standard candidates, list repository files to provide diagnostics
        if not candidates:
            repo_files = []
            for root, _, files in os.walk(repo_dir):
                for fn in files:
                    repo_files.append(os.path.relpath(os.path.join(root, fn), repo_dir))
            sample = repo_files[:20]
            return None, f"no checkpoint-like files found in repo; files: {sample} (total {len(repo_files)})"

        # Try loading each candidate file and mapping to a torchvision model
        for fpath in candidates:
            # If this is an ONNX model, try to load with onnxruntime and wrap
            if fpath.lower().endswith(".onnx"):
                try:
                    import onnxruntime as ort
                except Exception as e:
                    # onnxruntime not available, include in diagnostic
                    return None, f"onnx model present ({fpath}) but onnxruntime missing: {e}"

                class _ONNXWrapper(torch.nn.Module):
                    def __init__(self, session):
                        super().__init__()
                        self.session = session

                    def to(self, device):
                        # ONNX runtime runs on CPU/GPU via providers; nothing to move
                        return self

                    def eval(self):
                        return self

                    def __call__(self, x):
                        # Expect a single input; convert torch tensor to numpy
                        import numpy as _np
                        if isinstance(x, torch.Tensor):
                            inp = x.detach().cpu().numpy()
                        else:
                            inp = _np.asarray(x)
                        input_name = self.session.get_inputs()[0].name
                        outputs = self.session.run(None, {input_name: inp})
                        # Return first output as torch tensor
                        out = outputs[0]
                        return torch.from_numpy(out)

                try:
                    sess = ort.InferenceSession(fpath, providers=["CPUExecutionProvider"])
                    wrapper = _ONNXWrapper(sess)
                    return wrapper, f"loaded ONNX model via onnxruntime from {fpath}"
                except Exception as e:
                    continue

            try:
                if fpath.endswith(".safetensors"):
                    try:
                        from safetensors.torch import load_file as safetensors_load
                        obj = safetensors_load(fpath)
                    except Exception:
                        continue
                else:
                    obj = torch.load(fpath, map_location="cpu")
            except Exception:
                continue

            # If it's a state_dict, try to load into a torchvision mobilenet_v2
            if isinstance(obj, dict):
                possible_sds = [obj]
                if "state_dict" in obj and isinstance(obj["state_dict"], dict):
                    possible_sds.insert(0, obj["state_dict"])
                if "model" in obj and isinstance(obj["model"], dict):
                    possible_sds.insert(0, obj["model"])

                for sd in possible_sds:
                    try:
                        tv_model = models.mobilenet_v2(weights=None)
                    except Exception:
                        tv_model = models.mobilenet_v2(pretrained=False)
                    try:
                        tv_model.load_state_dict(sd, strict=False)
                        tv_model.to(device)
                        tv_model.eval()
                        return tv_model, f"loaded torchvision model from {fpath}"
                    except Exception:
                        continue

            # If the object is a full model instance
            if isinstance(obj, torch.nn.Module):
                try:
                    obj.to(device)
                    obj.eval()
                    return obj, f"loaded full torch model from {fpath}"
                except Exception:
                    continue

        # If we couldn't load any candidate, build actionable diagnostics.
        exts_set = {os.path.splitext(p)[1].lower() for p in candidates}
        exts_list = sorted(exts_set)
        # If only vendor-specific artifacts (e.g., .n2x) are present, return
        # a clear message explaining conversion options.
        vendor_exts = {'.n2x'}
        if exts_set and exts_set.issubset(vendor_exts):
            model_url = f"https://huggingface.co/{repo_id}"
            return None, (
                f"repo contains only vendor-specific artifacts {exts_list}. "
                f"These formats are not directly loadable by PyTorch/transformers. "
                f"Check the model page for conversion instructions: {model_url}. "
                f"You can download the repo locally with: from huggingface_hub import snapshot_download; "
                f"snapshot_download('{repo_id}'). If the repo provides a converter script or an ONNX export, "
                f"use that to produce an ONNX (.onnx) or PyTorch checkpoint (.pt/.pth/.safetensors) and re-run the loader."
            )

        # Otherwise return a general message listing the candidate extensions
        return None, f"no compatible checkpoint loaded from repo; candidate extensions: {exts_list}"

    fallback_model, fallback_msg = _try_load_hf_checkpoint_as_torchvision(model_name, device)
    if fallback_model is not None:
        return fallback_model

    # If fallback failed, raise aggregated error including fallback diagnostic
    raise RuntimeError(
        f"Failed to load Hugging Face model '{model_name}'. Attempts: {err_msgs}; fallback: {fallback_msg}"
    )

def load_resnet18_state_dict(path, device, num_classes=10):
    """
    Load ResNet18 state dict for CIFAR-10 (10 classes).
    num_classes: number of output classes (default 10 for CIFAR-10)
    """
    import torch.nn as nn
    # Convert to absolute path if relative
    if not os.path.isabs(path):
        path = _get_project_path(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"ResNet18 model file not found: {path}")
    model = models.resnet18(weights=None)  # no pretrained weights
    # Modify final layer for CIFAR-10 (10 classes instead of 1000)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state_dict = _torch_load(path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_resnet18_ptq_quantized(path, device, num_classes=10):
    """
    Load PTQ (Post-Training Quantization) quantized ResNet18 model.
    This model uses torchvision's quantization-aware architecture.
    """
    try:
        import torch.ao.quantization as tq
    except ImportError:
        import torch.quantization as tq
    import torchvision.models.quantization as qmodels
    # Convert to absolute path if relative
    if not os.path.isabs(path):
        path = _get_project_path(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"PTQ quantized model file not found: {path}")
    
    # Build quantized model architecture
    qmodel = qmodels.resnet18(weights=None, num_classes=num_classes, quantize=False)
    qmodel.eval()  # fuse requires eval mode
    qmodel.fuse_model()
    qmodel.qconfig = tq.get_default_qconfig("fbgemm")
    tq.prepare(qmodel, inplace=True)
    qmodel = tq.convert(qmodel, inplace=False)
    qmodel.eval()
    
    # Load the quantized state dict
    state_dict = _torch_load(path)
    qmodel.load_state_dict(state_dict)
    qmodel.to(device)
    qmodel.eval()
    return qmodel


# QAT requires a custom QuantizableResNet18 architecture
def _make_qat_layer(inplanes, planes, blocks, stride=1, norm_layer=None):
    """Helper to create a layer of QuantBasicBlock for QAT."""
    import torch.nn as nn
    try:
        import torch.ao.quantization as tq
    except ImportError:
        import torch.quantization as tq
    if norm_layer is None:
        norm_layer = nn.BatchNorm2d
    
    downsample = None
    if stride != 1 or inplanes != planes:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
            norm_layer(planes),
        )
    
    layers = []
    layers.append(_QuantBasicBlock(inplanes, planes, stride, downsample, norm_layer))
    inplanes = planes
    for _ in range(1, blocks):
        layers.append(_QuantBasicBlock(inplanes, planes, norm_layer=norm_layer))
    return nn.Sequential(*layers), inplanes


class _QuantBasicBlock(torch.nn.Module):
    """Basic block for quantizable ResNet18 (used in QAT)."""
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super().__init__()
        import torch.nn as nn
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.skip_add = nn.quantized.FloatFunctional()
    
    def fuse_model(self):
        try:
            import torch.ao.quantization as tq
        except ImportError:
            import torch.quantization as tq
        tq.fuse_modules(self, ["conv1", "bn1", "relu"], inplace=True)
        tq.fuse_modules(self, ["conv2", "bn2"], inplace=True)
    
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.skip_add.add(out, identity)
        return self.relu(out)


class _QuantizableResNet18(torch.nn.Module):
    """Custom quantizable ResNet18 for QAT (matches training script)."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        import torch.nn as nn
        try:
            import torch.ao.quantization as tq
        except ImportError:
            import torch.quantization as tq
        norm = nn.BatchNorm2d
        inplanes = 64
        self.conv1 = nn.Conv2d(3, inplanes, 7, stride=2, padding=3, bias=False)
        self.bn1 = norm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1, inplanes = _make_qat_layer(inplanes, 64, 2, stride=1, norm_layer=norm)
        self.layer2, inplanes = _make_qat_layer(inplanes, 128, 2, stride=2, norm_layer=norm)
        self.layer3, inplanes = _make_qat_layer(inplanes, 256, 2, stride=2, norm_layer=norm)
        self.layer4, inplanes = _make_qat_layer(inplanes, 512, 2, stride=2, norm_layer=norm)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * _QuantBasicBlock.expansion, num_classes)
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()
    
    def fuse_model(self):
        try:
            import torch.ao.quantization as tq
        except ImportError:
            import torch.quantization as tq
        tq.fuse_modules(self, ["conv1", "bn1", "relu"], inplace=True)
        for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
            for b in getattr(self, layer_name):
                b.fuse_model()
    
    def forward(self, x):
        x = self.quant(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.dequant(x)
        return x


def load_resnet18_qat_quantized(path, device, num_classes=10):
    """
    Load QAT (Quantization-Aware Training) quantized ResNet18 model.
    This uses a custom QuantizableResNet18 architecture.
    """
    try:
        import torch.ao.quantization as tq
    except ImportError:
        import torch.quantization as tq
    
    # Convert to absolute path if relative
    if not os.path.isabs(path):
        path = _get_project_path(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"QAT quantized model file not found: {path}")
    
    # Build the custom QuantizableResNet18 architecture
    qmodel = _QuantizableResNet18(num_classes=num_classes)
    qmodel.eval()
    qmodel.fuse_model()  # Fuse in eval mode
    qmodel.train()  # QAT prepare requires TRAIN mode
    qmodel.qconfig = tq.get_default_qat_qconfig("fbgemm")
    tq.prepare_qat(qmodel, inplace=True)
    qmodel = tq.convert(qmodel, inplace=False)  # Convert to quantized ops
    qmodel.eval()
    
    # Load the quantized state dict
    state_dict = _torch_load(path)
    missing, unexpected = qmodel.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[QAT warn] Missing keys (first 5): {list(missing)[:5]}")
    if unexpected:
        print(f"[QAT warn] Unexpected keys (first 5): {list(unexpected)[:5]}")
    
    qmodel.to(device)
    qmodel.eval()
    return qmodel


def load_state_dict_model(baseline_dir, state_dict_path, device, is_dynamic_quantized=False):
    """
    Loads a model architecture from baseline_dir and applies a state_dict.
    is_dynamic_quantized: If True, applies dynamic quantization before loading state_dict.
    """
    # Convert to absolute paths if relative
    if not os.path.isabs(baseline_dir):
        baseline_dir = _get_project_path(baseline_dir)
    if not os.path.isabs(state_dict_path):
        state_dict_path = _get_project_path(state_dict_path)
    
    if not os.path.exists(baseline_dir):
        raise FileNotFoundError(f"Baseline model directory not found: {baseline_dir}")
    if not os.path.exists(state_dict_path):
        raise FileNotFoundError(f"State dict file not found: {state_dict_path}")
    
    print("Loading baseline architecture:", baseline_dir)
    try:
        from transformers import DistilBertForSequenceClassification
    except Exception as e:
        raise ImportError("Missing dependency 'transformers'. Install it with `pip install transformers`.") from e
    model = DistilBertForSequenceClassification.from_pretrained(baseline_dir)
    
    # For dynamic quantized models, we need to apply quantization first
    # This converts Linear layers to DynamicQuantizedLinear
    if is_dynamic_quantized:
        try:
            import torch.ao.quantization as tq
        except ImportError:
            import torch.quantization as tq
        print("Applying dynamic quantization before loading state_dict...")
        model = tq.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    
    print("Loading state_dict:", state_dict_path)
    state_dict = _torch_load(state_dict_path)
    
    # Load with strict=False for quantized models to handle any mismatches
    missing, unexpected = model.load_state_dict(state_dict, strict=not is_dynamic_quantized)
    if is_dynamic_quantized:
        if missing:
            print(f"[Dynamic quant warn] Missing keys (first 5): {list(missing)[:5]}")
        if unexpected:
            print(f"[Dynamic quant warn] Unexpected keys (first 5): {list(unexpected)[:5]}")
    elif missing or unexpected:
        # For non-quantized models, strict loading should work
        raise RuntimeError(f"Failed to load state_dict. Missing: {missing}, Unexpected: {unexpected}")

    model.to(device)
    model.eval()
    return model


def load_user_uploaded_model(path, device, model_type=None):
    """
    Smart loader:
    - If directory → treat as HuggingFace model folder
    - If .pt or .bin → treat as a raw state_dict
    - model_type: "ResNet18" or "DistilBERT" to determine which baseline to use
    """
    if os.path.isdir(path):
        return load_hf_model(path, device)

    if path.endswith(".pt") or path.endswith(".pth") or path.endswith(".bin"):
        # Determine which baseline to use based on model_type
        if model_type == "ResNet18":
            # For ResNet18, load directly as state_dict
            return load_resnet18_state_dict(path, device)
        elif model_type == "DistilBERT":
            # For DistilBERT, use baseline architecture
            BASELINE = "models/distilbert_baseline"
            return load_state_dict_model(BASELINE, path, device)
        else:
            # Try to auto-detect: if model_type not provided, default to DistilBERT
            # but warn the user
            import warnings
            warnings.warn(f"Model type not specified for uploaded file. Defaulting to DistilBERT. "
                         f"Please specify model_type if this is a ResNet18 model.")
            BASELINE = "models/distilbert_baseline"
            try:
                return load_state_dict_model(BASELINE, path, device)
            except Exception:
                # If DistilBERT fails, try ResNet18
                return load_resnet18_state_dict(path, device)

    raise ValueError(f"Unsupported user model format: {path}")


###############################################################################
# 2) --- REGISTRY -------------------------------------------------------------
###############################################################################

MODEL_REGISTRY = {
    # ---- Baseline FP32 -------------------------------------------------------
    "distilbert_baseline": {
        "type": "hf_dir",
        "path": "models/distilbert_baseline",
    },

    # ---- Dynamic Quant (INT8) ------------------------------------------------
    "distilbert_dynamic_int8": {
        "type": "state_dict_dynamic_quant",
        "path": "models/distilbert_dynamic_int8.pt",
        "baseline": "models/distilbert_baseline",
    },

    # ---- Static Quant (INT8) -------------------------------------------------
    "distilbert_static_int8": {
        "type": "state_dict",
        "path": "models/distilbert_static_int8.pt",
        "baseline": "models/distilbert_baseline",
    },

    # ---- QAT (INT8) ----------------------------------------------------------
    "distilbert_qat_int8": {
        "type": "state_dict",
        "path": "models/distilbert_qat.pt",
        "baseline": "models/distilbert_baseline",
    },

    # ---- User upload placeholder --------------------------------------------
    "user_upload": {
        "type": "user",
        "path": None,
    },
    
    # ================================
    # ResNet18 MODELS
    # ================================
    "resnet18_baseline": {
        "type": "resnet_state_dict",
        "path": "models/resnet18_baseline.pt",
    },
    "resnet18_ptq": {
        "type": "resnet_ptq_quantized",
        "path": "models/resnet18_quantized_ptq.pt",
    },
    "resnet18_qat": {
        "type": "resnet_qat_quantized",
        "path": "models/resnet18_quantized_qat.pt",
    },
}


###############################################################################
# 3) --- LOADABLE CHECK (for planner / UI) ------------------------------------
###############################################################################

def is_model_loadable(model_key: str) -> bool:
    """
    Return True if the model key is in the registry and its required file(s) exist.
    Use this to avoid trying to load variants that don't have files (e.g. not yet generated).
    """
    cfg = MODEL_REGISTRY.get(model_key)
    if cfg is None:
        return False
    t = cfg.get("type")
    if t == "user":
        return False
    if t == "hf_dir":
        p = cfg["path"]
        path_abs = _get_project_path(p) if not os.path.isabs(p) else p
        return os.path.exists(path_abs)
    if t in ("state_dict", "state_dict_dynamic_quant"):
        p = cfg["path"]
        path_abs = _get_project_path(p) if not os.path.isabs(p) else p
        return os.path.exists(path_abs)
    if t in ("resnet_state_dict", "resnet_ptq_quantized", "resnet_qat_quantized"):
        p = cfg["path"]
        path_abs = _get_project_path(p) if not os.path.isabs(p) else p
        return os.path.exists(path_abs)
    return False


###############################################################################
# 4) --- MASTER LOADER --------------------------------------------------------
###############################################################################

def load_model(model_key, device, user_path=None, user_model_type=None):
    """
    Unified loader used by your UI.
    model_key = dropdown selection
    user_path = path chosen by user
    user_model_type = "ResNet18" or "DistilBERT" for user-uploaded models
    """
    cfg = MODEL_REGISTRY.get(model_key)
    if cfg is None:
        raise ValueError(f"Unknown model: {model_key}")

    model_type = cfg["type"]

    # ------------------------- USER UPLOADED ---------------------------------
    if model_type == "user":
        if not user_path:
            raise ValueError("user_path must be provided for user-uploaded models.")
        return load_user_uploaded_model(user_path, device, model_type=user_model_type)

    # -------------------------- HF DIRECTORY ---------------------------------
    if model_type == "hf_dir":
        return load_hf_model(cfg["path"], device)

    # ------------------------- STATE_DICT MODEL -------------------------------
    if model_type == "state_dict":
        baseline = cfg["baseline"]
        state_dict = cfg["path"]
        # Check if files exist before attempting to load
        state_dict_abs = _get_project_path(state_dict) if not os.path.isabs(state_dict) else state_dict
        if not os.path.exists(state_dict_abs):
            raise FileNotFoundError(
                f"Model file not found: {state_dict}\n"
                f"Expected at: {state_dict_abs}\n"
                f"This model variant may not be available yet. Please generate it first or use a different variant."
            )
        return load_state_dict_model(baseline, state_dict, device, is_dynamic_quantized=False)
    
    # ------------------------- STATE_DICT DYNAMIC QUANTIZED MODEL -------------
    if model_type == "state_dict_dynamic_quant":
        baseline = cfg["baseline"]
        state_dict = cfg["path"]
        # Check if files exist before attempting to load
        state_dict_abs = _get_project_path(state_dict) if not os.path.isabs(state_dict) else state_dict
        if not os.path.exists(state_dict_abs):
            raise FileNotFoundError(
                f"Dynamic quantized model file not found: {state_dict}\n"
                f"Expected at: {state_dict_abs}\n"
                f"This model variant may not be available yet. Please generate it first or use a different variant."
            )
        return load_state_dict_model(baseline, state_dict, device, is_dynamic_quantized=True)
    
    if model_type == "resnet_state_dict":
        # Check if file exists before attempting to load
        path_abs = _get_project_path(cfg["path"]) if not os.path.isabs(cfg["path"]) else cfg["path"]
        if not os.path.exists(path_abs):
            raise FileNotFoundError(
                f"Model file not found: {cfg['path']}\n"
                f"Expected at: {path_abs}\n"
                f"This model variant may not be available yet. Please generate it first or use a different variant."
            )
        return load_resnet18_state_dict(cfg["path"], device)
    
    # ------------------------- PTQ QUANTIZED MODEL -------------------------------
    if model_type == "resnet_ptq_quantized":
        # Check if file exists before attempting to load
        path_abs = _get_project_path(cfg["path"]) if not os.path.isabs(cfg["path"]) else cfg["path"]
        if not os.path.exists(path_abs):
            raise FileNotFoundError(
                f"PTQ quantized model file not found: {cfg['path']}\n"
                f"Expected at: {path_abs}\n"
                f"This model variant may not be available yet. Please generate it first or use a different variant."
            )
        return load_resnet18_ptq_quantized(cfg["path"], device)
    
    # ------------------------- QAT QUANTIZED MODEL -------------------------------
    if model_type == "resnet_qat_quantized":
        # Check if file exists before attempting to load
        path_abs = _get_project_path(cfg["path"]) if not os.path.isabs(cfg["path"]) else cfg["path"]
        if not os.path.exists(path_abs):
            raise FileNotFoundError(
                f"QAT quantized model file not found: {cfg['path']}\n"
                f"Expected at: {path_abs}\n"
                f"This model variant may not be available yet. Please generate it first or use a different variant."
            )
        return load_resnet18_qat_quantized(cfg["path"], device)

    raise ValueError(f"Unsupported model type: {model_type}")
