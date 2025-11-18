# model_loader.py
import os
import torch
from transformers import DistilBertForSequenceClassification


###############################################################################
# 1) --- HANDLERS -------------------------------------------------------------
###############################################################################

def load_hf_model(model_dir, device):
    """Loads an entire HuggingFace folder."""
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return model


def load_state_dict_model(baseline_dir, state_dict_path, device):
    """Loads a model architecture from baseline_dir and applies a state_dict."""
    print("Loading baseline architecture:", baseline_dir)
    model = DistilBertForSequenceClassification.from_pretrained(baseline_dir)

    print("Loading state_dict:", state_dict_path)
    state_dict = torch.load(state_dict_path, map_location="cpu")
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model


def load_user_uploaded_model(path, device):
    """
    Smart loader:
    - If directory → treat as HuggingFace model folder
    - If .pt or .bin → treat as a raw state_dict
    """
    if os.path.isdir(path):
        return load_hf_model(path, device)

    if path.endswith(".pt") or path.endswith(".bin"):
        BASELINE = "models/distilbert_baseline"
        return load_state_dict_model(BASELINE, path, device)

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
        "type": "state_dict",
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
}


###############################################################################
# 3) --- MASTER LOADER --------------------------------------------------------
###############################################################################

def load_model(model_key, device, user_path=None):
    """
    Unified loader used by your UI.
    model_key = dropdown selection
    user_path = path chosen by user
    """
    cfg = MODEL_REGISTRY.get(model_key)
    if cfg is None:
        raise ValueError(f"Unknown model: {model_key}")

    model_type = cfg["type"]

    # ------------------------- USER UPLOADED ---------------------------------
    if model_type == "user":
        if not user_path:
            raise ValueError("user_path must be provided for user-uploaded models.")
        return load_user_uploaded_model(user_path, device)

    # -------------------------- HF DIRECTORY ---------------------------------
    if model_type == "hf_dir":
        return load_hf_model(cfg["path"], device)

    # ------------------------- STATE_DICT MODEL -------------------------------
    if model_type == "state_dict":
        baseline = cfg["baseline"]
        state_dict = cfg["path"]
        return load_state_dict_model(baseline, state_dict, device)

    raise ValueError(f"Unsupported model type: {model_type}")
