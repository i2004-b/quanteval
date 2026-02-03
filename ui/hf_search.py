from huggingface_hub import HfApi, hf_hub_download

def is_pytorch_model(model_id: str) -> bool:
    """
    Check if a Hugging Face model has PyTorch weights.
    """
    api = HfApi()
    try:
        files = api.list_repo_files(model_id)
        return any(
            f in files
            for f in ["pytorch_model.bin", "model.safetensors"]
        )
    except Exception:
        return False


def search_hf_models(architecture: str, limit: int = 15):
    api = HfApi()

    query = architecture.lower()

    models = api.list_models(search=query, limit=limit)

    results = []
    for m in models:
        name = m.modelId

        # Prefer quantized models
        if any(x in name.lower() for x in ["int8", "quant", "qat", "ptq"]):
            if is_pytorch_model(name):
                results.append(name)

    # fallback: normal PyTorch models only
    if not results:
        for m in models:
            if is_pytorch_model(m.modelId):
                results.append(m.modelId)

    return results[:5]  # keep UI clean