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


def search_hf_models_debug(architecture: str, limit: int = 20):
    """
    Debugging search: run multiple query variants and return per-query diagnostics.

    Returns:
      (candidates: list[str], diagnostics: list[dict])
    """
    api = HfApi()
    arch = architecture.strip()
    queries = [
        arch,
        arch.lower(),
        arch.replace(" ", "-"),
        f"{arch} cifar10",
        f"{arch} sst2",
        f"{arch} resnet",
        f"{arch} mobilenet",
        f"{arch} efficientnet",
        f"{arch} quant",
        f"{arch} int8",
    ]

    seen = set()
    candidates = []
    diagnostics = []

    for q in queries:
        try:
            models = api.list_models(search=q, limit=limit)
            model_ids = [m.modelId for m in models]
            pytorch_counts = 0
            sample = []
            for mid in model_ids[:10]:
                ok = False
                try:
                    ok = is_pytorch_model(mid)
                except Exception:
                    ok = False
                if ok:
                    pytorch_counts += 1
                sample.append({"id": mid, "pytorch": ok})

            diagnostics.append({
                "query": q,
                "total_found": len(model_ids),
                "pytorch_count": pytorch_counts,
                "sample": sample[:5],
            })

            # Prefer quantized/intentional names first
            for mid in model_ids:
                low = mid.lower()
                if mid in seen:
                    continue
                if any(x in low for x in ["int8", "quant", "qat", "ptq"]):
                    if is_pytorch_model(mid):
                        candidates.append(mid)
                        seen.add(mid)

            # Fallback: add any pytorch models
            for mid in model_ids:
                if mid in seen:
                    continue
                try:
                    if is_pytorch_model(mid):
                        candidates.append(mid)
                        seen.add(mid)
                except Exception:
                    continue

        except Exception as e:
            diagnostics.append({"query": q, "error": str(e)})
            continue

    # Trim and return
    return candidates[:10], diagnostics