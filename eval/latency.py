import time, torch
@torch.no_grad()
def measure_latency_s(model, example_batch, runs=10, warmup=2, device="cpu"):
    """
    Measures average latency (in seconds) of a single forward pass.
    Works for both image tensors and text token dictionaries.
    """
    model.eval()

    # Move example to device
    if isinstance(example_batch, dict):
        # NLP models (e.g., DistilBERT)
        example_batch = {k: v.to(device) for k, v in example_batch.items()}
    elif isinstance(example_batch, (list, tuple)):
        # Just take the first tensor in case it's a list
        example_batch = example_batch[0].to(device)
    else:
        # Normal tensor
        example_batch = example_batch.to(device)

    # Warmup
    for _ in range(warmup):
        _ = model(**example_batch) if isinstance(example_batch, dict) else model(example_batch)

    # Timed runs
    start = time.perf_counter()
    for _ in range(runs):
        _ = model(**example_batch) if isinstance(example_batch, dict) else model(example_batch)
    end = time.perf_counter()

    latency = (end - start) / runs
    return latency