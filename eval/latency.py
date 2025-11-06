import time, torch
@torch.no_grad()
def measure_latency_s(model, example_batch, runs=30, warmup=5, device="cpu"):
    
    model.eval()
    model.to(device)

    # Handle different batch types
    if isinstance(example_batch, (list, tuple)):
        xb = example_batch[0].to(device)
    elif isinstance(example_batch, dict):
        xb = {k: v.to(device) for k, v in example_batch.items() if torch.is_tensor(v)}
    else:
        raise TypeError(f"Unsupported example batch type: {type(example_batch)}")

    # Warm-up (to stabilize GPU timing)
    for _ in range(warmup):
        _ = model(xb)

    # Timed runs
    torch.cuda.synchronize() if device == "cuda" else None
    start = time.perf_counter()

    for _ in range(runs):
        _ = model(xb)
    torch.cuda.synchronize() if device == "cuda" else None

    total = time.perf_counter() - start
    avg_latency = total / runs
    return avg_latency