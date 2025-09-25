import time, torch
@torch.no_grad()
def measure_latency_s(model, example_batch, runs=50, warmup=10, device="cpu"):
    model.eval().to(device)
    xb = example_batch[0].to(device)
    if isinstance(xb, (list, tuple)): xb = xb[0]
    for _ in range(warmup):
        _ = model(xb); 
        if device == "cuda": torch.cuda.synchronize()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = model(xb)
        if device == "cuda": torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return sum(ts)/len(ts)
