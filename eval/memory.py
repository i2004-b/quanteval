import os, tempfile, torch, gc #psutil
def param_bytes(model): 
    return sum(p.numel() * p.element_size() for p in model.parameters())
def on_disk_bytes_state_dict(model):
    with tempfile.NamedTemporaryFile(delete=False) as f:
        torch.save(model.state_dict(), f.name)
        size = os.path.getsize(f.name)
    os.remove(f.name); return size
def peak_gpu_mem_once(model, example_batch):
    if not torch.cuda.is_available(): return 0
    device = torch.device("cuda")
    xb = example_batch[0].to(device)
    model.eval().to(device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode(): _ = model(xb)
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device)
