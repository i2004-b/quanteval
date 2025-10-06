# scripts/quantize_cifar10_ptq.py
# Post-Training Quantization (PTQ) of a pretrained ResNet18 on CIFAR-10 using PyTorch.
# Loads a FP32 baseline model from models/resnet18_baseline.pth
import torch
import torchvision
import torch.quantization as tq
import time, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]        #uses path relative to the repo
BASELINE = ROOT / "models" / "resnet18_baseline.pt"

# 1. Load the FP32 baseline ResNet18 model
fp32_weights = str(BASELINE)
# Use the quantization-aware resnet18 architecture to load weights, so we can easily fuse and quantize
model = torchvision.models.quantization.resnet18(weights=None, num_classes=10, quantize=False)
model.load_state_dict(torch.load(fp32_weights))
model.eval()
print("Loaded FP32 ResNet18 baseline model.")

# 2. Fuse the model's layers (conv/bn/relu) to prepare for quantization
model.fuse_model()  # fuse_model with default is_qat=False for PTQ
print("Fused Conv, BN, ReLU layers for PTQ.")

# 3. Set quantization configuration and prepare the model
model.qconfig = tq.get_default_qconfig('fbgemm')
# (If running on ARM devices, use 'qnnpack' instead of 'fbgemm')
tq.prepare(model, inplace=True)
print("Inserted observers for PTQ (calibration).")

# 4. Calibration step – run the model on a subset of data to collect activation stats
# We'll use the training dataset or a part of it for calibration. For simplicity, use the test set itself to calibrate.
# Load CIFAR-10 test data
from torchvision.datasets import CIFAR10
from torchvision import transforms
calib_dataset = CIFAR10(root='./data', train=False, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914,0.4822,0.4465), (0.2470,0.2435,0.2616))
                        ]))
calib_loader = torch.utils.data.DataLoader(calib_dataset, batch_size=128, shuffle=False)
# Run through a few batches to calibrate (we don't need to use all data; let's use 1000 samples or so)
num_calib_batches = 10
batch_count = 0
with torch.no_grad():
    for images, labels in calib_loader:
        model(images)  # simply feed forward to collect stats
        batch_count += 1
        if batch_count >= num_calib_batches:
            break
print(f"Calibration done using {batch_count*calib_loader.batch_size} samples.")

# 5. Convert the model to quantized INT8 version
quantized_model = tq.convert(model, inplace=False)
quantized_model.eval()
print("Converted model to INT8 (static quantization).")

# 6. Evaluate the quantized model on the test set to get accuracy
test_dataset = CIFAR10(root='./data', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914,0.4822,0.4465), (0.2470,0.2435,0.2616))
                       ]))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
correct = 0
total = 0
start_time = time.time()
with torch.no_grad():
    for images, labels in test_loader:
        outputs = quantized_model(images)  # model is on CPU by default after quantization
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
inference_time = time.time() - start_time
accuracy = 100 * correct / total
print(f"PTQ INT8 ResNet18 Accuracy: {accuracy:.2f}% (vs FP32 baseline).")
print(f"Inference time on CPU for {total} images: {inference_time:.2f} seconds.")

# 7. Save the quantized model weights
MODELS = ROOT / "models"

save_path = "models/resnet18_quantized_ptq.pt"
torch.save(quantized_model.state_dict(), save_path)
print(f"Saved PTQ INT8 model to {save_path}")

# 8. Report model size reduction
fp32_size = os.path.getsize(fp32_weights) / 1e6  # in MB
int8_size = os.path.getsize(save_path) / 1e6     # in MB
print(f"FP32 model size: {fp32_size:.2f} MB, Quantized INT8 model size: {int8_size:.2f} MB")

# --- JSON report (append-only) ---
import json
from datetime import datetime

# where to store the report
REPORT_DIR = ROOT / "outputs" / "reports" / "resnet18_cifar10_ptq"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = REPORT_DIR / "report.json"

# compute a couple extra fields
per_image_ms = (inference_time / total) * 1000.0 if total else None
calibration_samples = min(num_calib_batches * calib_loader.batch_size, len(calib_dataset))

report = {
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "task": "cifar10",
    "model": "resnet18",
    "quant_method": "ptq_static_int8",
    "backend": "fbgemm",
    "paths": {
        "baseline_fp32": fp32_weights,
        "quantized_int8": save_path,
        "report_json": str(REPORT_PATH),
        "repo_root": str(ROOT),
    },
    "settings": {
        "num_calib_batches": num_calib_batches,
        "calibration_samples": calibration_samples,
        "test_batch_size": 128
    },
    "metrics": {
        "accuracy_top1_percent": round(accuracy, 2),
        "inference_time_total_s": round(inference_time, 3),
        "latency_per_image_ms": None if per_image_ms is None else round(per_image_ms, 4),
        "model_size_mb": {
            "fp32": round(fp32_size, 2),
            "int8": round(int8_size, 2)
        }
    },
    "env": {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "device_eval": "cpu"
    }
}

with open(REPORT_PATH, "w") as f:
    json.dump(report, f, indent=2)

print(f"Wrote PTQ JSON report -> {REPORT_PATH}")
# --- end JSON report ---
