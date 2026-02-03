import torch
from torchvision import models
from torchvision.models.quantization import resnet18
import os

# Create output directory
os.makedirs("test_models", exist_ok=True)

# ---- Baseline FP32 model ----
print("Downloading FP32 ResNet18...")
fp32_model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)
fp32_model.eval()
torch.save({
    "arch": "resnet18",
    "quantized": False,
    "state_dict": fp32_model.state_dict()
}, "test_models/fp32_model.pt")

# ---- Real INT8 quantized model (CPU) ----
print("Downloading INT8 Quantized ResNet18...")
int8_model = resnet18(
    weights="ResNet18_QuantizedWeights.DEFAULT",
    quantize=True
)
int8_model.eval()
torch.save({
    "arch": "resnet18",
    "quantized": True,
    "state_dict": int8_model.state_dict()
}, "test_models/int8_model.pt")

print("Done. Models are downloaded and ready.")
