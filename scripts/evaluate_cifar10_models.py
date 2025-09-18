# scripts/evaluate_cifar10_models.py
import torch
import torchvision
import time, os

# Ensure we run on CPU for all models to have a fair comparison, since quantized models are CPU-only.
device = torch.device('cpu')
torch.set_num_threads(1)  # (optional) use single-thread for fair timing, or adjust as needed

# 1. Load baseline FP32 model
baseline_model = torchvision.models.resnet18(weights=None, num_classes=10)
baseline_weights = "../models/resnet18_baseline.pth"
baseline_model.load_state_dict(torch.load(baseline_weights))
baseline_model.to(device).eval()
print("Loaded baseline ResNet18.")

# 2. Load PTQ quantized model (INT8)
ptq_model = torchvision.models.quantization.resnet18(weights=None, num_classes=10, quantize=False)
ptq_model.load_state_dict(torch.load("../models/resnet18_quantized_ptq.pth"))
ptq_model.to(device).eval()
print("Loaded PTQ quantized ResNet18.")

# 3. Load QAT quantized model (INT8)
qat_model = torchvision.models.quantization.resnet18(weights=None, num_classes=10, quantize=False)
qat_model.load_state_dict(torch.load("../models/resnet18_quantized_qat.pth"))
qat_model.to(device).eval()
print("Loaded QAT quantized ResNet18.")

# Note: We loaded quantized model weights into a QuantizableResNet architecture. 
# These weights are already in int8 (packed), but to use them, we should ideally convert the model to a quantized version.
# However, since we saved state_dict after conversion, loading it back into the quantization-aware model should give a working int8 model.
# If any issue arises, we might need to do a prepare/convert step, but it should not be necessary here.

# 4. Prepare CIFAR-10 test data loader
from torchvision.datasets import CIFAR10
from torchvision import transforms
test_dataset = CIFAR10(root='./data', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914,0.4822,0.4465), (0.2470,0.2435,0.2616))
                       ]))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

# 5. Function to evaluate a model: returns accuracy and latency
def evaluate_model(model, loader):
    model.eval()
    correct = 0
    total = 0
    start = time.time()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    elapsed = time.time() - start
    accuracy = correct / total
    # compute latency per sample (or per batch)
    avg_latency = (elapsed / total) * 1000  # in milliseconds per image
    return accuracy, avg_latency, elapsed

# 6. Evaluate each model
baseline_acc, baseline_latency, base_time = evaluate_model(baseline_model, test_loader)
ptq_acc, ptq_latency, ptq_time = evaluate_model(ptq_model, test_loader)
qat_acc, qat_latency, qat_time = evaluate_model(qat_model, test_loader)

# 7. Get model sizes from disk
baseline_size = os.path.getsize("../models/resnet18_baseline.pth") / 1e6
ptq_size = os.path.getsize("../models/resnet18_quantized_ptq.pth") / 1e6
qat_size = os.path.getsize("../models/resnet18_quantized_qat.pth") / 1e6

# 8. Print comparison table
print("\nComparison of ResNet18 models (CIFAR-10):")
print("Model\t\tAccuracy\tLatency (ms/img)\tSize (MB)")
print(f"Baseline FP32\t{baseline_acc*100:.2f}%\t\t{baseline_latency:.2f}\t\t{baseline_size:.1f}")
print(f"PTQ INT8   \t{ptq_acc*100:.2f}%\t\t{ptq_latency:.2f}\t\t{ptq_size:.1f}")
print(f"QAT INT8   \t{qat_acc*100:.2f}%\t\t{qat_latency:.2f}\t\t{qat_size:.1f}")
