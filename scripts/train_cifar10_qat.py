# scripts/train_cifar10_qat.py
# Quantization-Aware Training (QAT) of a ResNet18 on CIFAR-10 using PyTorch.
# Starts from a pretrained FP32 baseline model (models/resnet18_baseline.pth)
# Outputs quantized INT8 model weights to models/resnet18_quantized_qat.pth
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.quantization as tq

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters for QAT fine-tuning
num_epochs = 5  # QAT fine-tuning epochs (starting from pre-trained model)
batch_size = 128
learning_rate = 1e-4  # smaller LR for fine-tuning

# 1. Load the baseline trained model weights
baseline_weights = "../models/resnet18_baseline.pth"
# Use a quantization-friendly ResNet18 architecture
# (QuantizableResNet adds quant/dequant stubs for quantization)
quantizable_model = torchvision.models.quantization.resnet18(weights=None, num_classes=10, quantize=False)
quantizable_model.load_state_dict(torch.load(baseline_weights), strict=False)
# strict=False to ignore missing keys for quant/dequant stubs (if any)
print("Loaded baseline weights into quantizable ResNet18 model.")

# 2. Prepare data loaders (same as in train_cifar10.py)
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# 3. Prepare the model for QAT
# First, fuse Conv, BN, ReLU modules to single units where appropriate.
quantizable_model.fuse_model(is_qat=True)  # fuse conv/bn/relu for QAT
print("Fused model layers for QAT.")

# Setup QAT configuration
quantizable_model.qconfig = tq.get_default_qat_qconfig('fbgemm')
# Note: 'fbgemm' is the quantization backend optimized for x86 CPUs (use 'qnnpack' for ARM/mobile).
# The default QAT qconfig will insert fake quantization modules that simulate INT8 quantization during training.

# Move model to device (GPU for QAT training) then prepare QAT.
quantizable_model.to(device)
tq.prepare_qat(quantizable_model, inplace=True)
print("Prepared model for Quantization-Aware Training (QAT).")

# 4. Fine-tune (train) the model with QAT
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(quantizable_model.parameters(), lr=learning_rate)
print("Starting QAT fine-tuning...")
for epoch in range(1, num_epochs+1):
    quantizable_model.train()
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = quantizable_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (batch_idx + 1) % 100 == 0:
            print(f"QAT Epoch [{epoch}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    avg_loss = running_loss / len(train_loader)
    print(f"QAT Epoch [{epoch}/{num_epochs}] completed, Average Loss: {avg_loss:.4f}")
    
    # We can optionally freeze quantization parameters after a couple epochs to fine-tune further 
    # but for simplicity we won't do that here.

# 5. Convert to quantized INT8 model
quantizable_model.to('cpu')  # conversion must happen on CPU
quantizable_model.eval()
# At conversion, the model's fake-quant layers will be replaced with real quantized ops
quantized_model = tq.convert(quantizable_model, inplace=False)
print("Converted QAT-trained model to actual quantized INT8 model.")

# Evaluate the quantized model on test set
quantized_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to('cpu'), labels.to('cpu')  # ensure data on CPU for quantized model
        outputs = quantized_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
qat_accuracy = 100 * correct / total
print(f"QAT Quantized ResNet18 Accuracy on CIFAR-10 test set: {qat_accuracy:.2f}%")

# 6. Save the quantized model weights
save_path = "../models/resnet18_quantized_qat.pth"
torch.save(quantized_model.state_dict(), save_path)
print(f"Saved QAT INT8 model weights to {save_path}")
