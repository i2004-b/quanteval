# scripts/train_cifar10.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Device configuration: use GPU if available, else CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters for training
num_epochs = 10  # You can increase this for higher accuracy (e.g., 20 or 30)
batch_size = 128
learning_rate = 0.001

# 1. Load CIFAR-10 dataset
# CIFAR-10 has 50k train and 10k test 32x32 color images in 10 classes.
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))  # mean/std for CIFAR-10
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# 2. Define the ResNet18 model (for 10 classes)
model = torchvision.models.resnet18(weights=None, num_classes=10)  # untrained ResNet18
# Note: weights=None means we don't load any pretrained weights; we train from scratch on CIFAR-10.
model = model.to(device)

# For training smaller models, we could consider half precision (FP16), but we'll stick to FP32 for baseline.

# 3. Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 4. Training loop
print("Starting training ResNet18 on CIFAR-10...")
for epoch in range(1, num_epochs+1):
    model.train()
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        # Print progress for every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch}/{num_epochs}] completed, Average Loss: {avg_loss:.4f}")

# 5. After training, evaluate on test set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        # Predicted class is the one with highest score
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
baseline_accuracy = 100 * correct / total
print(f"Baseline ResNet18 Accuracy on CIFAR-10 test set: {baseline_accuracy:.2f}%")

# 6. Save the trained model weights
save_path = "../models/resnet18_baseline.pth"
torch.save(model.state_dict(), save_path)
print(f"Saved baseline model weights to {save_path}")
