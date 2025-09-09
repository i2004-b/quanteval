import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CIFAR_PATH = "outputs/baselines/resnet18_cifar10.pt"
SST2_PATH = "outputs/baselines/distilbert_sst2"

# ---------------- CIFAR-10 Test ----------------
def test_cifar10():
    if not os.path.exists(CIFAR_PATH):
        print(f"❌ CIFAR-10 checkpoint not found at {CIFAR_PATH}")
        return

    print("🔹 Testing CIFAR-10 ResNet18...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=16, shuffle=False)

    model = resnet18(weights="IMAGENET1K_V1")
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(CIFAR_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    images, labels = next(iter(testloader))
    images, labels = images.to(DEVICE), labels.to(DEVICE)

    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

    print(f"✅ CIFAR-10 batch preds: {preds[:10].tolist()} (labels: {labels[:10].tolist()})")

# ---------------- SST-2 Test ----------------
def test_sst2():
    if not os.path.exists(SST2_PATH):
        print(f"❌ SST-2 checkpoint not found at {SST2_PATH}")
        return

    print("🔹 Testing SST-2 DistilBERT...")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(SST2_PATH)
    model = model.to(DEVICE)
    model.eval()

    sentences = [
        "I loved this movie, it was fantastic!",
        "This was the worst film I have ever seen."
    ]
    inputs = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=-1)

    print(f"✅ SST-2 predictions: {preds.tolist()} for sentences: {sentences}")

# ---------------- Main ----------------
if __name__ == "__main__":
    test_cifar10()
    test_sst2()
    