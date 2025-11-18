import torch, json
from torchvision import transforms

CIFAR10_NORM = transforms.Normalize((0.4914,0.4822,0.4465),
                                    (0.2470,0.2435,0.2616))

def build_cifar10_preprocess():
    return transforms.Compose([transforms.ToTensor(), CIFAR10_NORM])

def load_torch_vision_model(weights_path: str, num_classes=10):
    state = torch.load(weights_path, map_location="cpu")
    from torchvision.models import resnet18
    m = resnet18(weights=None)
    if m.fc.out_features != num_classes:
        import torch.nn as nn
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    m.load_state_dict(state, strict=False)
    m.eval()
    return m
