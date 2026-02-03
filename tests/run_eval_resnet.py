import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from ui.model_loader import load_model
from eval.metrics import top1
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

device = torch.device('cpu')
keys = ['resnet18_baseline','resnet18_ptq','resnet18_qat']
for k in keys:
    try:
        print('\n=== Evaluate', k)
        m = load_model(k, device)
        # Prepare CIFAR-10 loader
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470,0.2435,0.2616))
        ])
        testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
        subset = torch.utils.data.Subset(testset, list(range(min(len(testset), 200))))
        loader = DataLoader(subset, batch_size=128, shuffle=False)

        acc = top1(m, loader, device=device)
        print('Accuracy:', acc)
    except Exception as e:
        print('Error evaluating', k, e)
        import traceback; traceback.print_exc()
