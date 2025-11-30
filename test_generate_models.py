"""
test_generate_models.py

Generate various test models for uploading to the Quanteval UI.
This creates models of different architectures to test the generic loader.

Usage:
    python test_generate_models.py
    
This will create a 'test_models/' directory with various .pt files ready for upload.
"""

import torch
import torchvision.models as models
import os

# Create output directory
os.makedirs("test_models", exist_ok=True)

print("=" * 60)
print("GENERATING TEST MODELS FOR QUANTEVAL")
print("=" * 60)

# ==========================================================
# 1. RESNET18 - Popular CNN
# ==========================================================
print("\n[1/5] Downloading ResNet18 (ImageNet)...")
try:
    resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    resnet18.eval()
    torch.save(resnet18, "test_models/resnet18_imagenet.pt")
    size_mb = os.path.getsize("test_models/resnet18_imagenet.pt") / 1024 / 1024
    params = sum(p.numel() for p in resnet18.parameters())
    print(f"  ✓ Saved resnet18_imagenet.pt ({size_mb:.1f} MB)")
    print(f"    Parameters: {params:,}, Classes: 1000")
except Exception as e:
    print(f"  ✗ Failed: {e}")

# ==========================================================
# 2. MOBILENET V2 - Efficient mobile model
# ==========================================================
print("\n[2/5] Downloading MobileNetV2...")
try:
    mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    mobilenet.eval()
    torch.save(mobilenet, "test_models/mobilenet_v2.pt")
    size_mb = os.path.getsize("test_models/mobilenet_v2.pt") / 1024 / 1024
    params = sum(p.numel() for p in mobilenet.parameters())
    print(f"  ✓ Saved mobilenet_v2.pt ({size_mb:.1f} MB)")
    print(f"    Parameters: {params:,}, Classes: 1000")
except Exception as e:
    print(f"  ✗ Failed: {e}")

# ==========================================================
# 3. VGG16 - Classic deep CNN
# ==========================================================
print("\n[3/5] Downloading VGG16...")
try:
    vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    vgg16.eval()
    torch.save(vgg16, "test_models/vgg16.pt")
    size_mb = os.path.getsize("test_models/vgg16.pt") / 1024 / 1024
    params = sum(p.numel() for p in vgg16.parameters())
    print(f"  ✓ Saved vgg16.pt ({size_mb:.1f} MB)")
    print(f"    Parameters: {params:,}, Classes: 1000")
except Exception as e:
    print(f"  ✗ Failed: {e}")

# ==========================================================
# 4. SQUEEZENET - Very small model
# ==========================================================
print("\n[4/5] Downloading SqueezeNet...")
try:
    squeezenet = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.IMAGENET1K_V1)
    squeezenet.eval()
    torch.save(squeezenet, "test_models/squeezenet.pt")
    size_mb = os.path.getsize("test_models/squeezenet.pt") / 1024 / 1024
    params = sum(p.numel() for p in squeezenet.parameters())
    print(f"  ✓ Saved squeezenet.pt ({size_mb:.1f} MB)")
    print(f"    Parameters: {params:,}, Classes: 1000")
except Exception as e:
    print(f"  ✗ Failed: {e}")

# ==========================================================
# 5. EFFICIENTNET - Modern efficient architecture
# ==========================================================
print("\n[5/5] Downloading EfficientNet B0...")
try:
    efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    efficientnet.eval()
    torch.save(efficientnet, "test_models/efficientnet_b0.pt")
    size_mb = os.path.getsize("test_models/efficientnet_b0.pt") / 1024 / 1024
    params = sum(p.numel() for p in efficientnet.parameters())
    print(f"  ✓ Saved efficientnet_b0.pt ({size_mb:.1f} MB)")
    print(f"    Parameters: {params:,}, Classes: 1000")
except Exception as e:
    print(f"  ✗ Failed: {e}")

# ==========================================================
# BONUS: Modify ResNet18 for CIFAR-10 (10 classes)
# ==========================================================
print("\n[BONUS] Creating ResNet18 for CIFAR-10 (10 classes)...")
try:
    import torch.nn as nn
    resnet18_cifar = models.resnet18(weights=None)  # No pretrained weights
    # Modify for CIFAR-10
    resnet18_cifar.fc = nn.Linear(resnet18_cifar.fc.in_features, 10)
    resnet18_cifar.eval()
    torch.save(resnet18_cifar, "test_models/resnet18_cifar10.pt")
    size_mb = os.path.getsize("test_models/resnet18_cifar10.pt") / 1024 / 1024
    params = sum(p.numel() for p in resnet18_cifar.parameters())
    print(f"  ✓ Saved resnet18_cifar10.pt ({size_mb:.1f} MB)")
    print(f"    Parameters: {params:,}, Classes: 10 (CIFAR-10 compatible!)")
except Exception as e:
    print(f"  ✗ Failed: {e}")

# ==========================================================
# SUMMARY
# ==========================================================
print("\n" + "=" * 60)
print("GENERATION COMPLETE!")
print("=" * 60)

files = os.listdir("test_models")
if files:
    print(f"\nGenerated {len(files)} test models in 'test_models/' directory:")
    print()
    for f in sorted(files):
        size = os.path.getsize(f"test_models/{f}")
        if size < 1024 * 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size / 1024 / 1024:.1f} MB"
        print(f"  • {f:<35} {size_str:>10}")
    
    print("\n" + "-" * 60)
    print("TESTING INSTRUCTIONS:")
    print("-" * 60)
    print("1. Run Streamlit UI: streamlit run ui/app.py")
    print("2. Select 'Upload Custom Model'")
    print("3. Upload any .pt file from test_models/")
    print("4. Click 'Run Evaluation'")
    print()
    print("EXPECTED RESULTS:")
    print("  • tiny_cnn_10classes.pt       → Detects as CNN")
    print("  • tiny_transformer_2classes.pt → Detects as Transformer")
    print("  • tiny_mlp_10classes.pt       → Detects as MLP")
    print("  • resnet18_imagenet.pt        → Detects as CNN")
    print("  • mobilenet_v2_imagenet.pt    → Detects as CNN/Hybrid")
    print("  • distilbert_base.pt          → Detects as Transformer")
    print()
    print("NOTE: Models with 1000 classes (ImageNet) will fail CIFAR-10")
    print("      evaluation but should show profile and latency correctly.")
    print("=" * 60)
else:
    print("\nNo models were generated. Check error messages above.")

# ==========================================================
# BONUS: Create a README
# ==========================================================
readme_content = """# Test Models for Quanteval

This directory contains test models for the Quanteval UI.

## Models Included:

### Tiny Custom Models (Fast inference, small file size)
- **tiny_cnn_10classes.pt**: Small CNN with 10 output classes (CIFAR-10 compatible)
- **tiny_transformer_2classes.pt**: Small Transformer with 2 output classes
- **tiny_mlp_10classes.pt**: Simple MLP with 10 output classes

### Pretrained Models (Slower inference, larger file size, better accuracy)
- **resnet18_imagenet.pt**: ResNet18 pretrained on ImageNet (1000 classes)
- **mobilenet_v2_imagenet.pt**: MobileNetV2 pretrained on ImageNet (1000 classes)
- **distilbert_base.pt**: DistilBERT base model (if available)

## Usage:

1. Start the Quanteval UI: `streamlit run ui/app.py`
2. Select "Upload Custom Model"
3. Choose any .pt file from this directory
4. Run evaluation

## Expected Behavior:

| Model | Detected Type | CIFAR-10 Eval | Generic Eval |
|-------|---------------|---------------|--------------|
| tiny_cnn_10classes.pt | CNN | ✓ Works | ✓ Works |
| tiny_transformer_2classes.pt | Transformer | ✗ Fails | ✓ Works |
| tiny_mlp_10classes.pt | MLP | ✗ Fails | ✓ Works |
| resnet18_imagenet.pt | CNN | ✗ Wrong classes | ✓ Works |
| mobilenet_v2_imagenet.pt | CNN/Hybrid | ✗ Wrong classes | ✓ Works |
| distilbert_base.pt | Transformer | ✗ Fails | ✓ Works |

✓ = Should work correctly
✗ = Will fall back to generic evaluation (latency only)

## Notes:

- Models with 1000 output classes (ImageNet models) won't work with CIFAR-10 evaluation
- The UI should automatically fall back to generic profiling in these cases
- All models should show correct parameter counts, size estimates, and latency
"""

with open("test_models/README.md", "w") as f:
    f.write(readme_content)

print("\n✓ Created test_models/README.md with usage instructions")