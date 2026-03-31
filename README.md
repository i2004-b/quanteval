# Quanteval

## Overview
   **QuantEval** is a framework for evaluating trade-offs between full-precision (FP32) and quantized neural networks. It supports both preloaded benchmark models and user-supplied architectures, automatically generating comparable variants when possible.

The framework measures:
- Inference latency (hardware-level timing)
- Accuracy degradation
- Memory reduction  

This enables structured analysis of how quantization impacts speed, efficiency, and model performance.

---

## Key Features  
- Supports vision and NLP models (e.g., ResNet18, DistilBERT)  
- Handles FP32 baselines and quantized variants  
- Automated evaluation pipeline  
- Repeated inference sampling for stable timing  
- Graceful fallback when direct comparisons are not possible  

---

## Project Structure
    data/ # Dataset loading and preprocessing
    models/ # Model definitions
    scripts/ # Training and evaluation scripts
    outputs/
    baselines/ # Saved FP32 checkpoints (gitignored)
    requirements.txt # Dependencies
    .gitignore # Ignored files
    
---
## Methodology
QuantEval follows a structured evaluation pipeline:
- Load or generate an FP32 baseline model
- Apply quantization techniques (e.g., PTQ)
- Run repeated inference passes on hardware
- Measure:
    - Accuracy difference
    - Latency improvements
    - Memory savings
- Output results for comparison
---

## Usage 
### Train Baseline Models
python scripts/train_cifar10.py
python scripts/train_sst2.py
### Evaluate Models
Evaluation includes:
- Accuracy measurement
- Inference time benchmarking
- Quantization comparison
(Additional evaluation scripts may be added as the framework expands.)
---

## Git Workflow 
### Push Changes
    git add .
    git commit -m "your message"
    git push
### Pull Changes
    git pull origin main
### Merge 
If a merge conflict occurs:
- Open the conflicted file
- Manually resolve differences
- Run:
    git add <file>
    git commit
    git push
---
## Current Status
- Baseline models implemented (CIFAR-10, SST-2)
- Training and evaluation scripts functional
- Initial quantization evaluation pipeline in progress
---
## Notes
- Outputs and model checkpoints are excluded from version control
- Designed for CPU-based evaluation (extensible to GPU)
---
## Progress
- Week 1: Baselines (CIFAR-10, SST-2), added data, data/cifar10.py, data/sst2.py, outputs, outputs/baselines, scripts, scripts/test_baselines.py, scripts/train_cifar10.py, scripts/train_sst2.py, requirements.txt, .gitignore

## Running Week 1 Baselines
- CIFAR-10 (ResNet18):
  ```bash
  python scripts/train_cifar10.py
  python scripts/train_sst2.py

## Setup
```bash
BE IN CMD NOT POWERSHELL
python -m venv .venv
.venv\Scripts\activate.bat  # Windows
pip install -r requirements.txt
