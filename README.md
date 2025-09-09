# Quanteval

## Overview
    Quanteval is a group project to benchmark quantization methods (PTQ, QAT, low-bit adapters, etc.) on vision (CIFAR-10, ResNet18) and NLP (SST-2, DistilBERT) tasks. 

## Project Structure
    data/                # dataset helpers
    scripts/             # training + evaluation scripts
    outputs/baselines/   # saved checkpoints (ignored in git)
    requirements.txt     # dependencies
    .gitignore           # ignored files and folders

## How to push/pull
    To push: 
        git add .    
        git commit -m "message here"    
        git push
    To pull: git pull origin main

## If merge conflict
    If you both edited the same file, Git may show a merge conflict. In that case:
    Open the file → resolve conflict manually.
    Then run:
    git add <file>
    git commit
    git push
    
## Progress
- Week 1: Baselines (CIFAR-10, SST-2), added data, data/cifar10.py, data/sst2.py, outputs, outputs/baselines, scripts, scripts/test_baselines.py, scripts/train_cifar10.py, scripts/train_sst2.py, requirements.txt, .gitignore

## Running Week 1 Baselines
- CIFAR-10 (ResNet18):
  ```bash
  python scripts/train_cifar10.py
  python scripts/train_sst2.py

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
