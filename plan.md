[chatgpt-chat-here](https://chatgpt.com/share/691edf09-b128-800e-815c-826fc31de278)

# Graph Neural Networks for Peptide Fragment Ion Probability Prediction

## 1. Introduction

Fragmentation prediction is a key task in computational proteomics.  
Given a **peptide sequence + charge**, the goal is to predict the probability that each fragment ion (**b, y, a ions**, and possible charge states **1+, 2+, 3+**) appears in MS/MS.

Most tools require raw spectral data (RAW, mzML). However, **this project intentionally uses the Pep2Prob tabular dataset**, containing:
- peptide sequence  
- precursor charge  
- probability columns such as `('b',1,5)`  

This is perfect for machine-learning pipelines.

---

## 2. Goal of the Project

Build a **Graph Neural Network (GNN)** that predicts **78 fragmentation probabilities** per peptide.

### Model Input
- Peptide graph  
- Nodes = amino acids  
- Edges = peptide bonds  

### Model Output  
- 78-dim vector (b/y ions, 1+ and 2+)

---

## 3. Dataset Acquisition (Complete)

Dataset: https://huggingface.co/datasets/bandeiralab/Pep2Prob

Loaded via:
```python
from datasets import load_dataset
```

Dataset Stats:
- Rows: **3,148,198**
- Format: Apache Arrow
- Stored at: `~/.cache/huggingface/datasets/`

Status: ✔️ Loaded successfully

---

## 4. Environment Setup (Complete)

Environment created using:
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

GPU verification output:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 4050 Laptop GPU
CUDA: 12.1
Memory: ~6 GB
```

Status: ✔️ GPU enabled

---

## 5. Implementation Completed

### ✔ Graph builder implemented  
### ✔ Target extractor implemented  
### ✔ Custom PyG Dataset implemented  
### ✔ PepFragGNN implemented  
### ✔ Training loop fixed & optimized  
### ✔ Model trained on 50,000 peptides  
### ✔ Evaluation implemented  
### ✔ Prediction script implemented  

Training Loss:
```
Epoch 1: 0.4510
Epoch 2: 0.4133
Epoch 3: 0.4066
```

Evaluation:
```
Test Loss: 0.3928
Pearson Correlation: 0.8202
```

---

## 6. Next Steps

### A: Visualization (Recommended)
- Plot b-ion and y-ion predicted curves  
- Plot correlation scatter  

### B: Improve Model
- Use deeper GNN  
- Add positional encodings  

### C: Build Web Demo
- FastAPI + React  

### D: Write Final Report

---

## 7. Summary Table

| Component | Status |
|----------|--------|
| Dataset | ✔️ Done |
| Graph builder | ✔️ Done |
| Dataset class | ✔️ Done |
| GNN Model | ✔️ Done |
| Training | ✔️ Done |
| Evaluation | ✔️ Done |
| Prediction | ✔️ Done |
| Visualizations | ❌ Pending |
| Web App | ❌ Pending |
| Final Report | ❌ Pending |

---

## Project Status  
### **MODEL WORKING — READY FOR ANALYSIS & VISUALIZATION**