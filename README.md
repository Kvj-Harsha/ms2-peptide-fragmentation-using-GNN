# Peptide Fragment Ion Probability Prediction Using Graph Neural Networks

> Find the complete project report [here](https://drive.google.com/file/d/1_ysWikHUgiRHcbV-RSbOgolb0iYWr8IS/view?usp=sharing)

## Abstract

Mass spectrometry (MS) is a foundational technology in modern proteomics, enabling the identification and characterization of peptides through tandem mass spectrometry (MS/MS). Accurate prediction of peptide fragment ion probabilities is critical for improving peptide–spectrum matching, scoring functions, and protein identification pipelines.

This project presents a Graph Neural Network (GNN)–based framework for predicting b-ion and y-ion fragment probabilities directly from peptide sequences. Peptides are represented as graphs, where amino acids form nodes and peptide bonds define edges, allowing the model to capture both local and global structural dependencies.

The model is trained on the large-scale Pep2Prob benchmark dataset and produces probability distributions across 78 fragment ion channels. An interactive Streamlit-based visualization platform is provided to inspect predicted fragmentation patterns in real time.

<img width="806" height="443" alt="image" src="https://github.com/user-attachments/assets/307ab053-a470-4a58-ba77-366555e4b70d" />

## 1. Introduction

In MS/MS experiments, peptide precursor ions are fragmented to generate sequence-specific ions, primarily b-ions (N-terminal fragments) and y-ions (C-terminal fragments). The relative intensities of these ions are central to peptide identification and scoring in database search engines.

Traditional fragmentation prediction methods rely on rule-based heuristics or simplified physical models, which struggle to capture the complex biochemical factors governing peptide fragmentation. Recent advances in deep learning, particularly graph-based models, provide a powerful alternative for learning these nonlinear dependencies from large-scale data.

This project explores the use of Graph Neural Networks to model peptide fragmentation, leveraging the structured nature of peptide sequences and demonstrating strong predictive performance.

<img width="667" height="729" alt="image" src="https://github.com/user-attachments/assets/0b1de429-781a-4258-9edd-f46c8876fd8f" />

## 2. Problem Statement

How can a deep learning model accurately predict peptide fragment ion probabilities directly from amino acid sequences, without requiring raw mass spectrometry spectra, while generalizing across diverse peptide lengths and compositions?

## 3. Objectives

The primary objectives of this project are:

1. To preprocess the Pep2Prob dataset and convert peptide sequences into graph representations.
2. To design and implement a GNN architecture for fragment ion probability prediction.
3. To train and evaluate the model using appropriate loss functions and correlation metrics.
4. To deploy the trained model through a user-friendly Streamlit interface.
5. To visualize predicted fragment probabilities using interpretable plots and tabular outputs.

## 4. Dataset

This project uses the Pep2Prob benchmark dataset, a large-scale curated resource designed for fragment ion probability prediction in proteomics.

### Dataset Characteristics

- 610,117 unique peptide precursors
- Derived from over 183 million high-resolution MS/MS spectra
- Provides fragment probability vectors for up to 235 possible fragment ions
- Leakage-free train–test split to ensure unbiased evaluation

In this implementation, a subset of the dataset is used, focusing on b1–b39 and y1–y39 ions.

## 5. Methodology

### 5.1 Peptide Graph Construction

Each peptide is represented as a graph:

- Nodes: Amino acids
- Edges: Peptide bonds between adjacent residues
- Node Features: 21-dimensional one-hot vectors (20 standard amino acids + unknown token)

This representation preserves sequential and structural relationships between residues.

### 5.2 Model Architecture

The proposed model, referred to as PepFragGNN, consists of:

- Multiple graph convolution (GCN) layers with ReLU activation
- Global mean pooling to obtain a fixed-size graph embedding
- A multi-layer perceptron (MLP) regression head
- Sigmoid activation to produce probabilities in the range [0, 1]

Architecture Overview:

Peptide Sequence  
→ Graph Construction  
→ GNN Backbone (GCN Layers)  
→ Global Mean Pooling  
→ MLP Head  
→ Fragment Ion Probabilities  

### 5.3 Training Procedure

The model is trained using:

- Loss Function: Masked Binary Cross-Entropy (BCE)
- Optimizer: Adam (learning rate = 1 × 10⁻³)
- Batch Size: 32
- Epochs: 5
- Hardware: NVIDIA GPU

Masked loss ensures that fragment positions exceeding peptide length do not contribute to the loss.

Loss Function:

L = sum(BCE(y_pred, y_true) × mask) / sum(mask)

<img width="914" height="826" alt="image" src="https://github.com/user-attachments/assets/4e1abb9b-e2f2-445b-9273-8a858ad9d0ab" />

## 6. Evaluation

The model is evaluated on a held-out validation set using:

- Binary Cross-Entropy Loss
- Pearson Correlation Coefficient

Performance Summary:

- Validation BCE Loss: approximately 0.39
- Mean Pearson Correlation: approximately 0.82

These results indicate strong agreement between predicted and true fragment ion probabilities.

<img width="853" height="842" alt="image" src="https://github.com/user-attachments/assets/3206a9de-97f5-49ff-82a3-8e975ade735c" />

## 7. Visualization and Deployment

An interactive Streamlit application is provided to:

- Accept peptide sequences as input
- Perform real-time fragment probability prediction
- Visualize results using line plots, mirror plots, and tabular views
- Export predictions as CSV files

This enables rapid exploration of peptide fragmentation behavior without requiring MS/MS experiments.

<img width="666" height="261" alt="image" src="https://github.com/user-attachments/assets/0bcbfe95-e5b3-4d62-83ea-32273ba62c9c" />

<img width="562" height="542" alt="image" src="https://github.com/user-attachments/assets/d1710b91-5cb9-40e0-8712-79c92b5f6f7a" />


## 8. Repository Structure

.
├── src/
├── test-programs/
├── train.py
├── eval.py
├── predict.py
├── predict_cli_version.py
├── fragment_gnn.pt
├── requirements.txt
├── plan.md
└── README.md

## 9. Limitations and Future Work

Current limitations include:

- Prediction limited to b1 and y1 ions
- Single charge state modeling
- One-hot amino acid encoding

Future extensions may include:

- Higher charge states and neutral-loss ions
- Richer node features using physicochemical properties
- Instrument-specific fragmentation modeling
- Distributed training and inference

## 10. Conclusion

This project demonstrates that Graph Neural Networks are effective for modeling peptide fragmentation in MS/MS proteomics. By treating peptides as structured graphs, the proposed approach captures meaningful biochemical dependencies and achieves strong predictive performance.

The integration of a complete training, evaluation, and deployment pipeline highlights the practical applicability of graph-based deep learning in computational proteomics.
