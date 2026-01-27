# ms2-peptide-fragmentation-using-GNN

## Introduction

`ms2-peptide-fragmentation-using-GNN` is a project that leverages Graph Neural Networks (GNNs) to predict peptide fragmentation patterns in mass spectrometry (MS/MS) experiments. The repository provides tools and code for data preprocessing, model training, evaluation, and prediction of peptide spectra, facilitating research in proteomics and computational biology.

## Features

- Predicts peptide fragmentation using advanced GNN architectures.
- End-to-end pipeline: data preprocessing, feature extraction, model training, and inference.
- Support for custom datasets and peptide modifications.
- Utilities for spectrum processing and visualization.
- Highly modular codebase for easy customization and experimentation.

## Requirements

To run this project, ensure you have the following installed:

- Python 3.7+
- PyTorch (with CUDA support recommended)
- PyTorch Geometric
- NumPy
- tqdm
- pandas
- scikit-learn
- matplotlib
- Other packages as listed in `requirements.txt`

GPU acceleration is recommended for model training but not strictly required.

## Installation

Follow these steps to install and set up the project:

1. Clone the repository:
   ```bash
   git clone https://github.com/Kvj-Harsha/ms2-peptide-fragmentation-using-GNN.git
   cd ms2-peptide-fragmentation-using-GNN
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Download and prepare datasets as described in the Usage section.

## Usage

The repository provides scripts and modules for data preprocessing, training, and prediction. Below are the typical workflows:

### Data Preparation

Prepare your peptide and spectral data in the required format (CSV, mzML, or similar). Use the data preprocessing scripts to convert raw data into model-ready formats.

### Training the Model

Train a GNN model to predict peptide fragmentation:

```bash
python train.py --config configs/exp_default.yaml
```

- Customize configuration files in the `configs` directory to adjust hyperparameters, dataset paths, and model architecture.

### Predicting Fragmentation

Run inference on new peptide sequences:

```bash
python predict.py --input data/test_peptides.csv --model checkpoints/best_model.pth
```

### Evaluating Model Performance

Evaluate the trained model on a validation or test set:

```bash
python evaluate.py --model checkpoints/best_model.pth --data data/validation_set.csv
```

### Visualizing Results

Use provided utilities to visualize predicted vs. actual spectra:

```bash
python visualize.py --prediction results/predicted_spectra.csv --groundtruth results/actual_spectra.csv
```

## Configuration

Configuration is managed via YAML files in the `configs` directory. Key options include:

- **model:** Architecture, number of layers, and hidden units.
- **training:** Batch size, learning rate, epochs, optimizer settings.
- **data:** Paths to training, validation, and test datasets.
- **misc:** Logging, checkpointing, and device (CPU/GPU) selection.

Example configuration snippet:

```yaml
model:
  type: GNN
  layers: 4
  hidden_dim: 128

training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.001

data:
  train_path: data/train.csv
  val_path: data/val.csv
  test_path: data/test.csv
```

## Contributing

We welcome contributions! To contribute:

- Fork the repository and create a new branch.
- Make your changes and commit them with clear, concise messages.
- Ensure your code passes existing tests and add new tests as needed.
- Submit a pull request describing your changes.

Before contributing, please review the existing code, check for open issues, and discuss major changes in advance.

## License

This project is licensed under the MIT License.

---

```
MIT License

Copyright (c) 2026 Kvj-Harsha

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
