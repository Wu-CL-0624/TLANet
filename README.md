# TLANet

[![DOI](https://zenodo.org/badge/1255271460.svg)](https://doi.org/10.5281/zenodo.20478993)

TLANet is a PyTorch-based ECG arrhythmia classification project using a hybrid TCN-LSTM multi-head attention model. It includes code, train/validation data splits, trained weights, and control experiments on the MIT-BIH database, with a Python 3.8, PyTorch 1.13, CUDA 11.8 environment.

## Overview

TLANet integrates Temporal Convolutional Networks (TCN), Long Short-Term Memory (LSTM), and multi-head attention for ECG arrhythmia classification. The repository is intended to support reproducibility of the six-class MIT-BIH experiment and the five-class control experiment without class A.

## Repository Structure

```text
.
├── README.md
├── environment.yml
├── requirements.txt
├── ecg_train_val_minmax/
│   ├── train.csv                         # six-class training set
│   ├── val.csv                           # six-class validation set
│   ├── tcn-lstm-attention_engpicture.py  # TLANet training/evaluation script
│   └── best_ecg_model_multihead.pth      # trained TLANet weights
└── ecg_train_val_minmax_withoutA/
    ├── train.csv                         # training set for the five-class control task
    ├── val.csv                           # validation set for the five-class control task
    ├── withoutA.py                       # five-class control experiment script
    └── best_ecg_model_multihead.pth      # trained weights for the five-class task
```


## Data

The raw ECG records are from the publicly available **MIT-BIH Arrhythmia Database** on PhysioNet. The CSV files in this repository are derived train/validation splits generated from the MIT-BIH data after beat segmentation and min-max normalization.

Expected CSV format:

| Column | Description |
| --- | --- |
| `signals` | Comma-separated ECG signal values for one heartbeat sample |
| `type` | Class label, e.g., `N`, `V`, `/`, `A`, `L`, or `R` |

The six-class experiment uses labels `N`, `V`, `/`, `A`, `L`, and `R`. The five-class control experiment excludes class `A`.

## Note

This repository uses Git LFS to store large dataset files.
Please install Git LFS before cloning the repository:

git lfs install

git clone https://github.com/Wu-CL-0624/TLANet.git

## Environment

The experiments were run in the following environment:

- Python 3.8
- PyTorch 1.13
- CUDA 11.8-capable GPU environment
- NumPy
- pandas
- scikit-learn
- matplotlib
- tqdm

Create the environment with:

```bash
conda env create -f environment.yml
conda activate tlanet-py38-torch113
```

Alternatively, install Python packages with:

```bash
pip install -r requirements.txt
```

Note: PyTorch 1.13 wheels include their own CUDA runtime. The experiments were executed on a CUDA 11.8-capable system. If your platform requires a site-specific PyTorch/CUDA build, install PyTorch 1.13 according to your local GPU driver and cluster configuration.

## Running the Experiments

### Six-class TLANet experiment

```bash
cd ecg_train_val_minmax
python tcn-lstm-attention_engpicture.py
```

Expected inputs:

- `train.csv`
- `val.csv`

Expected outputs include:

- `best_ecg_model_multihead.pth`
- `tcn_lstm_multihead_training_history.svg`
- `tcn_lstm_multihead_confusion_matrix.svg`
- `tcn_lstm_multihead_learning_rate.svg`
- attention visualization SVG files

### Five-class control experiment without class A

```bash
cd ecg_train_val_minmax_withoutA
python withoutA.py
```

Expected inputs:

- `train.csv`
- `val.csv`

Expected outputs include trained weights, training curves, confusion matrix, learning-rate plot, and attention visualizations for the five-class task.

## Reproducibility Notes

- Use the same train/validation split files to reproduce the reported results.
- Keep random seeds, batch size, epoch number, optimizer settings, and early-stopping settings unchanged unless performing an ablation study.
- The scripts save the best model as `best_ecg_model_multihead.pth` based on validation performance.
- The current scripts assume the CSV files contain `signals` and `type` columns.
- For public release, avoid absolute local paths and use repository-relative paths.

## Suggested Data Availability Statement

The raw ECG data analyzed in this study are publicly available from the MIT-BIH Arrhythmia Database on PhysioNet. The source code, preprocessing scripts, train/validation split files, baseline implementations, trained model weights, and experimental result files supporting the findings of this study are available at `https://github.com/Wu-CL-0624/TLANet`. The computational environment uses Python 3.8, PyTorch 1.13, and CUDA 11.8, as specified in this repository.

## Citation

If you use this repository, please cite the associated manuscript after publication and cite the MIT-BIH Arrhythmia Database according to PhysioNet's citation instructions.

## Contact

For questions about the code or data splits, please contact the corresponding author listed in the manuscript.
