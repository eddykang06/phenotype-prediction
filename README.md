# Bacterial phenotype prediction using bulk RNA-seq data
## Overview

## Model description


## Repository Structure

```text
phenotype-prediction/
├── configs/
│   └── data_loader.yaml                 # Data path config
│
├── models/
│   └── diagonal_cfu_model.pkl           # Saved CFU prediction model
│
├── notebooks/
│   ├── 01-eda.ipynb                     # Exploratory data analysis
│   ├── 02-cfu-prediction.ipynb          # CFU prediction
│   ├── 03-cfu-interpretation.ipynb      # CFU model interpretation
│   ├── 04-cfu-evaluation.ipynb          # CFU model evaluation
│   ├── 05-synergy-prediction.ipynb      # Synergy prediction
│   └── 06-synergy-interpretation.ipynb  # Synergy model interpretation
│
├── src/
│   ├── dge_data.py                      # Loading DGE data
│   ├── eda.py                           # Exploratory data analysis on log2FC
│   ├── eval.py                          # Model evaluation
│   ├── eval_data.py                     # Loading external dataset for evaluation
│   ├── interpret.py                     # Feature/model interpretation
│   ├── metadata.py                      # Sample metadata
│   ├── split.py                         # Train/test splits
│   ├── tpm_data.py                      # Loading TPM data from feature counts
│   └── train.py                         # Training scripts
```

## Data

## Requirements and setup
**1. Clone the repository**
```bash 
git clone https://github.com/eddykang06/phenotype-prediction.git
cd phenotype-prediction
``` 
**2. Create and activate the environment**
```bash 
conda create -n phenotype-prediction python -y
conda activate phenotype-prediction
```
**3. Install dependencies**
```bash
pip install -r requirements.txt
```
