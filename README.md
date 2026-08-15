# Bacterial phenotype prediction using bulk RNA-seq data
## Overview
This repo contains the code for a project completed in the van Opijnen lab at Boston Children's Hospital. The primary goal was to model *Streptococcus pneumoniae* phenotype in response to multiple antibiotic combinations using bulk RNA-seq data. 

We predicted two different readouts of bacterial phenotype:
- **Bacterial viability**: measured as CFU (colony-forming units)
- **Drug synergy**: measured using EOB (excess over Bliss) score, which tells us how much more effective an antibiotic combination compared to the sum of its monotherapies.

## Analysis pipeline
1. **Data processing** - ntegrate gene expression, phenotype, and treatment metadata
2. **Prediction** - train PLS regression models with cross-validation
3. **Evaluation** - test generalization across antibiotic treatment conditions
4. **Interpretation** - Analyzing model coefficients and using GSEA to find genes and pathways associated with phenotypic outcomes

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
**Run from data root**
Update `configs/data_loader.yaml` to point to the local data directory, then run the notebooks in order.
> **Note:** Source data has not been publicly released at this time.
