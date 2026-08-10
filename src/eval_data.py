"""Functions for loading data from old entropy paper"""
import pandas as pd
import numpy as np
import os, re
import yaml
import string
from pathlib import Path
from src.tpm_data import (
    fcnts_to_tpms, 
    read_fcnts_as_df, 
    bind_tpm_data,
)
from src.metadata import (
    condition_to_drug_id, condition_to_timepoint
)


def get_od_and_cfu(od_path):
    """
    Get OD600 for all samples, then also get CFUs by converting

    Args:
        od_path: Path to OD600 values for all samples

    Returns:
        df : Dataframe of OD600 and CFU with sample names on index
    
    """
    # Get OD600 and converted CFU (10^8 conversion)
    df = pd.read_csv(od_path, header=[0, 1], index_col = 0)

    df.index.name = "time_min"
    df.columns.names = ["drug", "replicate"]

    df = (
        df.stack(["drug", "replicate"], future_stack = True)
        .rename("OD600")
        .dropna()
        .reset_index()
    )

    df["drug_id"] = (
        df["drug"]
        + df["time_min"].astype(str)
        + "min-"
        + df["replicate"]
    )

    df = df.set_index("drug_id")[["OD600"]]

    # Convert OD to CFU
    df["CFU"] = np.log10(df["OD600"] * 10**8) # Conversion factor approximate

    return df


def get_entropy(entropy_path):
    """
    Get entropy for each sample

    Args:
        entropy_path : Path to entropy files
    
    Returns:
        df : DataFrame of entropy with sample names on index
    """
    df = pd.read_csv(entropy_path)

    # Filter out unneeded info
    df = df.drop(columns = ["Survive", "Group", "MOA", "Prediction"])
    mask = (df["Strain"] == "T4") & (df["Adapted"] == False) & (df["Concentration"] == "L")
    df = df[mask]

    # New column of naming
    df["id"] = df["AB"] + df["Time"].astype(str) + "min"
    df["AB"].unique()

    n = 3
    suffixes = list(string.ascii_lowercase[:n])

    df = (
        df.loc[df.index.repeat(n)]
        .reset_index(drop = True)
    )

    suffix_column = np.tile(suffixes, len(df) // n)

    # Get sample ids
    df["id"] = (
        df["id"].astype(str)
        + "-"
        + suffix_column
    )
    df = df.set_index("id")
    df = df["Entropy"]

    return df


def get_growth_curves(root):
    """
    Load OD600 growth curve data from data config file
    """
    # Get data from configs
    config_path = Path(root / "configs" / "data_loader.yaml")
    
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Get growth curve data
    data_dir = Path(cfg["data_dir"])
    od_path = str(data_dir / "entropy_data" / "od600" / "growth_curves.csv")
    gc = get_od_and_cfu(od_path)
    gc = gc.drop(columns = ["CFU"])

    return gc

    
def get_entropy_data(root):
    """
    Load TPM, entropy, and OD600 values from entropy dataset using data config file

    Args:
        root: Path to root directory of repo

    Returns:
        df   : DataFrame of all TPM, entropy, OD600, and CFU values from entropy dataset
        meta : Associated metadata for all samples
    """
    # Get data from configs
    config_path = Path(root / "configs" / "data_loader.yaml")
    
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Entropy data paths
    data_dir = Path(cfg["data_dir"])
    fcnts_path = str(data_dir / "entropy_data" / "fcnts")
    od_path = str(data_dir / "entropy_data" / "od600" / "growth_curves.csv")
    entropy_path = data_dir / "entropy_data" / "entropy" / "entropy_values.csv"

    # Get each data modality
    tpm = bind_tpm_data(fcnts_to_tpms(
        read_fcnts_as_df(fcnts_path, entropy = True),
        strip_leading_digits = True
    ))
    phenotype = get_od_and_cfu(od_path)
    entropy = get_entropy(entropy_path)

    # Merge
    df = pd.merge(tpm, phenotype, left_index = True, right_index = True, how = "inner")
    df = pd.merge(df, entropy, left_index = True, right_index = True, how = "left")

    # Attach metadata
    df["drug_id"] = [condition_to_drug_id(x) for x in df.index]
    df["timepoint"] = [condition_to_timepoint(x) for x in df.index]
    df["drug1_dose"] = [1]*len(df.index)
    
    return df
