"""Functions to implement specific train-test splits"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def combination_stratified_split(
    df: pd.DataFrame, 
    num_folds: int, 
    seed: int | None = None):
    """
    Train-test split where train-test split is stratified by combination, so the 
    drug distribution is equal between train-test

    Args:
        df : Dataframe with metadata attached
    
    Returns:
        splits : List of tuples with (train_idx, test_idx)
    """
    # Store drug IDs to stratify by 
    ids = df["drug_id"].to_numpy()

    # Initialize splitter
    cv = StratifiedKFold(
        n_splits = num_folds,
        shuffle = True,
        random_state = seed
    )

    # Stratified split
    splits = list(cv.split(df, ids))

    return splits


def combination_held_out_split(
    df: pd.DataFrame
):
    """
    Train-test split where 1 drug combination is held out in each split

    Args:
        df : Dataframe with metadata attached
    
    Returns:
        splits : List of tuples with (train_idx, test_idx)
    """

    # Store drug IDs and select unique combinations
    ids = df["drug_id"].to_numpy()
    unique_combos = np.unique(ids)

    # Store splits
    splits = []

    for combo in unique_combos:
        
        # Get idx
        train_idx = np.array([i for i in range(len(ids)) if ids[i] != combo])
        test_idx = np.array([i for i in range(len(ids)) if ids[i] == combo])
        idx_tuple = (train_idx, test_idx)
        splits.append(idx_tuple)
    
    return splits


def timepoint_held_out_split(
    df: pd.DataFrame
):
    """
    Train-test split where 1 timepoint is held out in each split

    Args:
        df : Dataframe with metadata attached
    
    Returns:
        splits : List of tuples with (train_idx, test_idx)
    """

    # Store drug IDs and select unique combinations
    times = df["timepoint"].to_numpy()
    unique_times = np.unique(times)

    # Store splits
    splits = []

    for time in unique_times:
        
        # Get idx
        train_idx = np.array([i for i in range(len(times)) if times[i] != time])
        test_idx = np.array([i for i in range(len(times)) if times[i] == time])
        idx_tuple = (train_idx, test_idx)
        splits.append(idx_tuple)
    
    return splits


def random_combination_splits(
        data_df: pd.DataFrame,
        n_combo_datapoints: int,
        n_splits: int,
        seed: int | None = None
): 
    """
    Generate a specified number of train-test splits with a specified number of combination datapoints in the training set

    Args:
        data_df            : Dataframe with integers on index and attached metadata
        n_combo_datapoints : Number of combination datapoints to include in training set
        n_splits           : Number of different train-test splits to generate
        seed               : Random seed for numpy

    Returns:
        splits : List of tuples of train idx and test idx
    """
    # Reset index
    df = data_df.copy().reset_index()
    splits = []
    rng = np.random.default_rng(seed)

    # Get combination idx and single-drug idx
    combo_idx = df.index[df["num_drugs"] == 2].to_numpy()
    single_idx = df.index[df["num_drugs"] == 1].to_numpy()

    # Store combo dataframe for reference
    combo_df = df.iloc[combo_idx]
    n_combos = len(combo_df["drug_id"].unique())

    for i in range(n_splits):
        
        # For each drug pair, get n random combination datapoints
        random_idx = []

        for drug in combo_df["drug_id"].unique():

            # Subset to drug
            drug_index = combo_df.index[combo_df["drug_id"] == drug].to_numpy()
            drug_random_idx = rng.choice(drug_index, size = n_combo_datapoints // n_combos, replace = False)
            random_idx.append(drug_random_idx)

        # Array and flatten
        random_idx = np.ndarray.flatten(np.array(random_idx))

        # Train and test idx
        train_idx = np.concatenate((single_idx, random_idx))
        test_idx = np.array(list(set(df.index.to_numpy()) - set(train_idx)))
        idx_tuple = (train_idx, test_idx)
        splits.append(idx_tuple)

    return splits