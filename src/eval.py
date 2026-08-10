"""CFU model evaluation on old entropy data"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from pathlib import Path
from src.metadata import (
    condition_to_drug_id, condition_to_timepoint
)
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator


def plot_growth_curves(
    df: pd.DataFrame
):
    """
    Plot growth curves for each drug in the entropy dataset

    Args:
        df : OD600 values with sample names on index

    Returns:
        Growth curve plot for each drug in the dataset
    """
    index = df.index
    meta = pd.DataFrame(
        {
        "drug_id": [condition_to_drug_id(x) for x in index],
        "timepoint": [condition_to_timepoint(x) for x in index],
        "drug1_dose": [1]*len(index),
        "replicate": [x[-1] for x in index]
        }, 
        index = index
    )
    growth = pd.merge(df, meta, left_index = True, right_index = True, how = "inner")

    fig, ax = plt.subplots(
        figsize = (15, 15), 
        nrows = 4, 
        ncols = 4,
        sharey = True
    )
    ax = ax.ravel()

    for i, drug in enumerate(growth["drug_id"].unique()):
        df = growth[growth["drug_id"] == drug]
        sns.lineplot(
            data = df,
            x = "timepoint",
            y = "OD600",
            estimator = None,
            units = "replicate",
            ax = ax[i]
        )
        ax[i].set_title(drug)
        ax[i].set_xlabel("Time (min)")
    plt.tight_layout()


def plot_1x_predictions(
    model: BaseEstimator,
    train_df: pd.DataFrame
):
    """
    1x MIC training predictions

    Args:  
        model    : Trained model loaded from saved checkedpoint
        train_df : Dataframe containing all data from current RNA-seq experiments
    
    Returns:
        Plot of true vs. predicted log10CFU for all 1x MIC samples in training data

    """
    train_mask = (train_df["num_drugs"] == 1) | ((train_df["drug1_dose"]) == (train_df["drug2_dose"]))
    train = train_df[train_mask]
    train = train[train["drug1_dose"] == 1]

    X_old = train.iloc[:, train.columns.str.contains("SP")]
    y_old = train["CFU"]

    meta_old = train.iloc[:, ~train.columns.str.contains("SP")].drop(columns = ["CFU"])

    preds_old = model.predict(X_old)

    res_old = pd.DataFrame({
        "True log10 CFU": y_old,
        "Predicted log10 CFU": preds_old
    })
    res_old = pd.merge(res_old, meta_old, left_index = True, right_index = True, how = "left")
    sns.scatterplot(
        data = res_old,
        x = "True log10 CFU",
        y = "Predicted log10 CFU",
        hue = "drug_id"
    )
    r2 = r2_score(res_old["True log10 CFU"], res_old["Predicted log10 CFU"])
    plt.title(f"Model predictions on 1x MIC training data ($R^2$ = {r2:.3f})")


def plot_predictions(
    model: BaseEstimator,
    data_df: pd.DataFrame,
    use_entropy: bool = False
):
    """
    Plotting model predictions on entropy dataset

    Args:
        model       : Trained PLS regression model
        data_df     : DataFrame of entropy dataset
        use_entropy : Whether to use entropy or OD600 as comparison to model predictions

    Returns:
        Plot of true vs. predicted phenotype for entropy
    """
    X_test = data_df.iloc[:, data_df.columns.str.contains("SP")]
    y_test = data_df.iloc[:, ~data_df.columns.str.contains("SP")]

    preds = model.predict(X_test)

    res = pd.DataFrame({
        "True OD600": y_test["OD600"],
        "True log10 CFU": y_test["CFU"],
        "Entropy": y_test["Entropy"],
        "Predicted log10 CFU": preds,
    })
    meta_cols = ["drug_id", "timepoint", "drug1_dose"]
    meta = data_df[meta_cols]
    res = pd.merge(res, meta, left_index = True, right_index = True, how = "left")

    # Entropy or Od600
    if use_entropy:
        x_label = "Entropy"
    else:
        x_label = "True OD600"

    sns.scatterplot(
        data = res,
        x = x_label,
        y =  "Predicted log10 CFU",
        hue = "drug_id"
    )
    corr = stats.spearmanr(res[x_label], res["Predicted log10 CFU"])
    plt.title(f"Model predictions for entropy paper dataset (Spearman r = {corr.statistic:.3f})")