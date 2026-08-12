"""Training regression models"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNetCV
from src.split import random_combination_splits


def run_nested_pls_cv(
        df : pd.DataFrame, 
        splits: list[tuple[list, list]], 
        synergy: bool
):
    """
    Running nested CV for PLS regression
    """
    scores = []

    if synergy:
        target = "synergy_score"
    else:
        target = "CFU"

    for train_idx, test_idx in splits:
        train_df = df.iloc[train_idx]
        X_train = train_df.iloc[:, train_df.columns.str.contains("SP")]
        y_train = train_df[target]

        test_df = df.iloc[test_idx]
        X_test = test_df.iloc[:, test_df.columns.str.contains("SP")]
        y_test = test_df[target]

        # Create a nested hyperparameter tuning scheme
        param_grid = {
            "model__n_components": list(range(3, 20))
        }
        
        # Make pipeline for PLS regression
        pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("model", PLSRegression())
        ])

        # Setup GridSearch
        search = GridSearchCV(
            estimator = pipeline,
            cv = 5,
            param_grid = param_grid,
            scoring = "neg_mean_squared_error",
        )

        # Fit with best params
        search.fit(X_train, y_train)
        preds = search.predict(X_test)

        # Evaluate
        score = r2_score(y_test, preds)
        scores.append(score)
    
    mean_score = np.mean(scores)

    # Round
    scores = [round(score, 3) for score in scores]
    mean_score = round(mean_score, 3)

    return scores, mean_score


def run_nested_elasticnet_cv(
    df: pd.DataFrame, 
    splits: list[tuple[list, list]], 
    synergy: bool
):
    
    if synergy:
        target = "synergy_score"
    else:
        target = "CFU"

    scores = []

    for train_idx, test_idx in splits:

        # Separate train and test data
        train_df = df.iloc[train_idx]
        X_train = train_df.iloc[:, train_df.columns.str.contains("SP")]
        y_train = train_df[target]

        test_df = df.iloc[test_idx]
        X_test = test_df.iloc[:, test_df.columns.str.contains("SP")]
        y_test = test_df[target]

        # Pipeline for ElasticNet
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNetCV(cv = 5))
        ])

        # Fit and predict
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        # Evaluate
        score = r2_score(y_test, preds)
        scores.append(score)
    
    mean_score = np.mean(scores)

    # Round
    scores = [round(score, 3) for score in scores]
    mean_score = round(mean_score, 3)

    return scores, mean_score


def train_custom_cfu_model(
        df: pd.DataFrame, 
        train_mask: np.array, 
        test_mask: np.array, 
        title: str,
):  
    """
    Training CFU model with custom train and test mask, then plotting predictions

    Args:
        df         : Dataframe of TPM values, CFUs, and associated drug metadata
        train_mask : Binary mask for training data 
        test_mask  : Binary mask for test data
        title      : Title for final plot
    
    Returns:
    """

    df_new = df.copy()
    df_new = df_new.rename(columns = {
        "drug1_dose": "Drug 1 dose (x MIC)",
        "drug2_dose": "Drug 2 dose (x MIC)",
        "drug_id": "Drug ID"
    })

    train_df = df_new[train_mask]
    test_df = df_new[test_mask]

    # Train-test split
    X_train = train_df.iloc[:, train_df.columns.str.contains("SP")]
    y_train = train_df["CFU"]

    X_test = test_df.iloc[:, test_df.columns.str.contains("SP")]
    y_test = test_df["CFU"]
    meta = test_df.iloc[:, ~test_df.columns.str.contains("SP|CFU")]

    # PLS regression CV pipeline
    param_grid = {
        "model__n_components": list(range(3, 20))
    }

    # Make pipeline for PLS regression
    pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", PLSRegression())
    ])

    # Grid search
    search = GridSearchCV(
        estimator = pipeline,
        cv = 5,
        param_grid = param_grid,
        scoring = "neg_mean_squared_error",
    )

    # Fit with best params
    search.fit(X_train, y_train)
    y_pred = search.predict(X_test)

    # Evaluate
    score = r2_score(y_test, y_pred)

    # Store results with metadata
    results = meta.copy()
    results["true"] = y_test.to_frame()
    results["pred"] = y_pred

    # Plot
    fig, axes = plt.subplots(1, 3, figsize = (20, 5))

    sns.scatterplot(
        results,
        x = "true",
        y = "pred",
        hue = "Drug ID",
        ax = axes[0]
    )

    sns.scatterplot(
        results,
        x = "true",
        y = "pred",
        hue = "Drug 1 dose (x MIC)",
        style = "Drug ID",
        ax = axes[1]
    )

    sns.scatterplot(
        results,
        x = "true",
        y = "pred",
        hue = "Drug 2 dose (x MIC)",
        style = "Drug ID",
        ax = axes[2]
    )

    # Set axis limits and R^2 annotation
    for ax in axes:
        ax.set_xlim(5, 10)
        ax.set_ylim(5, 10)
        ax.text(5.5, 9.5, f"$R^2$ = {round(score, 3)}")
        ax.set_xlabel(f"True log10 CFU")
        ax.set_ylabel(f"Predicted log10 CFU")
        ax.legend(loc = "lower right")
    
    fig.suptitle(title)

    # Extract and return best model 
    best_model = search.best_estimator_

    return best_model


def train_custom_synergy_model(
        df: pd.DataFrame, 
        train_mask: np.array, 
        test_mask: np.array, 
        title: str,
):  
    """
    Training synergy model with custom train and test mask, then plotting predictions

    Args:
        df         : Dataframe with interaction scores, synergy scores, and associated metadata
        train_mask : Binary mask for training data
        test_mask  : Binary mask for test data
        title      : Final plot title
    
    Returns:
    """

    df_new = df.copy()
    df_new = df_new.rename(columns = {
        "drug1_dose": "Drug 1 dose (x MIC)",
        "drug2_dose": "Drug 2 dose (x MIC)",
        "drug_id": "Drug ID"
    }) 

    train_df = df_new[train_mask]
    test_df = df_new[test_mask]

    # Train-test split
    X_train = train_df.iloc[:, train_df.columns.str.contains("SP")]
    y_train = train_df["synergy_score"]

    X_test = test_df.iloc[:, test_df.columns.str.contains("SP")]
    y_test = test_df["synergy_score"]
    meta = test_df.iloc[:, ~test_df.columns.str.contains("SP|synergy_score")]

    # PLS regression CV pipeline
    param_grid = {
        "model__n_components": list(range(3, 20))
    }

    # Make pipeline for PLS regression
    pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", PLSRegression())
    ])

    # Grid search
    search = GridSearchCV(
        estimator = pipeline,
        cv = 5,
        param_grid = param_grid,
        scoring = "neg_mean_squared_error",
    )

    # Fit with best params
    search.fit(X_train, y_train)
    y_pred = search.predict(X_test)

    # Evaluate
    score = r2_score(y_test, y_pred)

    # Store results with metadata
    results = meta.copy()
    results["true"] = y_test.to_frame()
    results["pred"] = y_pred

    # Plot
    fig, axes = plt.subplots(1, 2, figsize = (13, 5))

    sns.scatterplot(
        results,
        x = "true",
        y = "pred",
        hue = "Drug 1 dose (x MIC)",
        ax = axes[0]
    )

    axes[0].legend(
        title = "Drug 1 dose (x MIC)",
        loc = "lower right"
    )

    sns.scatterplot(
        results,
        x = "true",
        y = "pred",
        hue = "Drug 2 dose (x MIC)",
        ax = axes[1]
    )

    axes[1].legend(
        title = "Drug 2 dose (x MIC)",
        loc = "lower right"
    )

    # Get data range and define padding for axes
    data_min = np.min([results["true"].min(), results["pred"].min()])
    data_max = np.max([results["true"].max(), results["pred"].max()])
    padding = 0.1 * (data_max - data_min)

    # Set axis limits and R^2 annotation
    for ax in axes:
        ax.set_xlim(data_min - padding, data_max + padding)
        ax.set_ylim(data_min - padding, data_max + padding)
        ax.text(0.9 * data_min, 0.9 * data_max, f"$R^2$ = {round(score, 3)}")
        ax.set_xlabel(f"True EOB score")
        ax.set_ylabel(f"Predicted EOB score")
    
    fig.suptitle(title)

    # Extract and return best model 
    best_model = search.best_estimator_

    return best_model


def plot_r2_over_data_increase(
    df: pd.DataFrame,
    step_size: int,
    n_splits: int,
    seed: int | None = None
):
    """
    Plot R^2 performance over time as more combination data is incrementaly added to the training set.

    Args:
        df        : Dataframe with attached metadata
        step_size : Number of combination datapoints to add from each drug pair at each increment
        n_splits  : Number of train-test splits to generate per increment
        seed      : Random seed for numpy

    Returns:
        Plot where x = # combination datapoints in training, y = R^2
    """
    # Filter out metadata
    X = df.iloc[:, df.columns.str.contains("SP")]
    y = df["CFU"]

    # Number of combination datapoints
    n_combo_data = (df["num_drugs"] == 2).sum()
    scores = []

    # Loop through # of 
    for i in range(0, n_combo_data, step_size):

        # Get train-test splits
        splits = random_combination_splits(
            data_df = df,
            n_combo_datapoints = i,
            n_splits = n_splits,
            seed = seed
        )
        split_scores = []

        # Train and evaluate a model for each split
        for train_idx, test_idx in splits:

            # Train-test split
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # PLS regression CV pipeline
            param_grid = {
                "model__n_components": list(range(3, 15))
            }

            # Make pipeline for PLS regression
            pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", PLSRegression())
            ])

            # Grid search
            search = GridSearchCV(
                estimator = pipeline,
                cv = 5,
                param_grid = param_grid,
                scoring = "neg_mean_squared_error",
            )

            # Fit with best params
            search.fit(X_train, y_train)
            y_pred = search.predict(X_test)

            # Evaluate
            score = r2_score(y_test, y_pred)
            split_scores.append(score)

        # Add to overall score tracker
        scores.append(split_scores)
    
    scores = np.array(scores)

    # Convert to dataframe
    results = pd.DataFrame(scores) 
    results["n_combo"] = list(range(0, n_combo_data, step_size))
    results = results.melt(id_vars = ["n_combo"], value_name = "R^2").drop(columns = ["variable"])

    # Plot
    ax = sns.stripplot(results, x = "n_combo", y = "R^2")
    sns.pointplot(
        results, 
        x = "n_combo", 
        y = "$R^2$", 
        ax = ax, 
        errorbar = "sd", 
        color = "red",     
        linewidth = 1,
        markersize = 4,
        err_kws = {"linewidth": 1})

    # Customize
    ax.set_title("$R^2$ as more combination data is added to the training set")
    ax.set_xlabel("Number of combination datapoints added to training set")
    ax.set_ylabel("$R^2$")

    plt.show()