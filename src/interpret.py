"""Functions for feature intpretation from trained models"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gseapy as gp
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV


def cv_feature_importances(
    df: pd.DataFrame,
    splits: list[tuple[list, list]]
):
    """
    Run nested CV and calculate feature stability across folds (mean/std coefficient across folds)

    Args:
        df     : Dataframe with 
        splits : List containing train and test idx for each CV split
    
    Returns:
        feature_df : 1 column dataframe with feature importances ordered by absolute value
    """
    # Initialize feature importance dataframe
    feature_df = []

    # Nested cross-validation to train models
    for train_idx, _ in splits:
        train_df = df.iloc[train_idx]
        X_train = train_df.iloc[:, train_df.columns.str.contains("SP")]
        y_train = train_df["synergy_score"]

        param_grid = {
                "model__n_components": list(range(3, 20))
        }

        pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("model", PLSRegression())
        ])

        search = GridSearchCV(
                estimator = pipeline,
                cv = 5,
                param_grid = param_grid,
                scoring = "neg_mean_squared_error",
        )

        search.fit(X_train, y_train)
        best_pipeline = search.best_estimator_

        # Add store parameters in feature importances
        coefs = best_pipeline.named_steps["model"].coef_.ravel()
        feature_df.append(coefs)
    
    # Reshape to dataframe
    feature_df = pd.DataFrame(feature_df).T
    feature_df.index = df.columns[df.columns.str.contains("SP")]

    # Calculate importances
    feature_df["coef"] = feature_df.mean(axis = 1) / feature_df.std(axis = 1)
    feature_df = feature_df["coef"].to_frame()

    # Sort by abs(coef)
    feature_df = feature_df.sort_values("coef", key = lambda x: x.abs(), ascending = False)

    return feature_df


def plot_top_features(
    coef_df: pd.DataFrame,
    annot_df: pd.DataFrame,
    top_n: int,
    xlabel: str,
    title: str,
):
    """
    Plot top coefficients from model in descending absolute order

    Args:
        coef_df  : Dataframe with genes on index and 1 column of coefficients or feature importances
        annot_df : Dataframe of annotations with genes on index
        top_n    : Number of features to show
        xlabel   : Label for y axis
        title    : Plot title

    Returns:
        Plot of top model coefficients with gene product annotations
    """
    # Outer join coefficients and anotations
    features = pd.merge(
        annot_df,
        coef_df,
        left_index = True,
        right_index = True,
        how = "outer"
    )

    fig, ax = plt.subplots(figsize = (7,7))

    # Sort by absolute value of coefficient
    features = features.iloc[np.argsort(-abs(features["coef"]))]
    features["coef"].iloc[:top_n].plot(kind = "barh", ax = ax).invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Gene")
    ax.set_title(title)
    ax.axvline(x = 0, color = ".5")

    # Add annotations as extra xis
    secax = ax.secondary_yaxis("left")
    secax.set_yticks(range(len(features.iloc[:top_n])))
    secax.set_yticklabels(features.iloc[:top_n]["Product"])
    secax.spines["left"].set_position(("outward", 130))
    secax.set_ylabel("Product")


def construct_gene_set(
        df: pd.DataFrame, 
        set_col: str
) -> dict[str:list[str]]:
    """
    Construct a custom gene set dictionary using a specified 

    Args:
        df      : Dataframe with gene IDs on index and annotations
        set_col : Name of column to be used as gene set 
    
    Returns:
        gene_sets : Dictionary mapping pathways to list of gene IDs
    """
    # Get unique gene sets and remove NA
    keys = list(df[set_col].unique().dropna())

    # Get corresponding gene index
    gene_sets = {key:list(df.index[df[set_col] == key]) for key in keys}

    return gene_sets


def run_custom_gsea(
    coef_df: pd.DataFrame,
    annot_df: pd.DataFrame,
    set_col: str,
    seed: int
):  
    """
    Run GSEA on genes ranked by coefficient with custom gene sets using annotation df

    Args:
        coef_df  : Dataframe with genes on index and column of coefficients
        annot_df : Dataframe with genes on index and annotations
        set_col  : Column from annotations to be used as gene set
        seed     : Set seed
    
    Returns:
        gs : gseapy object storing results of multilevel preranked GSEA
    """
    # Merge coefficients and annotations
    features = pd.merge(
        annot_df,
        coef_df,
        left_index = True,
        right_index = True,
        how = "outer"
    )

    # Generate custom gene sets
    gene_sets = construct_gene_set(
        df = features,
        set_col = set_col
    )

    # Compute rank using regression coefficients
    rnk = coef_df.copy()
    rnk = rnk.sort_values("coef", ascending = False)

    # Run gsea
    gs = gp.prerank(
        rnk = rnk,
        gene_sets = gene_sets,
        method = "multilevel",
        seed = seed
    )

    return gs