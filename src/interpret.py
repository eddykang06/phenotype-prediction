"""Functions for feature intpretation from trained models"""

import pandas as pd
import numpy as np


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