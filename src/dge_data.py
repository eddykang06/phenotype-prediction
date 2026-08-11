"""Data extraction functions for models using DGE data to predict synergy"""
import pandas as pd
import numpy as np
import os
import yaml
from src.metadata import attach_synergy_metadata
from pathlib import Path

def clean_csv_name(name):
    """
    Function to clean a csv name into a readable sample ID

    Args:
        name : ex: "T4-wt1AMX1hr-T4-wtNDC0hr-DGE-results"
    
    Returns:
        cleaned : ex: "1AMX1hr"
    """
    suffixes = [
        ".csv",
        "-T4-wtNDC0hr-DGE-results",
        "-T4-wtNDC1hr-DGE-results",
        "-T4-wtNDC2hr-DGE-results",
        "-T4-wtNDC4hr-DGE-results",
        "T4-wt"
    ]
    
    cleaned = name

    for suffix in suffixes:
        cleaned = cleaned.replace(suffix, "")

    return cleaned


def read_l2fc_as_df(data_dir, time_matched, pval = False):
    """
    Function to extract all l2fc values from a directory of DGE results for various conditions.
    The output will be a list of dataframes, where each DGE result is stored as a dataframe with gene IDs on index, and 1 column of l2fc values

    Args:
        data_dir     : path to folder containing DGE results
        time_matched : boolean indicating whether DGE results should be from time zero or time-matched NDC

    Returns:
        l2fc_df_list : list of dataframes storing l2fc results
        names        : sample IDs corresponding to each dataframe
    """
    all_files = os.listdir(data_dir)
    files = []

    # Time-matched NDC comparisons
    if time_matched:
        for f in all_files:
            if f.endswith(".csv") and "NDC0hr" not in f:
                files.append(f)
    
    # Time zero comparisons
    else:
        for f in all_files:
            if f.endswith(".csv") and "NDC0hr" in f:
                files.append(f)

    if len(files) == 0: 
        raise FileNotFoundError(f"No matching DGE files found in {data_dir}")

    # Sort and specify path to data files
    files = sorted(files)

    # Get sample IDs
    ids = [clean_csv_name(f) for f in files]

    # Convert csvs to dataframes
    filenames = ["".join([data_dir, "/" , f]) for f in files]
    l2fc_df_list = [pd.read_table(filenames[i], 
                        sep = ",", 
                        header = 0, 
                        index_col = 1)["log2FoldChange"].rename(ids[i]) # Genes
                        for i in range(len(filenames))]

    if pval:
        pval_df_list = [pd.read_table(filenames[i], 
                            sep = ",", 
                            header = 0, 
                            index_col = 1)["padj"].rename(ids[i]) # Genes
                            for i in range(len(filenames))]
        return l2fc_df_list, pval_df_list, ids
    
    return l2fc_df_list, ids


def bind_l2fc_data(l2fc_df_list, ids):
    """
    Function to take a list of l2fc DEG dataframes, then bind all into 1 dataframe

    Args: 
        l2fc_df_list : list of l2fc dataframes
        ids          : corresponding sample IDs

    Returns:
        all_l2fc [N,G] : dataframe with all feature values (N samples on row, G genes on column)
    """
    # Join and transpose to get genes on columns
    all_l2fc = pd.concat(l2fc_df_list, axis = 1, join = "outer").T
    all_l2fc.index = ids

    return all_l2fc


def read_avg_cfus(folder_path):
    """
    Function to extract CFUs into a dataframe with conditions on rows, 1 CFU column
    Args:
        folder_path : path to folder containing CFUs (same format as /all_cfus)

    Returns:
        all_avg_cfus [N,1] : df with condition names as index, 1 column of avg CFUs across triplicates (N = # samples)
    """

    # Get files
    files = os.listdir(folder_path)
    
    # Select CSV files
    cfu_files = [csv for csv in files if ".csv" in csv]
    cfu_files = sorted(cfu_files)
    cfu_files = ["".join([folder_path, "/", csv]) for csv in files]
    
    # Load each file as a dataframe
    cfu_dfs = [pd.read_table(csv, sep = ",", header = 0) for csv in cfu_files] 

    # change all of this to calculate average CFU for each condition
    for i, df in enumerate(cfu_dfs):

        # Remove the triplicate column
        df = df.drop(columns = "Triplicates")

        # Calculate mean of three triplicates and store
        means = df.mean()
        cfu_dfs[i] = means
    
    # Concat
    all_avg_cfus = pd.concat(cfu_dfs, axis = 0) # series
    all_avg_cfus = all_avg_cfus.rename("CFU").to_frame()

    # Remove space breaker characters
    all_avg_cfus.index = [id.replace("\xa0", "") for id in all_avg_cfus.index]
    
    return all_avg_cfus


def bind_l2fc_and_cfu_data(feature_df, cfu_df):
    """
    Function bind TPM and cfu dfs
    Args:
        featurem_df [N,G] : Dataframe of TPMs, N = # samples, G = # genes, labels on index
        cfu_df [N,1]      : Dataframe of CFUs, labels on index

    Returns:
        data_df : [N, G+1] : Dataframe of all TPMs and CFUs as last column
    """
    # Right join so that CFUs exist
    data_df = pd.merge(feature_df, cfu_df, left_index = True, right_index = True, how = "right")
    data_df = data_df.iloc[data_df.index.argsort()]

    return data_df


def get_l2fc_and_cfu_data(l2fc_dir, cfu_dir, time_matched):
    
    l2fc_df_list, ids = read_l2fc_as_df(l2fc_dir, time_matched)
    all_l2fc = bind_l2fc_data(l2fc_df_list, ids)
    all_avg_cfus = read_avg_cfus(cfu_dir)
    data_df = bind_l2fc_and_cfu_data(all_l2fc, all_avg_cfus)
    
    return data_df


"""Functions to calcualte transcriptional interaction scores"""
def simple_interaction_score(l2fc_a, l2fc_b, l2fc_ab):
    """
    Function to a compute a simple interaction score to quantify non-additivity in gene expression 
    using single drug log2foldchange and combination log2foldchange

    Args:
        l2fc_a   : Log2foldchange in condition A
        l2fc_b   : Log2foldchange in condition B
        l2fc_ab : Log2foldchange in combination

    Returns:
        score : Simple interaction score
    """
    score = l2fc_ab - l2fc_a - l2fc_b

    return score


"""Functions to calculate drug synergy scores"""
def eob_score(v_a, v_b, v_ab):
    """
    Function to calculate the excess over bliss (EOB) synergy score for given single drug survival and a 
    combination drug survival

    Args:
        v_a  : Fraction surviving cells in condition A
        v_b  : Fraction surviving cells in condition B
        v_ab : Fraction surviving cells in condition A+B

    Returns:
        score: EOB score calculated using fraction of surviving cells
    """
    # EOB score (calculated using survival fraction)
    score = v_a * v_b - v_ab

    return score


def hsa_score(v_a, v_b, v_ab):
    """  
    Function to calculate the HSA synergy score for given single drug survival and a 
    combination drug survival

    Args:
        v_a  : Fraction surviving cells in condition A
        v_b  : Fraction surviving cells in condition B
        v_ab : Fraction surviving cells in condition A+B

    Returns:
        score: HSA score calculated using fraction of surviving cells
    """
    # HSA score (calculated using survival fraction)
    score = np.min([v_a, v_b]) - v_ab

    return score


"""Functions to generate new data df for synergy prediction"""
def construct_synergy_df(df, interaction_score_method, synergy_score_method):
    """
    Function to construct a dataframe for synergy prediction. Each condition corresponds to a drug combination,
    and each feature will be a transcriptional interaction score calculated according to specified function.
    There will be one column of synergy score as the prediction target.

    Args:
        combo_df                 : Dataframe containing all log2foldchange data and metadata
        interaction_score_method : Function for calculating transcriptional interaction
        synergy_score_method     : Function for calculating synergy using survival fraction

    Returns:
        synergy_df : Dataframe with transcriptaional interaction score as features and EOB score as prediction target
    """
    # Check that the only unique num_drugs are 1 and 2
    unique_num_drugs = df["num_drugs"].unique()

    if set(unique_num_drugs) != set([1, 2]):
        raise KeyError("num_drugs column contains values other than 1, 2")
    
    # Separate data df into single and combination data
    single_df = df.iloc[(df["num_drugs"] == 1).to_numpy()]
    combo_df = df.iloc[(df["num_drugs"] == 2).to_numpy()]
    
    # Store data size
    num_single = single_df.shape[0]
    num_combos = combo_df.shape[0]

    # Identify gene / transcript columns
    gene_cols = combo_df.columns[combo_df.columns.str.contains("SP", na = False)]
    
    rows = []

    for i in range(num_combos):

        # Store current combo data and specify drug 1, drug 2, doses, and timepoint
        combo_data = combo_df.iloc[i]
        combo_id = combo_df.index[i]
        drug_id = combo_data["drug_id"]
        drug1 = combo_data["drug1"]
        drug2 = combo_data["drug2"]
        drug1_dose = combo_data["drug1_dose"]
        drug2_dose = combo_data["drug2_dose"]
        timepoint = combo_data["timepoint"]

        # Select the two corresponding single drug datapoints from the single drug dataframe
        single_data1_bool = [
            i for i in range(num_single) if 
            single_df.iloc[i]["drug1"] == drug1
            and pd.isna(single_df.iloc[i]["drug2"])
            and single_df.iloc[i]["drug1_dose"] == drug1_dose
            and single_df.iloc[i]["timepoint"] == timepoint
        ]
        single_data2_bool = [
            i for i in range(num_single) if 
            single_df.iloc[i]["drug1"] == drug2
            and pd.isna(single_df.iloc[i]["drug2"])
            and single_df.iloc[i]["drug1_dose"] == drug2_dose 
            and single_df.iloc[i]["timepoint"] == timepoint
        ]
        single_data1 = single_df.iloc[single_data1_bool].iloc[0]
        single_data2 = single_df.iloc[single_data2_bool].iloc[0]
        
        # Extract the survival fractions and compute synergy score
        surv_frac_combo = combo_data["survival_fraction"]
        surv_frac1 = single_data1["survival_fraction"]
        surv_frac2 = single_data2["survival_fraction"]

        synergy_score = synergy_score_method(surv_frac1, surv_frac2, surv_frac_combo)

        # Extract gene expression values
        l2fc_1 = single_data1[gene_cols]
        l2fc_2 = single_data2[gene_cols]
        l2fc_combo = combo_data[gene_cols]
        
        # Compute transcriptional interaction score for each gene
        interaction_scores = {
            gene: interaction_score_method(v_a, v_b, v_ab)
            for gene, v_a, v_b, v_ab in zip(
                gene_cols,
                l2fc_1,
                l2fc_2,
                l2fc_combo
            )
        }

        # Build row
        row = {
            "ID": combo_id,
            "drug_id": drug_id,
            "drug1": drug1,
            "drug2": drug2,
            "drug1_dose": drug1_dose,
            "drug2_dose": drug2_dose,
            "timepoint": timepoint,
            **interaction_scores,
            "synergy_score": synergy_score
        }
        rows.append(row)
    
    # Construct dataframe and move condition label to index
    synergy_df = pd.DataFrame(rows)
    synergy_df = synergy_df.set_index("ID")

    return synergy_df


def get_all_synergy_data(
        l2fc_dir, 
        cfu_dir, 
        interaction_score_method,
        synergy_score_method,
        time_matched,
):
    """
    Function to run the entire data loading pipeline to output a Dataframe containing
    all transcriptional interaction scores, synergy scores, and metadata
    """
    df = get_l2fc_and_cfu_data(l2fc_dir, cfu_dir, time_matched = time_matched)
    df = attach_synergy_metadata(df)
    df = construct_synergy_df(
        df = df,
        interaction_score_method = interaction_score_method,
        synergy_score_method = synergy_score_method
    )

    return df


def get_synergy(
    root: Path | str
):
    """
    
    Load all synergy data from data directory config
    """
    config_path = Path(root / "configs" / "data_loader.yaml")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["data_dir"])
    l2fc_dir = str(data_dir / "dge")
    cfu_dir = str(data_dir / "cfus")

    # Use synergy loading pipeline
    df = get_all_synergy_data(
        l2fc_dir = l2fc_dir,
        cfu_dir = cfu_dir,
        interaction_score_method = simple_interaction_score,
        synergy_score_method = eob_score,
        time_matched = True
    )

    # Drop genes with NA values
    df = df.dropna(axis = 1)

    return df

def get_l2fc(
    root: Path | StopIteration
):
    """
    Load all log2fc data from data directory config
    """
    config_path = Path(root / "configs" / "data_loader.yaml")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["data_dir"])
    l2fc_dir = str(data_dir / "dge")
    cfu_dir = str(data_dir / "cfus")
    df = attach_synergy_metadata(get_l2fc_and_cfu_data(l2fc_dir, cfu_dir, time_matched = True))

    return df


def get_pval(
    root: Path | str
):
    """
    Load all adjusted pvalues from data directory config
    """
    # Open config file
    config_path = Path(root / "configs" / "data_loader.yaml")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["data_dir"])
    l2fc_dir = str(data_dir / "dge")

    # Get list of pvalues
    _, pval_df_list, ids = read_l2fc_as_df(
        data_dir = l2fc_dir,
        time_matched = True,
        pval = True
    )
    # Bind and filter to relevant samples
    df = bind_l2fc_data(
        l2fc_df_list = pval_df_list,
        ids = ids
    )
    df = df.iloc[~df.index.str.contains("NDC")]

    return df

