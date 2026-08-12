import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gseapy as gp
import seaborn as sns
import yaml
from pathlib import Path


def get_annotations(
    root: Path | str
):
    """
    Load TIGR4 annotation file from data directory config

    Args:
        root : Root directory of repo
    
    Returns:
        annotations: Annotation file in dataframe format
    """
    # Open config file
    config_path = Path(root / "configs" / "data_loader.yaml")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["data_dir"])
    annot_path = str(data_dir / "Annotation_TIGR4.tsv")

    annotations = pd.read_table(annot_path, sep = "\t")
    annotations.set_index("TIGR4.old", inplace = True, drop = True)

    return annotations


def get_deg_count(
    l2fc_df: pd.DataFrame,
    pval_df: pd.DataFrame,
    pval_cutoff: float,
    l2fc_cutoff: int | float
):
    """
    Generate a dataframe that shows # of DEGs for each condition, using specified log2fc and pvalue cutoffs

    Args:
        l2fc_df     : Dataframe of log2FC values with samples on index
        pval_df     : DataFrame of adjusted p-values with samples on index
        pval_cutoff : Cutoff for DEG significance
        l2fc_cutoff : Cutoff for DEG significance

    Returns:
        out : df with "num_degs" column for each sample
    """
    # Identify gene columns
    gene_cols = l2fc_df.columns[
        l2fc_df.columns.str.contains("SP")
    ]

    # Separate metadata and gene-level values
    meta = l2fc_df.drop(columns = gene_cols)
    l2fc = l2fc_df.loc[:, gene_cols]

    # Align p-values with both the rows and columns of l2fc
    pval = pval_df.reindex(
        index = l2fc.index,
        columns = gene_cols
    )
    deg_mask = (
        l2fc.abs().gt(l2fc_cutoff)
        & pval.lt(pval_cutoff)
    )
    out = meta.copy()
    out.insert(0, "num_deg", deg_mask.sum(axis = 1))

    return out


def plot_degs_over_time(
    l2fc_df: pd.DataFrame,
    pval_df: pd.DataFrame,
    pval_cutoff: float,
    l2fc_cutoff: float | int,
    drug_id: str
):
    """
    Plot the # of DEGs over time for a specified drug, colored by time

    Args:
        l2fc_df     : Dataframe of log2FC values with samples on index
        pval_df     : DataFrame of adjusted p-values with samples on index
        pval_cutoff : Cutoff for DEG significance
        l2fc_cutoff : Cutoff for DEG significance
        drug_id     : Drug ID of interest (ex: "CEF", "CEF+RIF", "VNC")
    """
    df = get_deg_count(
        l2fc_df = l2fc_df,
        pval_df = pval_df,
        pval_cutoff = pval_cutoff,
        l2fc_cutoff = l2fc_cutoff
    )

    # Filter to drug of interest
    filtered = df[df["drug_id"] == drug_id]
    
    sns.lineplot(
        data = filtered,
        x = "timepoint",
        y = "num_deg",
        hue = "drug1_dose"
    )
    plt.xlabel("Time (h)")
    plt.ylabel("Number of DEGs")
    plt.title(f"Number of DEGs over time for {drug_id}")


def plot_consistent_degs(
    l2fc_df: pd.DataFrame,
    pval_df: pd.DataFrame,
    annotations : pd.DataFrame,
    drug_id: str,
    pval_cutoff: float,
    l2fc_cutoff: float | int,
    min_fraction: float,
    figsize: tuple[int, int]
):
    """
    Plot DEGs for a drug that are consistenly DEG in the same direction across all samples from that drug

    Args:
        l2fc_df      : Dataframe of log2FC values for all samples and genes, with samples on index
        pval_df      : Dataframe of adjusted p-values for all samples and genes, with samples on index
        annotations  : Datafrmae of annotations with genes on index
        drug_id      : ID name of drug (ex: "CEF", "CIP+VNC")
        pval_cutoff  : Cutoff for DEG significance
        l2fc_cutoff  : Cutoff for DEG significance
        min_fraction : (ex: 0.5 means that a DEG must be present in >= 50% of samples for it to be a consistent DEG)
        figsize      : Tuple of plot dimensions
    """
    single_df = l2fc_df[l2fc_df["drug_id"] == drug_id]
    single_df = single_df.sort_values(["timepoint", "drug1_dose"])

    # Filter to 
    gene_cols = l2fc_df.columns[l2fc_df.columns.str.contains("^SP", regex = True, na = False)]
    samples = l2fc_df.index[l2fc_df["drug_id"] == drug_id]
    filtered_l2fc = l2fc_df.loc[samples][gene_cols]
    filtered_pval = pval_df.loc[samples]

    # Filter to significant hits
    sig_pval = filtered_pval < pval_cutoff

    up_deg = (filtered_l2fc > l2fc_cutoff) & sig_pval
    down_deg = (filtered_l2fc < -l2fc_cutoff) & sig_pval

    up_fraction = up_deg.mean(axis = 0)
    down_fraction = down_deg.mean(axis = 0)

    # Construct dataframe of consistent degs
    consistent_deg = pd.DataFrame({
        "drug": drug_id,
        "n_samples": filtered_l2fc.notna().sum(axis = 0),
        "up_count": up_deg.sum(axis = 0),
        "up_fraction": up_fraction,
        "down_count": down_deg.sum(axis = 0),
        "down_fraction": down_fraction,
        "mean_l2fc": filtered_l2fc.mean(axis = 0),
        "median_l2fc": filtered_l2fc.median(axis = 0),
    })

    consistent_deg["direction"] = np.select(
        [
            consistent_deg["up_fraction"] >= min_fraction,
            consistent_deg["down_fraction"] >= min_fraction,
        ],
        [
            "up",
            "down",
        ],
        default = "none",
    )

    consistent_deg = consistent_deg[consistent_deg["direction"] != "none"]
    consistent_deg["max_fraction"] = consistent_deg[["up_fraction", "down_fraction"]].max(axis = 1)

    consistent_deg = consistent_deg.sort_values(
        ["direction", "max_fraction", "mean_l2fc"],
        ascending = [False, False, False]
    )

    # Heatmap 
    fig, ax = plt.subplots(figsize = figsize)
    sns.heatmap(
        data = single_df[consistent_deg.index].T, 
        cmap = "coolwarm", 
        vmax = 5,
        vmin = -5,
        ax = ax,
        cbar_kws = {"label": "Log2FC"}
    )

    # Add annotations as extra axis
    annots = annotations.reindex(consistent_deg.index)
    secax = ax.secondary_yaxis("left")
    secax.set_yticks(np.arange(len(annots)) + 0.5)
    secax.set_yticklabels(annots["Product"])
    secax.spines["left"].set_position(("outward", 100))
    secax.set_ylabel("Product")

    ax.set_title(f"Top consistently high/low DEGs for {drug_id}")



def find_consistent_interaction_genes(
    df: pd.DataFrame,
    combo: str,
    left_cutoff: float,
    right_cutoff: float,
    min_fraction: float = 0.5,
    combo_col: str = "drug_id",
    gene_cols: list[str] = None,
    gene_pattern: str = "^SP",
    include_equal: bool = True,
):
    """
    Find genes with interaction scores consistently outside left/right cutoffs
    for a specific drug combination.

    Args:
        df            : Dataframe with interaction scores, metadata, and combos on rows
        combo         : Combination label or list of labels to select from combo_col
        left_cutoff   : Lower interaction score cutoff
        right_cutoff  : Upper interaction score cutoff
        min_fraction  : Minimum fraction of selected samples required to call a gene
        combo_col     : Column containing combination labels
        gene_cols     : Optional list of gene columns. If None, gene_pattern is used
        gene_pattern  : Regex pattern used to identify gene columns
        include_equal : Whether cutoff comparisons include equality

    Returns:
        hits : Gene-indexed dataframe of genes above right_cutoff or below
               left_cutoff in at least min_fraction of selected samples
    """
    if left_cutoff >= right_cutoff:
        raise ValueError("left_cutoff must be less than right_cutoff")

    if gene_cols is None:
        gene_cols = df.columns[df.columns.str.contains(gene_pattern, regex = True, na = False)]

    gene_cols = list(gene_cols)
    if len(gene_cols) == 0:
        raise ValueError("No gene columns found")

    if isinstance(combo, (str, int, float)):
        combo_values = [combo]
    else:
        combo_values = list(combo)

    combo_df = df[df[combo_col].isin(combo_values)].copy()
    if combo_df.empty:
        raise ValueError(f"No rows found for {combo_col}: {combo_values}")

    scores = combo_df[gene_cols].apply(pd.to_numeric, errors = "coerce")
    n_samples = scores.notna().sum(axis = 0)

    if include_equal:
        above_cutoff = scores.ge(right_cutoff)
        below_cutoff = scores.le(left_cutoff)
    else:
        above_cutoff = scores.gt(right_cutoff)
        below_cutoff = scores.lt(left_cutoff)

    above_count = above_cutoff.sum(axis = 0)
    below_count = below_cutoff.sum(axis = 0)
    above_fraction = above_count / n_samples.replace(0, np.nan)
    below_fraction = below_count / n_samples.replace(0, np.nan)

    above_hit = above_fraction >= min_fraction
    below_hit = below_fraction >= min_fraction
    hit_mask = above_hit | below_hit

    results = pd.DataFrame({
        "combo": ", ".join(map(str, combo_values)),
        "n_samples": n_samples,
        "above_count": above_count,
        "above_fraction": above_fraction,
        "below_count": below_count,
        "below_fraction": below_fraction,
        "mean_score": scores.mean(axis = 0),
        "median_score": scores.median(axis = 0),
    })

    results["direction"] = np.select(
        [
            above_hit & below_hit,
            above_hit,
            below_hit,
        ],
        [
            "both",
            "above",
            "below",
        ],
        default = "none",
    )

    hits = results.loc[hit_mask].copy()
    hits["max_fraction"] = hits[["above_fraction", "below_fraction"]].max(axis = 1)
    hits = hits.sort_values(
        ["direction", "max_fraction", "mean_score"],
        ascending = [True, False, False]
    )

    return hits


def plot_l2fc_heatmap(
    df: pd.DataFrame,
    vmax: float | int,
    annotations = None,
    drug_order = None,
    annot_col = None,
    gene_categories = None,
    figsize = (16, 10),
    cmap = "coolwarm",
    show_xticklabels = False,
    show_yticklabels = False,
    genes = None,
    secondary_annot_col = None,
):
    """
    Plot heatmap of log2fc for specified samples and genes

    Args:
        df                  : Log2FC data and metadata for all samples and genes, with sample on index 
        vmax                : Outer bound of log2FC to bound heatmap color gradient
        annotations         : Dataframe with gene annotations, genes on index 
        drug_order          : List of drug groups to plot on x-axis, in order
        annot_col           : Annotation column used to group genes on y-axis.
        gene_categories     : Optional list of gene annotation categories to plot, in order
        figsize             : Figure size
        cmap                : Heatmap color map
        show_xticklabels    : Whether to show sample labels
        show_yticklabels    : Whether to show gene labels
        genes               : Optional list of specific genes to plot, in order. Supply either genes or gene_categories, but not both
        secondary_annot_col : Optional annotation column to show as a secondary y-axis. Requires annotations
    """
    if drug_order is None:
        raise ValueError("drug_order must be provided")

    if (genes is None) == (gene_categories is None):
        raise ValueError("Provide exactly one of genes or gene_categories")

    if secondary_annot_col is not None:
        if annotations is None:
            raise ValueError(
                "annotations is required when using secondary_annot_col"
            )
        if secondary_annot_col not in annotations.columns:
            raise KeyError(
                f"{secondary_annot_col} not found in annotations columns"
            )

    # Filter and order samples by drug category
    plot_df = df[df["drug_id"].isin(drug_order)].copy()
    plot_df["drug_id"] = pd.Categorical(
        plot_df["drug_id"],
        categories = drug_order,
        ordered = True
    )

    plot_df = plot_df.sort_values([
        "drug_id",
        "timepoint",
        "drug1_dose",
        "drug2_dose"
    ])

    if genes is not None:
        if isinstance(genes, str):
            raise TypeError("genes must be a list-like collection, not a string")

        # dict.fromkeys removes duplicates while retaining the requested order.
        ordered_genes = list(dict.fromkeys(genes))
        if not ordered_genes:
            raise ValueError("genes must contain at least one gene")

        missing_genes = [gene for gene in ordered_genes if gene not in df.columns]
        if missing_genes:
            raise ValueError(
                "Genes not found in df columns: " + ", ".join(map(str, missing_genes))
            )
    else:
        if annotations is None or annot_col is None:
            raise ValueError(
                "annotations and annot_col are required when using gene_categories"
            )
        if annot_col not in annotations.columns:
            raise KeyError(f"{annot_col} not found in annotations columns")

        # Isolate gene columns and keep those represented in the annotations.
        gene_cols = df.columns[
            df.columns.str.contains("^SP", regex = True, na = False)
        ]
        shared_genes = annotations.index.intersection(gene_cols)

        # Get gene annotations and filter to selected categories.
        gene_annot = annotations.loc[shared_genes, [annot_col]].copy()
        gene_annot = gene_annot[gene_annot[annot_col].isin(gene_categories)]

        # Order genes by selected annotation categories.
        gene_annot[annot_col] = pd.Categorical(
            gene_annot[annot_col],
            categories = gene_categories,
            ordered = True
        )

        gene_annot = gene_annot.sort_values(annot_col)
        ordered_genes = gene_annot.index.tolist()

        if not ordered_genes:
            raise ValueError("No genes found for the requested gene_categories")

    # Gene x sample matrix
    mat = plot_df[ordered_genes].T.astype(float)

    # Generate symmetric color gradient around 0
    vmax = vmax

    fig, ax = plt.subplots(figsize = figsize)

    sns.heatmap(
        mat,
        ax = ax,
        cmap = cmap,
        center = 0,
        vmin = -vmax,
        vmax = vmax,
        xticklabels = show_xticklabels,
        yticklabels = show_yticklabels,
        cbar_kws={"label": "Log2FC"}
    )

    # -------------------------
    # X-axis drug group labels
    # -------------------------
    x_group_sizes = plot_df["drug_id"].value_counts(sort = False).reindex(drug_order)

    x_boundaries = x_group_sizes.cumsum()[:-1]
    for boundary in x_boundaries:
        ax.axvline(boundary, color = "black", linewidth = 2)

    x_starts = np.r_[0, x_group_sizes.cumsum().values[:-1]]
    x_centers = x_starts + x_group_sizes.values / 2

    for center, label in zip(x_centers, drug_order):
        ax.text(
            center,
            -1.5,
            label,
            ha = "center",
            va = "bottom",
            fontsize = 12,
            fontweight = "bold",
            clip_on = False
        )

    if genes is None:
        # -------------------------
        # Y-axis annotation groups
        # -------------------------
        y_groups = gene_annot.loc[mat.index, annot_col]
        y_group_sizes = y_groups.value_counts(sort = False).reindex(gene_categories)

        y_boundaries = y_group_sizes.cumsum()[:-1]
        for boundary in y_boundaries:
            ax.axhline(boundary, color = "black", linewidth = 2)

        y_starts = np.r_[0, y_group_sizes.cumsum().values[:-1]]
        y_centers = y_starts + y_group_sizes.values / 2

        for center, label in zip(y_centers, gene_categories):
            ax.text(
                -7,
                center,
                label,
                ha = "center",
                va = "center",
                fontsize = 11,
                fontweight = "bold",
                rotation = 90,
                clip_on = False
            )

    if secondary_annot_col is not None:
        # Align per-gene annotations to the plotted matrix, including when genes
        # were reordered by annotation category.
        secondary_labels = (
            annotations.reindex(mat.index)[secondary_annot_col]
            .fillna("")
            .astype(str)
        )
        secondary_ax = ax.secondary_yaxis("left")
        secondary_ax.set_yticks(np.arange(len(mat.index)) + 0.5)
        secondary_ax.set_yticklabels(secondary_labels)
        secondary_ax.spines["left"].set_position(("outward", 130))
        secondary_ax.set_ylabel(secondary_annot_col)

    ax.set_xlabel("Samples grouped by drug condition")
    if genes is None:
        ax.set_ylabel(f"Genes grouped by {annot_col}")
    else:
        ax.set_ylabel("Genes")

    plt.tight_layout()
    plt.show()
