#!/usr/bin/env python3
"""
Downsample Kamath 2022 and Siletti 2023 to ~10k cells each, run QC, and cache
a (preQC, postQC) H5AD pair per source for notebooks/03_integrate_kamath_siletti.ipynb.

Kamath et al. 2022 (reference for ingest)
  DA-first stratified draw across the 7 cell-type files:
  - --kamath-da-per-subtype cells from each of the 10 DA subtypes
    (author_cell_type in the DA neurons file; default 300 -> 3,000 DA cells)
  - the rest of --cells-per-source from the 58 non-DA author_cell_types:
    at least --kamath-non-da-min per type (or all if fewer), then a random
    top-up from the remaining non-DA cells (so common types stay common)

Siletti et al. 2023 ("new" dataset for ingest)
  Pooled across the 3 midbrain dissections (SN-RN, SN, RN):
  - EVERY Allen-annotated "DA VGLUT2" cell (~810)
  - topped up at random to --cells-per-source

Downsampling happens BEFORE QC, so scrublet only runs on the selected cells.
QC (mt%, scrublet doublet score, total counts, genes detected) follows
sc-rnaseq-wf's cohort_analysis defaults, with the upper count/gene limits
raised for Kamath's large DA neurons (see the original repo).

Inputs
  data/ files from 01_download_kamath_siletti.py

Outputs
  data/kamath/intermediate/kamath_n{N}_da{D}_nonda_min{M}_{preQC,postQC}.h5ad
  data/siletti/intermediate/siletti_midbrain_n{N}_all_da_{preQC,postQC}.h5ad

Dry-run by default: prints the planned per-type cell counts. Pass --execute
to load the cells, run QC and write the caches.

Status tags
  [CACHED]   cache pair already present, source skipped
  [DRY-RUN]  cache pair would be written
  [WRITE]    cache file is being written
"""

import os
import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
from _da_neurons_utils import (
    DATA_DIR,
    KAMATH_DIR,
    SILETTI_DIR,
    KAMATH_FILE_STEMS,
    KAMATH_DA_FILE_STEM,
    SILETTI_REGION_FILE_STEMS,
    ALLEN_CELL_METADATA_PATH,
    ALLEN_MEMBERSHIP_PATH,
    ALLEN_DA_NEUROTRANSMITTER,
    DEFAULT_CELLS_PER_SOURCE,
    DEFAULT_KAMATH_DA_PER_SUBTYPE,
    DEFAULT_KAMATH_NON_DA_MIN,
    kamath_cache_paths,
    siletti_cache_paths,
    extract_raw_counts,
    chdir_to_repo_root,
    log_run_command,
)

warnings.filterwarnings("ignore", category=FutureWarning)

_SEED = 42

# CELLxGENE 'disease' values not in this map (e.g. Lewy body dementia) -> "Other".
_CELLXGENE_DISEASE_TO_CONDITION_ID = {"normal": "Control", "Parkinson disease": "PD"}

# QC thresholds: sc-rnaseq-wf cohort_analysis defaults, with the upper limits
# raised because real Kamath DA neurons reach ~317k UMIs and ~13k genes.
_QC_PCT_COUNTS_MT_MAX = 10
_QC_DOUBLET_SCORE_MAX = 0.2
_QC_TOTAL_COUNTS_LIMITS = (100, 150_000)
_QC_N_GENES_LIMITS = (100, 20_000)

# The 13 mitochondrial protein-coding genes (GRCh38 Ensembl IDs); Ensembl IDs
# are used because both sources' var_names are Ensembl IDs.
_MT_GENE_ENSEMBL_IDS = {
    "ENSG00000198888", "ENSG00000198763", "ENSG00000198804", "ENSG00000198712",
    "ENSG00000228253", "ENSG00000198899", "ENSG00000198938", "ENSG00000198840",
    "ENSG00000212907", "ENSG00000198886", "ENSG00000198786", "ENSG00000198695",
    "ENSG00000198727",
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def harmonize_disease(disease: pd.Series) -> pd.Series:
    """
    Map a CELLxGENE "disease" column to "PD", "Control" or "Other".

    Parameters
    ----------
    disease : pandas.Series
        CELLxGENE disease labels.

    Returns
    -------
    pandas.Series
        Harmonised condition labels.
    """
    # astype(str) first: .map() on a categorical keeps only mapped categories.
    return disease.astype(str).map(_CELLXGENE_DISEASE_TO_CONDITION_ID).fillna("Other")


def run_scrublet_per_batch(adata: ad.AnnData, batch_key: str) -> np.ndarray:
    """
    Score doublets with scrublet separately for each batch, on 2,000 HVGs.

    Per-batch scoring matches how doublets form (within one library) and
    avoids scanpy's own `batch_key=` path, which crashes on real data.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData with raw counts in `layers["counts"]`.
    batch_key : str
        `obs` column with the donor/sample/library unit.

    Returns
    -------
    numpy.ndarray
        Doublet score per cell; NaN for batches scrublet couldn't score.
    """
    # flavor="seurat" on log-normalised counts (seurat_v3 needs scikit-misc,
    # which has no wheel for every platform).
    lognorm = ad.AnnData(X=adata.layers["counts"].astype(np.float32), obs=adata.obs[[]], var=adata.var[[]])
    sc.pp.normalize_total(lognorm, target_sum=1e4)
    sc.pp.log1p(lognorm)
    hvg = sc.pp.highly_variable_genes(
        lognorm, flavor="seurat", n_top_genes=min(2000, adata.n_vars), inplace=False,
    )["highly_variable"].to_numpy()
    adata_hvg = adata[:, hvg].copy()
    adata_hvg.X = adata_hvg.layers["counts"]

    scores = pd.Series(np.nan, index=adata_hvg.obs_names)
    for batch in adata_hvg.obs[batch_key].unique():
        mask = (adata_hvg.obs[batch_key] == batch).to_numpy()
        if mask.sum() < 5:
            continue
        adata_batch = adata_hvg[mask].copy()
        try:
            sc.pp.scrublet(adata_batch, n_prin_comps=min(30, int(mask.sum()) - 1), verbose=False)
            scores.loc[adata_batch.obs_names] = adata_batch.obs["doublet_score"].to_numpy()
        except ValueError:
            pass
    return scores.to_numpy()


def apply_qc_filters(adata: ad.AnnData, label: str, doublet_batch_key: str) -> ad.AnnData:
    """
    Filter cells on mt%, doublet score, total counts and genes detected.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData with raw counts in `layers["counts"]`; `obs` gains
        "pct_counts_mt", "total_counts", "n_genes" and "doublet_score".
    label : str
        Label for log lines.
    doublet_batch_key : str
        `obs` column passed to `run_scrublet_per_batch`.

    Returns
    -------
    anndata.AnnData
        Filtered copy.
    """
    counts = adata.layers["counts"]
    total_counts = np.asarray(counts.sum(axis=1)).ravel()
    n_genes = np.asarray((counts > 0).sum(axis=1)).ravel()
    mt_counts = np.asarray(counts[:, adata.var_names.isin(_MT_GENE_ENSEMBL_IDS)].sum(axis=1)).ravel()
    pct_counts_mt = np.where(total_counts > 0, mt_counts / np.maximum(total_counts, 1) * 100, 0.0)
    doublet_score = run_scrublet_per_batch(adata, doublet_batch_key)

    adata.obs["total_counts"] = total_counts
    adata.obs["n_genes"] = n_genes
    adata.obs["pct_counts_mt"] = pct_counts_mt
    adata.obs["doublet_score"] = doublet_score

    pass_mt = pct_counts_mt <= _QC_PCT_COUNTS_MT_MAX
    pass_doublet = doublet_score <= _QC_DOUBLET_SCORE_MAX  # NaN fails
    pass_counts = (total_counts >= _QC_TOTAL_COUNTS_LIMITS[0]) & (total_counts <= _QC_TOTAL_COUNTS_LIMITS[1])
    pass_genes = (n_genes >= _QC_N_GENES_LIMITS[0]) & (n_genes <= _QC_N_GENES_LIMITS[1])
    keep = pass_mt & pass_doublet & pass_counts & pass_genes
    print(
        f"  QC ({label}): {adata.n_obs:,} -> {int(keep.sum()):,} cells "
        f"(fail mt%: {int((~pass_mt).sum()):,}, doublet: {int((~pass_doublet).sum()):,}, "
        f"counts: {int((~pass_counts).sum()):,}, genes: {int((~pass_genes).sum()):,}; can overlap)"
    )
    return adata[keep].copy()


def write_cache_pair(preqc: ad.AnnData, postqc: ad.AnnData, paths: tuple[str, str]) -> None:
    """
    Write a (preQC, postQC) pair of H5ADs.

    Parameters
    ----------
    preqc : anndata.AnnData
        Downsampled, unfiltered cells.
    postqc : anndata.AnnData
        Same cells after QC.
    paths : tuple[str, str]
        Destination paths (preQC, postQC).

    Returns
    -------
    None
    """
    for adata, path in zip((preqc, postqc), paths, strict=True):
        print(f"  [WRITE] {adata.n_obs:,} cells x {adata.n_vars:,} genes → {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        adata.write_h5ad(path, compression="gzip")


def random_subset(items: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Draw `n` items without replacement (all of them if `n` >= len).

    Parameters
    ----------
    items : numpy.ndarray
        Items to draw from.
    n : int
        Number of items to draw.
    rng : numpy.random.Generator
        Random generator.

    Returns
    -------
    numpy.ndarray
        The drawn items.
    """
    if n >= len(items):
        return items
    return items[rng.choice(len(items), size=n, replace=False)]


# ---------------------------------------------------------------------------
# Kamath
# ---------------------------------------------------------------------------

def scan_kamath_obs(da_subtype_column: str) -> pd.DataFrame:
    """
    Read the cell-type label of every Kamath cell in backed mode (no matrices).

    Parameters
    ----------
    da_subtype_column : str
        `obs` column with fine-grained labels (incl. the 10 DA subtypes).

    Returns
    -------
    pandas.DataFrame
        One row per cell: "file_stem", "row" (position in its file),
        "label" and "is_da_file".
    """
    frames = []
    for stem in KAMATH_FILE_STEMS:
        path = os.path.join(KAMATH_DIR, f"{stem}.h5ad")
        backed = sc.read_h5ad(path, backed="r")
        labels = backed.obs[da_subtype_column].astype(str).to_numpy()
        backed.file.close()
        frames.append(pd.DataFrame({
            "file_stem": stem,
            "row": np.arange(len(labels)),
            "label": labels,
            "is_da_file": stem == KAMATH_DA_FILE_STEM,
        }))
        print(f"  Scanned {stem}: {len(labels):,} cells")
    return pd.concat(frames, ignore_index=True)


def plan_kamath_selection(
    cells: pd.DataFrame,
    n_total: int,
    da_per_subtype: int,
    non_da_min: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Choose which Kamath cells to keep (DA-first stratified draw).

    Parameters
    ----------
    cells : pandas.DataFrame
        Output of `scan_kamath_obs`.
    n_total : int
        Total cells to draw.
    da_per_subtype : int
        Cells per DA subtype (all of a subtype if it has fewer).
    non_da_min : int
        Minimum cells per non-DA label before the random top-up.
    rng : numpy.random.Generator
        Random generator.

    Returns
    -------
    pandas.DataFrame
        The selected rows of `cells`.
    """
    da = cells[cells["is_da_file"]]
    non_da = cells[~cells["is_da_file"]]

    da_idx = np.concatenate([
        random_subset(group.index.to_numpy(), da_per_subtype, rng)
        for _, group in da.groupby("label")
    ])
    floor_idx = np.concatenate([
        random_subset(group.index.to_numpy(), non_da_min, rng)
        for _, group in non_da.groupby("label")
    ])
    surplus = np.setdiff1d(non_da.index.to_numpy(), floor_idx)
    n_topup = max(0, n_total - len(da_idx) - len(floor_idx))
    topup_idx = random_subset(surplus, n_topup, rng)

    print(
        f"  Selected {len(da_idx):,} DA ({da['label'].nunique()} subtypes x <= {da_per_subtype}) + "
        f"{len(floor_idx):,} non-DA floor ({non_da['label'].nunique()} types x <= {non_da_min}) + "
        f"{len(topup_idx):,} non-DA random top-up = {len(da_idx) + len(floor_idx) + len(topup_idx):,} cells"
    )
    return cells.loc[np.sort(np.concatenate([da_idx, floor_idx, topup_idx]))]


def load_kamath(selection: pd.DataFrame, da_subtype_column: str) -> ad.AnnData:
    """
    Load the selected Kamath cells and add harmonised `obs` columns.

    Parameters
    ----------
    selection : pandas.DataFrame
        Output of `plan_kamath_selection`.
    da_subtype_column : str
        `obs` column with fine-grained labels.

    Returns
    -------
    anndata.AnnData
        Selected cells with raw counts in `layers["counts"]`.
    """
    chunks = []
    for stem, rows in selection.groupby("file_stem")["row"]:
        backed = sc.read_h5ad(os.path.join(KAMATH_DIR, f"{stem}.h5ad"), backed="r")
        chunk = backed[np.sort(rows.to_numpy())].to_memory()
        backed.file.close()
        chunk.layers["counts"] = extract_raw_counts(chunk, stem)
        print(f"  Loaded {chunk.n_obs:,} cells from {stem}")
        chunks.append(chunk)
    kamath = ad.concat(chunks, join="inner", merge="same")

    is_da = kamath.obs["cell_type"].astype(str).str.contains("dopaminergic", case=False)
    kamath.obs["source"] = "kamath"
    kamath.obs["batch"] = "kamath_" + kamath.obs["donor_id"].astype(str)
    kamath.obs["condition_id"] = harmonize_disease(kamath.obs["disease"])
    kamath.obs["is_da"] = is_da.to_numpy()
    kamath.obs["kamath_cell_type"] = kamath.obs["cell_type"].astype(str)
    kamath.obs["kamath_author_cell_type"] = kamath.obs[da_subtype_column].astype(str)
    kamath.obs["kamath_da_subtype"] = kamath.obs["kamath_author_cell_type"].where(is_da, other=np.nan)
    kamath.obs["original_barcode"] = kamath.obs_names.astype(str)
    kamath.obs_names = "kamath_" + kamath.obs_names.astype(str)
    return kamath


# ---------------------------------------------------------------------------
# Siletti
# ---------------------------------------------------------------------------

def load_allen_annotations(barcodes: pd.Index) -> pd.DataFrame:
    """
    Look up Allen Brain Cell Atlas annotations for Siletti barcodes.

    Parameters
    ----------
    barcodes : pandas.Index
        Siletti `obs_names` (same format as Allen's "cell_label").

    Returns
    -------
    pandas.DataFrame
        Indexed by barcode, with "neurotransmitter", "supercluster",
        "cluster" and "subcluster" (NaN for barcodes Allen doesn't annotate).
    """
    # keep_default_na=False: "None" is a real neurotransmitter category.
    membership = pd.read_csv(ALLEN_MEMBERSHIP_PATH, compression="gzip", keep_default_na=False)
    cell_meta = pd.read_csv(
        ALLEN_CELL_METADATA_PATH, compression="gzip", usecols=["cell_label", "cluster_alias"],
    )
    cell_meta = cell_meta[cell_meta["cell_label"].isin(barcodes)].set_index("cell_label")

    annotations = pd.DataFrame(index=barcodes)
    for term_set in ("neurotransmitter", "supercluster", "cluster", "subcluster"):
        alias_to_term = membership.loc[
            membership["cluster_annotation_term_set_name"] == term_set,
            ["cluster_alias", "cluster_annotation_term_name"],
        ].drop_duplicates("cluster_alias").set_index("cluster_alias")["cluster_annotation_term_name"]
        annotations[term_set] = cell_meta["cluster_alias"].map(alias_to_term).reindex(barcodes).to_numpy()
    print(f"  Allen annotations found for {annotations['cluster'].notna().sum():,}/{len(barcodes):,} barcodes")
    return annotations


def scan_siletti_obs() -> pd.DataFrame:
    """
    List every barcode in the 3 Siletti midbrain dissections (backed mode).

    Returns
    -------
    pandas.DataFrame
        One row per cell, indexed by barcode, with "region" and "row".
    """
    frames = []
    for region, stem in SILETTI_REGION_FILE_STEMS.items():
        backed = sc.read_h5ad(os.path.join(SILETTI_DIR, f"{stem}.h5ad"), backed="r")
        frames.append(pd.DataFrame(
            {"region": region, "row": np.arange(backed.n_obs)}, index=backed.obs_names.copy(),
        ))
        backed.file.close()
        print(f"  Scanned Siletti {region}: {frames[-1].shape[0]:,} cells")
    return pd.concat(frames)


def plan_siletti_selection(
    cells: pd.DataFrame,
    allen: pd.DataFrame,
    n_total: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Choose which Siletti cells to keep: every DA cell, then a random top-up.

    Parameters
    ----------
    cells : pandas.DataFrame
        Output of `scan_siletti_obs`.
    allen : pandas.DataFrame
        Output of `load_allen_annotations`.
    n_total : int
        Total cells to draw.
    rng : numpy.random.Generator
        Random generator.

    Returns
    -------
    pandas.DataFrame
        The selected rows of `cells`.
    """
    is_da = (allen["neurotransmitter"] == ALLEN_DA_NEUROTRANSMITTER).reindex(cells.index).fillna(False).to_numpy()
    da_idx = np.flatnonzero(is_da)
    topup_idx = random_subset(np.flatnonzero(~is_da), max(0, n_total - len(da_idx)), rng)
    selected = cells.iloc[np.sort(np.concatenate([da_idx, topup_idx]))]
    per_region = pd.DataFrame({"DA (all kept)": cells[is_da]["region"].value_counts(),
                               "selected": selected["region"].value_counts(),
                               "available": cells["region"].value_counts()})
    print(per_region.fillna(0).astype(int).to_string())
    print(f"  Selected {len(da_idx):,} DA + {len(topup_idx):,} random = {len(selected):,} cells")
    return selected


def load_siletti(selection: pd.DataFrame, allen: pd.DataFrame) -> ad.AnnData:
    """
    Load the selected Siletti cells and add harmonised `obs` columns.

    Parameters
    ----------
    selection : pandas.DataFrame
        Output of `plan_siletti_selection`.
    allen : pandas.DataFrame
        Output of `load_allen_annotations`.

    Returns
    -------
    anndata.AnnData
        Selected cells with raw counts in `layers["counts"]`.
    """
    chunks = []
    for region, stem in SILETTI_REGION_FILE_STEMS.items():
        rows = np.sort(selection.loc[selection["region"] == region, "row"].to_numpy())
        backed = sc.read_h5ad(os.path.join(SILETTI_DIR, f"{stem}.h5ad"), backed="r")
        chunk = backed[rows].to_memory()
        backed.file.close()
        chunk.layers["counts"] = extract_raw_counts(chunk, stem)
        chunk.obs["siletti_region"] = region
        print(f"  Loaded {chunk.n_obs:,} cells from Siletti {region}")
        chunks.append(chunk)
    siletti = ad.concat(chunks, join="inner", merge="same")

    cell_allen = allen.reindex(siletti.obs_names)
    siletti.obs["source"] = "siletti"
    siletti.obs["batch"] = "siletti_" + siletti.obs["sample_id"].astype(str)
    siletti.obs["condition_id"] = harmonize_disease(siletti.obs["disease"])
    siletti.obs["is_da"] = (cell_allen["neurotransmitter"] == ALLEN_DA_NEUROTRANSMITTER).to_numpy()
    siletti.obs["siletti_cell_type"] = siletti.obs["cell_type"].astype(str)
    siletti.obs["siletti_supercluster"] = siletti.obs["supercluster_term"].astype(str)
    for term_set in ("neurotransmitter", "supercluster", "cluster", "subcluster"):
        siletti.obs[f"siletti_allen_{term_set}"] = cell_allen[term_set].to_numpy()
    siletti.obs["original_barcode"] = siletti.obs_names.astype(str)
    siletti.obs_names = "siletti_" + siletti.obs_names.astype(str)
    return siletti


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def caches_exist(paths: tuple[str, str]) -> bool:
    """
    Report whether a (preQC, postQC) cache pair already exists.

    Parameters
    ----------
    paths : tuple[str, str]
        Cache paths.

    Returns
    -------
    bool
        True if both files exist.
    """
    if all(os.path.exists(p) for p in paths):
        print(f"  [CACHED] {paths[0]}\n  [CACHED] {paths[1]}")
        return True
    return False


def main(args: argparse.Namespace) -> None:
    """
    Downsample, QC and cache Kamath and Siletti.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    None
    """
    rng = np.random.default_rng(_SEED)

    print("\n=== Kamath 2022 (DA-first stratified draw) ===")
    kamath_paths = kamath_cache_paths(args.cells_per_source, args.kamath_da_per_subtype, args.kamath_non_da_min)
    if not caches_exist(kamath_paths):
        kamath_cells = scan_kamath_obs(args.da_subtype_column)
        kamath_sel = plan_kamath_selection(
            kamath_cells, args.cells_per_source, args.kamath_da_per_subtype, args.kamath_non_da_min, rng,
        )
        da_counts = kamath_sel[kamath_sel["is_da_file"]]["label"].value_counts()
        print("  DA cells per subtype:\n" + da_counts.to_string())
        if args.execute:
            kamath = load_kamath(kamath_sel, args.da_subtype_column)
            kamath_postqc = apply_qc_filters(kamath, "Kamath", doublet_batch_key="donor_id")
            write_cache_pair(kamath, kamath_postqc, kamath_paths)
        else:
            print(f"  [DRY-RUN] would write {kamath_paths[0]}\n  [DRY-RUN] would write {kamath_paths[1]}")

    print("\n=== Siletti 2023 (all DA VGLUT2 + random top-up) ===")
    siletti_paths = siletti_cache_paths(args.cells_per_source)
    if not caches_exist(siletti_paths):
        siletti_cells = scan_siletti_obs()
        allen = load_allen_annotations(siletti_cells.index)
        siletti_sel = plan_siletti_selection(siletti_cells, allen, args.cells_per_source, rng)
        if args.execute:
            siletti = load_siletti(siletti_sel, allen)
            siletti_postqc = apply_qc_filters(siletti, "Siletti", doublet_batch_key="sample_id")
            n_da_pre, n_da_post = int(siletti.obs["is_da"].sum()), int(siletti_postqc.obs["is_da"].sum())
            print(f"  DA VGLUT2 cells: {n_da_pre:,} preQC -> {n_da_post:,} postQC")
            write_cache_pair(siletti, siletti_postqc, siletti_paths)
        else:
            print(f"  [DRY-RUN] would write {siletti_paths[0]}\n  [DRY-RUN] would write {siletti_paths[1]}")

    if args.execute:
        print("\nDone. Run notebooks/03_integrate_kamath_siletti.ipynb next, with the same downsampling values.")
    else:
        print("\nDry-run only. Re-run with --execute to load cells, run QC and write the caches.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "-x",
        "--execute",
        action="store_true",
        help="Load cells, run QC and write the caches. Without it, only print the plan.\n"
             "[Default: False]",
    )
    parser.add_argument(
        "-n",
        "--cells-per-source",
        type=int,
        default=DEFAULT_CELLS_PER_SOURCE,
        help="Cells drawn per source, before QC.\n"
             f"[Default: {DEFAULT_CELLS_PER_SOURCE}]",
    )
    parser.add_argument(
        "-d",
        "--kamath-da-per-subtype",
        type=int,
        default=DEFAULT_KAMATH_DA_PER_SUBTYPE,
        help="Kamath cells drawn per DA subtype (10 subtypes).\n"
             f"[Default: {DEFAULT_KAMATH_DA_PER_SUBTYPE}]",
    )
    parser.add_argument(
        "-m",
        "--kamath-non-da-min",
        type=int,
        default=DEFAULT_KAMATH_NON_DA_MIN,
        help="Minimum Kamath cells per non-DA author_cell_type, before the\n"
             "random top-up.\n"
             f"[Default: {DEFAULT_KAMATH_NON_DA_MIN}]",
    )
    parser.add_argument(
        "-c",
        "--da-subtype-column",
        default="author_cell_type",
        help="Kamath obs column with fine-grained labels (incl. DA subtypes).\n"
             "[Default: author_cell_type]",
    )
    args = parser.parse_args()

    chdir_to_repo_root()
    log_run_command(args, log_dir=DATA_DIR)
    main(args)
