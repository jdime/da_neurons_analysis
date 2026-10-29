"""
Shared constants and helpers for scripts/01_download_kamath_siletti.py,
scripts/02_downsample_kamath_siletti.py and notebooks/03_integrate_kamath_siletti.ipynb.

Adapted from spatial-sc-rnaseq-integration-wf's
reference_building/da_neurons_identification_analysis/utils/_reference_building_utils.py,
restricted to the Kamath 2022 and Siletti 2023 sources.
"""

import os
import re
import sys
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import anndata as ad
from scipy.sparse import csr_matrix, issparse

# ---------------------------------------------------------------------------
# Paths (relative to the repo root; every script chdirs there first)
# ---------------------------------------------------------------------------

DATA_DIR = "data"
KAMATH_DIR = os.path.join(DATA_DIR, "kamath")
SILETTI_DIR = os.path.join(DATA_DIR, "siletti")
ALLEN_DIR = os.path.join(DATA_DIR, "allen-brain-cell-atlas")
OUTPUT_DIR = "outputs"

# ---------------------------------------------------------------------------
# CELLxGENE sources
# ---------------------------------------------------------------------------

CELLXGENE_API = "https://api.cellxgene.cziscience.com/curation/v1"
KAMATH_COLLECTION_ID = "b0f0b447-ac37-45b0-b1bf-5c0b7d871120"
SILETTI_COLLECTION_ID = "283d65eb-dd53-496d-adb7-7570c7caa443"

# The 7 Kamath cell-type files. The 2 Nurr-sorted files are skipped: their
# cells already appear in these 7 (barcode overlap, see the original repo's
# 02_kamath_overlap.py).
KAMATH_DA_FILE_STEM = "human_da_neurons_10x_scrna_seq"
KAMATH_FILE_STEMS = (
    KAMATH_DA_FILE_STEM,
    "human_non_da_neurons_10x_scrna_seq",
    "human_astrocytes_10x_scrna_seq",
    "human_microglia_10x_scrna_seq",
    "human_oligodendrocytes_10x_scrna_seq",
    "human_opc_cells_10x_scrna_seq",
    "human_endothelial_cells_10x_scrna_seq",
)

# The 3 Siletti midbrain dissections carrying ~81% of the atlas' Allen-annotated
# "DA VGLUT2" cells. Region label -> file stem.
SILETTI_REGION_FILE_STEMS = {
    "SN-RN": "dissection_midbrain_m_substantia_nigra_red_nucleus_and_nearby_nuclei_sn_rn",
    "SN": "dissection_midbrain_m_substantia_nigra_sn",
    "RN": "dissection_midbrain_rn_red_nucleus_rn",
}

# ---------------------------------------------------------------------------
# Allen Brain Cell Atlas (only source of Siletti's DA labels; public S3)
# ---------------------------------------------------------------------------

ALLEN_S3_URL = "https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com"
ALLEN_CELL_METADATA_KEY = "metadata/WHB-10Xv3/20241115/cell_metadata.csv"
ALLEN_MEMBERSHIP_KEY = "metadata/WHB-taxonomy/20240330/cluster_to_cluster_annotation_membership.csv"
# Stored gzipped: 01_download_kamath_siletti.py compresses each CSV after download
# (cell_metadata.csv: 830 MB -> ~217 MB).
ALLEN_CELL_METADATA_PATH = os.path.join(ALLEN_DIR, ALLEN_CELL_METADATA_KEY + ".gz")
ALLEN_MEMBERSHIP_PATH = os.path.join(ALLEN_DIR, ALLEN_MEMBERSHIP_KEY + ".gz")
ALLEN_DA_NEUROTRANSMITTER = "DA VGLUT2"

# ---------------------------------------------------------------------------
# Biology
# ---------------------------------------------------------------------------

# Kamath et al. (Nat Neurosci 2022) DA marker gene symbols.
KAMATH_DA_MARKER_SYMBOLS = (
    # Pan-DA identity
    "TH", "SLC6A3", "SLC18A2", "DDC",
    # Branch markers (the two main DA subdivisions)
    "SOX6", "CALB1",
    # Subtype-defining markers (e.g. SOX6_AGTR1, CALB1_GEM)
    "AGTR1", "DDT", "GFRA2", "PART1", "CALCR", "CRYM", "CCDC68", "GEM", "PPP1R17", "RBP4", "TRHR",
)

# ---------------------------------------------------------------------------
# Downsampling defaults (shared by 02 to downsample and 03 to find the caches)
# ---------------------------------------------------------------------------

DEFAULT_CELLS_PER_SOURCE = 10_000
DEFAULT_KAMATH_DA_PER_SUBTYPE = 300
DEFAULT_KAMATH_NON_DA_MIN = 50


# ---------------------------------------------------------------------------
# Cache paths
# ---------------------------------------------------------------------------

def kamath_cache_paths(n_total: int, da_per_subtype: int, non_da_min: int) -> tuple[str, str]:
    """
    Return the (preQC, postQC) cache paths for a Kamath downsampling.

    Parameters
    ----------
    n_total : int
        Total cells drawn before QC.
    da_per_subtype : int
        Cells drawn per DA subtype.
    non_da_min : int
        Minimum cells drawn per non-DA cell type.

    Returns
    -------
    tuple[str, str]
        Paths to the preQC and postQC H5ADs.
    """
    stem = f"kamath_n{n_total}_da{da_per_subtype}_nonda_min{non_da_min}"
    d = os.path.join(KAMATH_DIR, "intermediate")
    return os.path.join(d, f"{stem}_preQC.h5ad"), os.path.join(d, f"{stem}_postQC.h5ad")


def siletti_cache_paths(n_total: int) -> tuple[str, str]:
    """
    Return the (preQC, postQC) cache paths for a Siletti downsampling.

    Parameters
    ----------
    n_total : int
        Total cells drawn before QC (all DA cells included).

    Returns
    -------
    tuple[str, str]
        Paths to the preQC and postQC H5ADs.
    """
    stem = f"siletti_midbrain_n{n_total}_all_da"
    d = os.path.join(SILETTI_DIR, "intermediate")
    return os.path.join(d, f"{stem}_preQC.h5ad"), os.path.join(d, f"{stem}_postQC.h5ad")


# ---------------------------------------------------------------------------
# Gene helpers (integration notebook)
# ---------------------------------------------------------------------------

def marker_ensembl_ids(adata: ad.AnnData, symbols: tuple[str, ...]) -> list[str]:
    """
    Resolve gene symbols to the Ensembl IDs used as `var_names`.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData with symbols in `var["feature_name"]`.
    symbols : tuple[str, ...]
        Gene symbols.

    Returns
    -------
    list[str]
        Ensembl IDs found (symbols not in the panel are reported and skipped).
    """
    found = adata.var.index[adata.var["feature_name"].isin(symbols)].tolist()
    missing = sorted(set(symbols) - set(adata.var["feature_name"]))
    if missing:
        print(f"  Markers not in the gene panel: {missing}")
    return found


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def normalize_label(label: str) -> str:
    """
    Normalise a CELLxGENE dataset title into a safe filename stem.

    Parameters
    ----------
    label : str
        Dataset title, e.g. "Human DA Neurons 10x scRNA-seq".

    Returns
    -------
    str
        Lower-case, underscore-separated stem, e.g. "human_da_neurons_10x_scrna_seq".
    """
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def chdir_to_repo_root() -> Path:
    """
    Change the working directory to the repo root so relative data paths resolve.

    Returns
    -------
    pathlib.Path
        The repo root.
    """
    repo_root = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
    if Path.cwd() != repo_root:
        print(f"[INFO] Changing working directory to repo root: {repo_root}")
        os.chdir(repo_root)
    return repo_root


def extract_raw_counts(adata, source_name: str = "") -> csr_matrix:
    """
    Extract the raw integer count matrix as sparse int32.

    Checks `layers["counts"]`, then `adata.raw.X`, then `adata.X`, and
    validates the matrix is integer-valued (whole-number floats pass).

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData to extract counts from.
    source_name : str, optional
        Label for error/info messages.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse int32 count matrix (cells x genes).

    Raises
    ------
    ValueError
        If the matrix contains non-integer values.
    """
    if "counts" in adata.layers:
        mat = adata.layers["counts"]
    elif adata.raw is not None:
        mat = adata.raw.X
    else:
        mat = adata.X

    mat = csr_matrix(mat) if not issparse(mat) else mat.tocsr()
    if not np.issubdtype(mat.dtype, np.integer) and mat.data.size > 0:
        non_int = mat.data % 1 != 0
        if non_int.any():
            prefix = f"{source_name}: " if source_name else ""
            raise ValueError(
                f"{prefix}count matrix contains non-integer values "
                f"(e.g. {mat.data[non_int][:5].tolist()}). Raw counts must be integer-valued."
            )
    return mat.astype(np.int32) if mat.dtype != np.int32 else mat


# ---------------------------------------------------------------------------
# Run logging
# ---------------------------------------------------------------------------

class _Tee:
    """Mirror writes to both the original stream and a log file."""

    def __init__(self, original, log_file) -> None:
        self._original = original
        self._log = log_file

    def write(self, data: str) -> int:
        self._original.write(data)
        self._log.write(data)
        return len(data)

    def flush(self) -> None:
        self._original.flush()
        self._log.flush()

    def __getattr__(self, name):
        return getattr(self._original, name)


def log_run_command(args: argparse.Namespace, log_dir: str) -> None:
    """
    Log the invocation and tee stdout/stderr to `{log_dir}/{script_stem}.log`.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments; `args._log_file_path` is set to the log path.
    log_dir : str
        Directory for the log file.

    Returns
    -------
    None
    """
    log_file = Path(log_dir) / f"{Path(sys.argv[0]).stem}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    f = open(log_file, "w", encoding="utf-8")  # noqa: SIM115 — intentionally kept open
    f.write(f"[{timestamp}] {' '.join(sys.argv)}\n\n")
    sys.stdout = _Tee(sys.stdout, f)
    sys.stderr = _Tee(sys.stderr, f)
    args._log_file_path = str(log_file)
