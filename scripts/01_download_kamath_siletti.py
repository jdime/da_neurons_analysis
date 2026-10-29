#!/usr/bin/env python3
"""
Download the raw Kamath 2022 and Siletti 2023 H5ADs from CELLxGENE, plus the
two Allen Brain Cell Atlas metadata files that label Siletti's DA neurons.

Only the files used downstream are downloaded (~4.9 GB in total):

Kamath et al. 2022, Nat Neurosci (~3.2 GB)
  The 7 cell-type files from
  https://cellxgene.cziscience.com/collections/b0f0b447-ac37-45b0-b1bf-5c0b7d871120
  The 2 Nurr-sorted files are skipped (their cells are already in the other 7).

Siletti et al. 2023, Science (~0.8 GB)
  The 3 midbrain dissections (SN-RN, SN, RN) from
  https://cellxgene.cziscience.com/collections/283d65eb-dd53-496d-adb7-7570c7caa443

Allen Brain Cell Atlas (~0.83 GB download, ~0.22 GB gzipped, public S3)
  WHB-10Xv3 cell_metadata.csv and WHB-taxonomy cluster membership. Siletti's
  CELLxGENE deposition has no dopaminergic label; Allen's "DA VGLUT2"
  neurotransmitter annotation provides it. Each CSV is gzipped after download
  (the uncompressed copy is removed).

Files land under data/{kamath,siletti,allen-brain-cell-atlas}/. Files already
present are skipped.

Dry-run by default: prints what would be downloaded. Pass --execute to download.

Status tags
  [CACHED]    file already present, skipped
  [DRY-RUN]   file would be downloaded
  [DOWNLOAD]  file is being downloaded
  [GZIP]      downloaded CSV is being compressed
"""

import os
import sys
import gzip
import shutil
import argparse
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
from _da_neurons_utils import (
    CELLXGENE_API,
    KAMATH_COLLECTION_ID,
    SILETTI_COLLECTION_ID,
    KAMATH_DIR,
    SILETTI_DIR,
    KAMATH_FILE_STEMS,
    SILETTI_REGION_FILE_STEMS,
    ALLEN_S3_URL,
    ALLEN_CELL_METADATA_KEY,
    ALLEN_MEMBERSHIP_KEY,
    ALLEN_CELL_METADATA_PATH,
    ALLEN_MEMBERSHIP_PATH,
    DATA_DIR,
    normalize_label,
    chdir_to_repo_root,
    log_run_command,
)


def fetch_collection_h5ad_assets(collection_id: str) -> dict[str, dict]:
    """
    Query the CELLxGENE curation API for a collection's H5AD assets.

    Parameters
    ----------
    collection_id : str
        CELLxGENE collection ID.

    Returns
    -------
    dict[str, dict]
        Keyed by normalised dataset title; values have "title", "url" and
        "filesize" (bytes).
    """
    resp = requests.get(f"{CELLXGENE_API}/collections/{collection_id}", timeout=30)
    resp.raise_for_status()
    assets = {}
    for dataset in resp.json().get("datasets", []):
        title = dataset.get("title", dataset.get("dataset_id", "unknown"))
        # The API has used both 'assets' and 'dataset_assets' across versions.
        for asset in dataset.get("assets", dataset.get("dataset_assets", [])):
            if asset.get("filetype", "").upper() == "H5AD":
                assets[normalize_label(title)] = {
                    "title": title,
                    "url": asset.get("url", asset.get("download_url", "")),
                    "filesize": asset.get("filesize", 0),
                }
    return assets


def download_file(url: str, dest_path: str, label: str, size_bytes: int, execute: bool) -> None:
    """
    Stream a file to disk, skipping it if already present.

    Parameters
    ----------
    url : str
        Source URL.
    dest_path : str
        Destination path.
    label : str
        Human-readable label for log lines.
    size_bytes : int
        Expected size in bytes (0 if unknown), for log lines only.
    execute : bool
        If False, only report what would be downloaded.

    Returns
    -------
    None
    """
    size_txt = f" ({size_bytes / 1e6:,.0f} MB)" if size_bytes else ""
    if os.path.exists(dest_path):
        print(f"  [CACHED]   {label} ({os.path.getsize(dest_path) / 1e6:,.0f} MB) → {dest_path}")
        return
    if not execute:
        print(f"  [DRY-RUN]  {label}{size_txt} → {dest_path}")
        return

    print(f"  [DOWNLOAD] {label}{size_txt} → {dest_path}")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    tmp_path = dest_path + ".part"
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(tmp_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                fh.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"\r    {downloaded / 1e6:,.0f} / {total / 1e6:,.0f} MB "
                          f"({downloaded / total * 100:.0f}%)", end="", flush=True)
    os.replace(tmp_path, dest_path)  # only a complete download gets the final name
    print(f"\n    Done: {downloaded / 1e6:,.0f} MB")


def gzip_file(src_path: str, dest_path: str) -> None:
    """
    Gzip a file, then remove the uncompressed original.

    Parameters
    ----------
    src_path : str
        File to compress.
    dest_path : str
        Destination ".gz" path.

    Returns
    -------
    None
    """
    print(f"  [GZIP]     {src_path} → {dest_path}")
    tmp_path = dest_path + ".part"
    with open(src_path, "rb") as src, gzip.open(tmp_path, "wb") as dest:
        shutil.copyfileobj(src, dest, length=8 * 1024 * 1024)
    os.replace(tmp_path, dest_path)  # only a complete archive gets the final name
    os.remove(src_path)
    print(f"    Done: {os.path.getsize(dest_path) / 1e6:,.0f} MB")


def download_and_gzip(url: str, gz_path: str, label: str, execute: bool) -> None:
    """
    Download a CSV and store it gzipped, skipping it if the archive exists.

    Parameters
    ----------
    url : str
        Source URL of the uncompressed file.
    gz_path : str
        Destination ".gz" path; the download lands next to it without ".gz".
    label : str
        Human-readable label for log lines.
    execute : bool
        If False, only report what would be done.

    Returns
    -------
    None
    """
    if os.path.exists(gz_path):
        print(f"  [CACHED]   {label} ({os.path.getsize(gz_path) / 1e6:,.0f} MB) → {gz_path}")
        return
    if not execute:
        print(f"  [DRY-RUN]  {label} → {gz_path} (download, then gzip)")
        return
    csv_path = gz_path.removesuffix(".gz")
    download_file(url, csv_path, label, 0, execute)  # reuses a leftover uncompressed copy
    gzip_file(csv_path, gz_path)


def download_collection_files(
    collection_id: str,
    file_stems: list[str],
    dest_dir: str,
    execute: bool,
) -> None:
    """
    Download the named H5ADs of a CELLxGENE collection.

    Parameters
    ----------
    collection_id : str
        CELLxGENE collection ID.
    file_stems : list[str]
        Normalised dataset titles to download (see `normalize_label`).
    dest_dir : str
        Directory to save `{stem}.h5ad` files into.
    execute : bool
        If False, only report what would be downloaded.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If a requested dataset is not in the collection.
    """
    dest_paths = {stem: os.path.join(dest_dir, f"{stem}.h5ad") for stem in file_stems}
    if all(os.path.exists(p) for p in dest_paths.values()):
        for stem, path in dest_paths.items():
            download_file("", path, stem, 0, execute)
        return

    assets = fetch_collection_h5ad_assets(collection_id)
    missing = set(file_stems) - assets.keys()
    if missing:
        raise RuntimeError(
            f"{len(missing)} dataset(s) not found in collection {collection_id}: {sorted(missing)}. "
            f"CELLxGENE titles may have changed. Available: {sorted(assets)}"
        )
    for stem, path in dest_paths.items():
        asset = assets[stem]
        download_file(asset["url"], path, asset["title"], asset["filesize"], execute)


def main(args: argparse.Namespace) -> None:
    """
    Download Kamath, Siletti and Allen files.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    None
    """
    print("\n=== Kamath 2022: 7 cell-type H5ADs ===")
    download_collection_files(KAMATH_COLLECTION_ID, list(KAMATH_FILE_STEMS), KAMATH_DIR, args.execute)

    print("\n=== Siletti 2023: 3 midbrain dissection H5ADs ===")
    download_collection_files(
        SILETTI_COLLECTION_ID, list(SILETTI_REGION_FILE_STEMS.values()), SILETTI_DIR, args.execute,
    )

    print("\n=== Allen Brain Cell Atlas metadata (Siletti DA labels) ===")
    for key, path in (
        (ALLEN_CELL_METADATA_KEY, ALLEN_CELL_METADATA_PATH),
        (ALLEN_MEMBERSHIP_KEY, ALLEN_MEMBERSHIP_PATH),
    ):
        download_and_gzip(f"{ALLEN_S3_URL}/{key}", path, Path(key).name, args.execute)

    if args.execute:
        print("\nDone. Run scripts/02_downsample_kamath_siletti.py next.")
    else:
        print("\nDry-run only. Re-run with --execute to download.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "-x",
        "--execute",
        action="store_true",
        help="Download the files. Without it, only print what would be downloaded.\n"
             "[Default: False]",
    )
    args = parser.parse_args()

    chdir_to_repo_root()
    log_run_command(args, log_dir=DATA_DIR)
    main(args)
