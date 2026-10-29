# da_neurons_analysis

Integration of two public single-nucleus RNA-seq datasets of the human midbrain with
scanpy, focused on the dopaminergic (DA) neurons lost in Parkinson's disease (PD), followed
by a PD vs Control differential expression (DE) analysis of those neurons. It is small
enough (~10k cells per source by default) to run on a laptop, following `scanpy`'s
`BBKNN` [tutorial](https://scanpy.readthedocs.io/en/stable/tutorials/basics/integrating-data-using-ingest.html).

| Source | What | Donors |
|---|---|---|
| [Kamath et al. 2022](https://pubmed.ncbi.nlm.nih.gov/35513515/) (*Nat Neurosci*) | Substantia nigra; 22,048 DA neurons in 10 subtypes (e.g. the vulnerable `SOX6_AGTR1`) + other cell types | PD, Control and other neurodegeneration: used for the DE |
| [Siletti et al. 2023](https://pubmed.ncbi.nlm.nih.gov/37824663/) (*Science*) | Whole-brain atlas; only the 3 midbrain dissections (SN-RN, SN, RN) | Control only |

Siletti's CELLxGENE deposition has no dopaminergic label. DA neurons are identified
with the [Allen Brain Cell Atlas](https://alleninstitute.github.io/abc_atlas_access/descriptions/WHB-10Xv3.html)
annotation of the same cells (`DA VGLUT2`).


## Workflow

| Step | File | What it does | Outputs |
|---|---|---|---|
| 1 | `scripts/01_download_kamath_siletti.py` | Downloads only the needed files (~4.9 GB): 7 Kamath cell-type H5ADs, 3 Siletti midbrain dissections, 2 Allen metadata CSVs (stored gzipped) | `data/{kamath,siletti,allen-brain-cell-atlas}/` |
| 2 | `scripts/02_downsample_kamath_siletti.py` | Downsamples to 10k cells per source, then QC (mt%, scrublet, counts, genes) | `data/{kamath,siletti}/intermediate/*_{preQC,postQC}.h5ad` |
| 3 | `notebooks/03_integrate_kamath_siletti.ipynb` | Uncorrected vs `BBKNN` integration (UMAPs), then PD vs Control DE in Kamath DA neurons (tables, dotplot, per-donor heatmap, interactive volcano plot) | None; all results are shown inline |
| — | `utils/_da_neurons_utils.py` | Shared paths, constants and helpers | — |

### Downsampling (step 2)

- **Kamath (10k):** 300 cells from each of the 10 DA subtypes (3,000 DA), then at least
  50 cells per non-DA `author_cell_type` (58 types), topped up at random to 10k.
- **Siletti (10k):** every Allen `DA VGLUT2` cell in the 3 midbrain dissections (~810),
  topped up at random to 10k.

### Integration and differential expression (step 3)

1. **Uncorrected:** PCA → neighbours → UMAP on the merged data, showing the batch effect.
2. **`BBKNN`:** batch-balanced neighbours across sources → UMAP.
3. **PD vs Control DE:** Wilcoxon rank-sum test (`sc.tl.rank_genes_groups`) on
   log-normalised expression of **Kamath DA neurons only** (Siletti has no PD donors, so
   adding its controls would confound condition with source). All shared genes detected in
   ≥10% of either group's cells are tested. Because cells from one donor are not
   independent, a donor-level check (Mann-Whitney on per-donor mean expression, donors
   with ≥10 DA cells) is reported next to the cell-level statistics.

### Results with default settings

On a 4-core Intel MacBook Pro, step 2 takes ~1 min and the step 3 notebook ~2 min.

| | Kamath | Siletti |
|---|---|---|
| Cells after QC | 9,959 (2,976 DA) | 9,959 (789 Allen DA VGLUT2) |

- **Uncorrected:** cells separate by source; **`BBKNN`** mixes the sources, with Kamath and
  Siletti DA neurons in one cluster (`TH`, `SLC6A3` high; `AGTR1` marking the
  `SOX6_AGTR1` corner).
- **DE:** 475 PD DA neurons (6 donors) vs 1,961 Control (9 donors), 12,825 genes tested.
  Down in PD and consistent across donors: `ALDH1A1`, `SLC6A3`, `SLC18A2`, `TIAM2`,
  `SEMA5A`, in line with the loss of the vulnerable `SOX6`/`ALDH1A1` DA neurons. Many of
  the top genes up in PD (e.g. `RUNX1`, `SNX31`, `TTR`) are not supported at the donor
  level.
- **DE caveats:** ~8,200 genes pass cell-level FDR < 0.05, an inflation from treating cells
  as independent. One PD donor contributes 358/475 PD cells, and only 3 PD donors have ≥10
  DA cells. A pseudobulk test (e.g. `pydeseq2`) would be the rigorous follow-up.

## How to run

Scripts 1 and 2 are **dry-run by default** (they print what they would do); add `--execute`
(`-x`) to run them. Run from anywhere; each switches to the repo root.

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python3 scripts/01_download_kamath_siletti.py --execute
python3 scripts/02_downsample_kamath_siletti.py --execute
jupyter lab notebooks/03_integrate_kamath_siletti.ipynb
```

Step 2's downsampling options (`--cells-per-source`, `--kamath-da-per-subtype`,
`--kamath-non-da-min`) must match the notebook's Parameters cell, so it finds step 2's
caches. Use `--help` on either script for all options.

If the Plotly volcano plot shows a blank output cell (common in VS Code), set a renderer
before plotting, e.g. `plotly.io.renderers.default = "notebook_connected"`.

## Known issue: "spiral" `BBKNN` UMAPs

If `annoy` is compiled locally with `-ffast-math` by a recent clang (e.g. no prebuilt
wheel for your Python), it returns wrong neighbours **without any error**, and `BBKNN`
UMAPs turn into long spirals instead of clusters. The notebook therefore uses
`BBKNN_COMPUTATION = "cKDTree"` (exact, and fast at this size). To repair `annoy` itself:

```bash
ANNOY_COMPILER_ARGS="-D_CRT_SECURE_NO_WARNINGS,-fpermissive,-O3,-DANNOYLIB_MULTITHREADED_BUILD,-std=c++14,-mmacosx-version-min=10.12" \
  pip install --force-reinstall --no-deps --no-binary annoy --no-cache-dir annoy==1.17.3
```

## Contributing

Make changes on a branch and open a pull request into `main`. Fill in the PR template
([`.github/pull_request_template.md`](.github/pull_request_template.md)), which GitHub
loads into the description automatically.

## Acknowledgements

This repository is inspired in code developed in
[ASAP-CRN/spatial-sc-rnaseq-integration-wf](https://github.com/ASAP-CRN/spatial-sc-rnaseq-integration-wf)
(`reference_building/da_neurons_identification_analysis/`), adapted here to the Kamath and
Siletti sources only, and replacing `scVI`/`scANVI` integration with scanpy's `BBKNN`.

## Authorship

Javier Diaz ([@jdime](https://github.com/jdime))

## License

MIT; see [LICENSE](LICENSE).
