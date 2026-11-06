# Kaggle version

Same analysis as the parent folder, using Kaggle's input and output paths.

| File | Contents |
|---|---|
| `brca-subtypes-survival.ipynb` | The notebook, already executed, with all 5 figures inline. Upload this. |
| `brca-subtypes-survival.py` | The same code as a plain script. |

## Setting it up

The data files are large (65 MB), so they are **not** duplicated here. Two options:

**A. Let the notebook download them (simplest).** Turn *Internet* on in the notebook settings and Run All. It fetches all three files from UCSC Xena on first run.

**B. Attach them as a dataset (works with internet off).** Upload the three files from `../data/` — `HiSeqV2.gz`, `BRCA_clinicalMatrix`, `BRCA_survival.txt` — as a dataset titled **`tcga-brca-xena`**, then *+ Add Input*. Any other title still works; the notebook searches `/kaggle/input/` for the filenames.

Runtime is roughly 15-20 minutes on a CPU session. No GPU needed.

## What it needs

`lifelines` is installed by the first cell if absent; everything else (pandas, scikit-learn, scipy, matplotlib) is in the Kaggle image. The pathway-enrichment step calls the Enrichr web API and is skipped gracefully if internet is off — the rest of the analysis is unaffected.

## Output

Figures and tables go to `/kaggle/working/figures/` and `/kaggle/working/results/`, listed by the last cell and available under the *Output* tab.

## Licence and citation

TCGA data are open access under the NIH Genomic Data Sharing Policy. Cite UCSC Xena (Goldman et al., Nat Biotechnol 2020), the TCGA BRCA marker paper (Nature 2012) and Liu et al. (Cell 2018) for the survival endpoints. Full references are in `../data/SOURCE.md`.
