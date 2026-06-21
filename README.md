# Generative embedding extends tabular foundation models to dengue early warning from sparse surveillance data

The repository reproduces the full featurisation pipeline and the headline raw-versus-enriched comparison reported in the manuscript from a single weekly dengue and rainfall dataset and a small set of Python scripts. One command expands two raw weekly columns into roughly 105 latent features and passes them to a tabular foundation model.

## Background

Routine dengue surveillance returns very little structured signal each week. In practice a surveillance unit holds two usable weekly columns, notified case counts and rainfall. That is too sparse for a tabular foundation model to perform well on directly. The two columns nonetheless carry latent structure: annual seasonality, monsoon and ENSO low-frequency climate variation, rainfall accumulation and threshold effects, and short-term transmission acceleration. This pipeline makes that structure explicit. It expands the two raw columns into about 105 domain-grounded, leakage-safe features, the step we describe as generative embedding, and then passes the embedded representation to TabPFN to produce a weekly early-warning probability.

The reported result is the lift obtained by moving from the raw two-column input to the embedded feature set, not the absolute score. Every run records three rungs of an enrichment ladder, raw two-column to case-only to enriched, so the contribution of the embedding is read directly rather than inferred. The pipeline is evaluated at three nested geographic scales, Quezon City, the 17 Philippine administrative regions, and eight dengue-endemic countries, under a leave-one-year-out protocol in which the imputer, the rank-Gaussian transform, and the classifier are all fitted on the training years alone.

The early-warning target is peak-anchored and therefore case-derived, which places a ceiling on achievable discrimination at an AUROC of roughly 0.85. The work is presented as an internal-validity demonstration of the featurisation method rather than a claim of operational forecast skill.

## Authors

The author block below is inherited from the companion manuscript by the same group and is provided as a starting roster. It should be revised to list the contributors to the present manuscript in accordance with ICMJE authorship criteria before submission.

Keanu John Pelitro<sup>1</sup>, Julia Fye Manzano<sup>1</sup>, Troy Owen Matavia<sup>1</sup>, Kylone Soriano<sup>1</sup>, Klara Bilbao<sup>1</sup>, Gereka Marie Garcia<sup>1</sup>, Aira Joy Delos Angeles<sup>1</sup>, Alfredo Mahar Lagmay<sup>1,3</sup>, and DJ Darwin Bandoy<sup>1,2,*</sup>

<sup>1</sup> University of the Philippines Resilience Institute, Quezon City, Philippines.
<sup>2</sup> College of Veterinary Medicine, University of the Philippines, Los Baños, Laguna, Philippines.
<sup>3</sup> National Institute of Geological Sciences, University of the Philippines Diliman, Quezon City, Philippines.

<sup>*</sup> Corresponding author: drbandoy@up.edu.ph

## Repository structure

```
SparseFE/
├── README.md
├── AGENTS.md                                guidance for an automated agent
├── requirements.txt
├── LICENSE                                  MIT
├── CITATION.cff
│
├── data/
│   └── Dengue-Rainfall_Dataset.xlsx         deidentified weekly data
│
├── code/
│   ├── labels.py                            peak-anchored early-warning target
│   ├── enrich.py                            the method: 2 columns to ~105 causal latent features
│   ├── evaluate.py                          TabPFN-or-stand-in, leave-one-year-out CV
│   ├── run.py                               one-command entry point
│   └── make_figures.py                      raw-versus-enriched comparison charts
│
└── outputs/                                 created on first run
```

The pipeline is self-contained. The dataset ships inside the folder, the featurisation is in plain code, and one command produces the headline comparison, so there is no hidden state to reconstruct. `AGENTS.md` holds the operating rules for an automated agent run inside the repository and restates the causality and reporting constraints described below.

## Data

The complete weekly dataset is a single Excel workbook deposited on Zenodo:

*Dengue-Rainfall_Dataset.xlsx.* Zenodo. https://doi.org/10.5281/zenodo.19448854

The workbook contains five sheets. Two metadata sheets (Dataset Summary and Data Dictionary) describe provenance and column definitions. The three analytical sheets each correspond to one geographic scale of the analysis.

| Sheet           | Records | Period    | Case column used   | Rainfall column used |
|-----------------|---------|-----------|--------------------|----------------------|
| `QC Data`       | 832     | 2010–2025 | `DC_QC`            | `RF_PAGASA`          |
| `Regional Data` | 7 072   | 2016–2025 | `DC_DOH`           | `RF_NASA`            |
| `Country Data`  | 3 216   | 2016–2025 | `DC_OPENDENGUE`    | `RF_HDX`             |

`YR` is the ISO calendar year and `WN` is the ISO epidemiological week number; each observation is indexed to the Monday of the corresponding ISO week. The `DC_*` columns are the weekly suspected dengue case counts at the corresponding scale, and the `RF_*` columns are the matched weekly rainfall series. `DC_QC` is from the Quezon City Epidemiology and Surveillance Division. `DC_DOH` is from PIDSR-reported regional submissions archived at the Humanitarian Data Exchange (https://data.humdata.org). `DC_OPENDENGUE` is from the OpenDengue repository (https://opendengue.org).

This pipeline consumes both a case column and a rainfall column at each scale, which is the difference from threshold-based detectors that use cases alone. The data-quality flags (`FLAG_COVID`, `FLAG_PLAUSIBILITY`, `FLAG_SINGLE_CELL_RF`, `FLAG_DEKADAL_APPROX`, `FLAG_TERMINAL_GAP`) are read to drop pandemic and terminal-gap years, so the year exclusions come from the data itself rather than a hardcoded list. All flag and auxiliary columns are documented in full in the `Data Dictionary` sheet of the workbook.

Cases were classified as suspected dengue under the Philippine Integrated Disease Surveillance and Response (PIDSR) Programme, consistent with the WHO 2009 dengue case definition adopted by PIDSR in 2011. Laboratory confirmation was not required, consistent with passive surveillance practice throughout the study period. All surveillance data are aggregated weekly totals containing no individual-level identifiers.

## Reproducing the analysis

### Requirements

Python 3.10 or newer under Linux, macOS, or Windows. The core dependency set is `pandas`, `numpy`, `scipy`, `scikit-learn`, `openpyxl`, and `matplotlib`, with `tabpfn` for the foundation-model backend. Install everything with:

```bash
pip install -r requirements.txt
```

Two dependencies are optional and the pipeline degrades gracefully when they are absent. `pycatch22` supplies the canonical 22 hctsa time-series features (columns `c22_*`); without it the pipeline falls back to a labelled numpy subset (columns `c22fb_*`). `timesfm` supplies the comparator bar in the first figure; without it the comparator falls back to a labelled classical trend forecaster (`ts_trend_baseline`). Wheels for both can be unavailable on very new Python releases, which is why they are kept optional.

### Backend selection

The pipeline runs out of the box on an offline stand-in model so the plumbing can be verified without an account. The stand-in is labelled `standin_histgb` in every output and its scores are not foundation-model numbers.

```bash
# offline stand-in, no account required
#   Windows PowerShell:
$env:AUTOTABPFN_BACKEND="standin"; py code/run.py --scale qc
#   macOS or Linux:
AUTOTABPFN_BACKEND=standin python code/run.py --scale qc
```

To produce the published numbers, install `tabpfn`, obtain a free token at https://ux.priorlabs.ai (register, open the License tab, accept, copy the token), set it once, and drop the backend override. The backend then auto-detects `tabpfn`.

```bash
# Windows: setx TABPFN_TOKEN "your-token"   then reopen the terminal
py code/run.py --scale qc
```

If the interactive token login fails on Windows because of the browser-authentication bug, set `TABPFN_TOKEN` directly from the License tab at https://ux.priorlabs.ai, or run with the stand-in override above.

### Run order

```bash
pip install -r requirements.txt

py code/run.py --scale qc           # Quezon City
py code/run.py --scale regional     # 17 Philippine regions
py code/run.py --scale country      # 8 endemic countries

py code/make_figures.py             # comparison figures, after one or more scales
```

Each scale writes its own subdirectory under `outputs/`, created on first run. `make_figures.py` reads whichever scales have been run and writes the figure set; the regional climate-gradient figure requires the regional run.

### Runtime

A single scale completes in a few minutes on a recent laptop under the stand-in backend. Wall-clock time under the real `tabpfn` backend depends on the TabPFN inference path and the number of evaluable years at the chosen scale.

## Pipeline overview

### labels.py: the early-warning target

`labels.py` builds the peak-anchored early-warning label. The target is defined relative to the annual epidemic peak within each unit-year, which is what ties the achievable discrimination to the case series and motivates the internal-validity framing stated above.

### enrich.py: generative embedding of two columns into ~105 latent features

`enrich.py` is the method. It takes the two raw weekly columns and reconstructs the latent structure they contain as an explicit feature set of about 105 columns, grouped into mechanism-grounded families: calendar seasonality, week-of-year climatology, rainfall accumulation and threshold crossings, transmission-acceleration measures, and the catch22 time-series descriptors. Every feature carries a source tag (rain, case, or calendar) and a one-line rationale, exported in `feature_dictionary.csv`.

Every feature is causal. At week *t* a feature uses only weeks at or before *t* within the same unit-year, and any week-of-year climatology is built from prior years only, so no future information enters the representation. Feature engineering is confined to this module, which keeps the embedding step in one auditable place.

### evaluate.py: leave-one-year-out evaluation

`evaluate.py` runs the leave-one-year-out, expanding-window cross-validation. The model trains on earlier years and is tested on the held-out year. The imputer, the rank-Gaussian transform, and the classifier are fitted on the training years alone, so no statistic crosses from test to train. The classifier is TabPFN when the backend is available and the labelled stand-in otherwise. This protocol is fixed and is what keeps the reported numbers honest; it should not be altered.

### run.py: entry point and outputs

`run.py` is the one-command entry point. For the requested scale it writes `outputs/<scale>/` containing `metrics.json` (the raw to case-only to enriched ladder, with AUROC, AUPRC, and Brier score, and the `backend` recorded), `enriched.csv` (the full ~105-feature reconstruction), `feature_dictionary.csv`, `preds_enriched.npz` (out-of-fold `y_true` and `y_prob` used for calibration), and, for the regional and country scales, `regional_metrics.csv` with per-unit leave-one-year-out metrics and PAGASA climate type.

### make_figures.py: the comparison figures

`make_figures.py` writes `outputs/figures/`: the enrichment ladder per scale (raw versus case-only versus enriched), the feature-attribution chart ranking features by mutual information, the regional climate-gradient chart of AUROC by PAGASA climate type, and the reliability curve of the enriched model. The file `outputs/region_climate_type.csv` holds a best-effort dominant PAGASA type per region and can be edited to match a chosen reference, after which the gradient figure updates accordingly.

## Methodological notes

### Causal, leakage-safe features

Every feature respects the same-unit-year, weeks-at-or-before-*t* rule, and week-of-year baselines are drawn from prior years only. The intent is that nothing in the embedded representation could be unavailable to a real surveillance unit at the moment the probability is issued.

### Evaluation protocol

The leave-one-year-out design trains on earlier years and tests on the held-out year, with all fitting confined to the training years. Reporting always shows both ends of the ladder, the raw two-column rung and the enriched rung, because the contribution of the embedding is the comparison between them rather than either score on its own.

### Year exclusions

Pandemic and terminal-gap years are dropped using the workbook's own `FLAG_*` columns rather than a fixed list, so the exclusion logic is data-driven and travels with the dataset.

### Backend labelling

Results are tagged by backend in every output file. The `standin_histgb` results exist to demonstrate that the plumbing runs and are not foundation-model numbers; only `tabpfn` results are reported as such. Absolute scores differ between backends, while the lift from enrichment is the stable signal across them. The lift is larger under TabPFN, which, unlike a tree model, cannot exploit raw temporal order and therefore gains most from the explicit features.

### Honest caveat

The early-warning label is peak-anchored and case-derived, which caps achievable AUROC at about 0.85. The pipeline is a demonstration of the featurisation method, not an operational forecast-skill claim.

## Citation

If you use this code or the dataset, please cite both the manuscript and the dataset deposition:

Pelitro KJ, Manzano JF, Matavia TO, Soriano K, Bilbao K, Garcia GM, Delos Angeles AJ, Lagmay AM, Bandoy DJD. *Generative embedding extends tabular foundation models to dengue early warning from sparse surveillance data.* DOI to be assigned upon acceptance.

Pelitro KJ, et al. *Dengue-Rainfall_Dataset* (Version 1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.19448854

A `CITATION.cff` file is provided so that the GitHub repository renders a "Cite this repository" widget once the manuscript DOI is assigned.

## Licence

The code in this repository is released under the MIT Licence (see `LICENSE`).

The data are distributed under the Open Data Commons Open Database License (ODC-ODbL) v1.0 (https://opendatacommons.org/licenses/odbl/1.0/), which permits the use, distribution, and adaptation of data, provided that appropriate attribution is given to the UP Resilience Institute–NOAH (UPRI-NOAH) and its contributors and that derivative databases are shared under the same license.

## Funding

Research reported in this publication was supported by the National Institute of Environmental Health Sciences of the National Institutes of Health under Award Number P20ES036118. The content is solely the responsibility of the authors and does not necessarily represent the official views of the National Institutes of Health.

## Acknowledgements

We thank the Quezon City Epidemiology and Surveillance Division for the city-level dengue surveillance data, the Philippine Atmospheric, Geophysical, and Astronomical Services Administration for the meteorological data archived alongside the case series, and the Humanitarian Data Exchange platform for the regional dengue dataset. The country-level case counts were drawn from the OpenDengue repository. The tabular foundation model is TabPFN, accessed through Prior Labs.

## Contact

For questions about the code or to report a reproducibility issue, please open a GitHub issue. For questions about the underlying surveillance data or the manuscript, contact Keanu John Pelitro at kapelitro@up.edu.ph.
