# IFE M-Protein Clinical Decision Support Dashboard

Interactive web-based Clinical Decision Support (CDS) system for cascade M-protein classification from 6-channel capillary zone electrophoresis immunotyping (CZE-IT) signals.

Built with [Streamlit](https://streamlit.io/) · No GPU required · Renders pre-computed cohort
results, plus an optional frozen-inference path for user-supplied signals

> **Research use only. Not a medical device.** This dashboard is a research prototype and must
> not be used to guide patient care. The same notice is shown as a non-dismissible banner on
> every screen of the application and in the footer of every exported PDF.

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/ditopcu/CDS-IT-mprotein-cascade-dashboard.git
cd CDS-IT-mprotein-cascade-dashboard
pip install -r requirements.txt

# 2. Run — demo mode is the default and needs no additional data
streamlit run app.py
```

The committed demo cohort (50 anonymized patients) ships with the repository, so a fresh clone
runs out of the box. To run against the full, non-committed cohort instead:

```bash
DEMO_MODE=false streamlit run app.py
```

`DEMO_MODE` (default `true`) selects the file suffix used throughout `config.py`: `_demo`
variants when on, the full files when off. The full files hold patient data and are never
committed.

---

## Project Structure

```
CDS-IT-mprotein-cascade-dashboard/
│
├── app.py                             # Streamlit UI orchestration
├── config.py                          # Constants, thresholds, reflex matrix, DEMO_MODE
├── data_loader.py                     # Pickle I/O, SHAP retrieval, feature NLP
├── inference.py                       # Frozen cascade math + upload pipeline
├── plotting.py                        # Plotly (web) + Matplotlib (PDF) visuals
├── pdf_export.py                      # ReportLab PDF report
├── llm_interpret.py                   # Optional LLM interpretation
├── requirements.txt                   # Python dependencies
│
├── cascade_src/                       # Frozen algorithm code, vendored byte-identically
│   ├── features.py  cascade.py        #   from the model repository — never edited here
│   ├── confidence.py  calibration.py
│   └── constants.py
│
├── pkl/                               # ✓ in repo — committed, md5-gated 5-fold ensembles
│   └── L{1,2,3}_xgb_peak_optuna_models.pkl
│
├── data/                              # ✓ demo variants in repo
│   ├── dataset_demo.pkl               #   50 anonymized patients
│   ├── feature_dictionary.pkl         #   Feature descriptions (no demo variant)
│   └── reflex_matrix.xlsx             #   Optional editable reflex rules
│
└── results/                           # ✓ demo variants in repo
    ├── flow_df_demo.pkl
    ├── L4_shap_dense_full_demo.pkl
    └── L4_ext_validation_results_demo.pkl
```

The full-cohort counterparts (`dataset.pkl`, `flow_df.pkl`, `L4_shap_dense_full.pkl`,
`L4_ext_validation_results.pkl`) hold patient data, are gitignored, and are never committed.

---

## Modules

| Module | Responsibility |
|--------|----------------|
| `app.py` | UI orchestration only. The sidebar filters build the cohort subset; the main panel resolves one patient and renders the tabs. All layout and inline CSS live here. |
| `config.py` | Constants, `DEMO_MODE` and file paths, label maps, zone thresholds and colours, and the hardcoded reflex testing decision matrix (`REFLEX_MATRIX`, `UNIVERSAL_BASELINE`, `reflex_group()`). |
| `data_loader.py` | All pickle I/O behind a cached `load_all_data()`, merging the internal and external cohorts into one `master_df`. Also SHAP retrieval, the feature-name → clinical-prose NLP, and the reflex-rules loader. |
| `inference.py` | The frozen cascade math and the upload pipeline: `build_p9`, `conformal_set`, `cohort_confidence`, `run_frozen_cascade`, and the md5 integrity helpers. |
| `plotting.py` | Two rendering engines kept in sync: `plotly_*` for the interactive view, `mpl_*` for the PDF. |
| `pdf_export.py` | The ReportLab report, built from the Matplotlib figures and embedded through `BytesIO` — never a temp file, since the plots are patient-derived. |
| `llm_interpret.py` | Optional Anthropic interpretation, with Research (ground truth visible) and Clinical (ground truth hidden) prompt modes. Without a key it degrades to template text and never crashes the app. |
| `cascade_src/` | The frozen algorithm code, vendored byte-identically from the model repository. Edit upstream and re-copy; never edit it here. |

### Two paths

The **cohort view** renders pre-computed results — predictions, confidence, SHAP matrices — and
runs no inference at all.

The **upload path** runs frozen forward inference on user-supplied de-identified signals, using
the committed 5-fold ensembles in `pkl/` together with `cascade_src/`. It never trains, refits,
or recalibrates. The model files are md5-gated against the publication lineage before they are
unpickled; on a mismatch the upload path is disabled and the cohort view keeps working. Uploads
are parsed from an in-memory buffer, held only for the session, and never written to disk.

The pinned block at the bottom of `requirements.txt` is load-bearing for this path — relaxing
those versions silently changes predictions.

---

## Data Files

> **The full-cohort files are not in the repository** — they contain patient data. The `_demo`
> variants (50 anonymized patients) *are* committed, and are what the default `DEMO_MODE=true`
> loads. `feature_dictionary.pkl` has no demo variant; both modes share it.
>
> The schemas below describe the full files. The demo variants have the same structure with
> reduced first dimensions.

### `data/dataset.pkl` (37.4 MB)

```
dict with 33 keys:

Signals:
  X_3d             ndarray (2219, 6, 300) float32    Internal 6-channel CZE signals
  X_ext_3d         ndarray (498, 6, 300)  float32    External 6-channel CZE signals
  X                ndarray (2219, 1800)   float32    Internal signals flattened (6×300)
  X_ext            ndarray (498, 1800)    float32    External signals flattened

Patient IDs:
  sample_ids       ndarray (2219,) int64             Internal patient IDs
  ext_sample_ids   ndarray (498,)  int64             External patient IDs

Ground truth labels (string):
  y_class9         ndarray (2219,) object            9-class: FREE_KAPPA..NEGATIVE
  y_binary         ndarray (2219,) object            NEGATIVE / POSITIVE
  y_heavy          ndarray (2219,) object            IGG/IGA/IGM/FREE (+1176 NaN for negatives)
  y_light          ndarray (2219,) object            KAPPA/LAMBDA (+1176 NaN for negatives)
  y_ext_class9     ndarray (498,)  object            External 9-class (8 unique, no IGM_LAMBDA)

Ground truth labels (encoded):
  y_class9_enc     ndarray (2219,) int8              0-8 encoded
  y_binary_enc     ndarray (2219,) int8              0=NEG, 1=POS
  y_heavy_enc      ndarray (2219,) int8              0=IGG,1=IGA,2=IGM,3=FREE (-1=neg)
  y_light_enc      ndarray (2219,) int8              0=KAPPA,1=LAMBDA (-1=neg)
  pos_mask         ndarray (2219,) bool              True for L1=Positive samples
  (same _ext_ variants for external)

Encoding maps:
  class9_map       dict    {FREE_KAPPA:0, FREE_LAMBDA:1, ..., NEGATIVE:8}
  binary_map       dict    {NEGATIVE:0, POSITIVE:1}
  heavy_map        dict    {IGG:0, IGA:1, IGM:2, FREE:3}
  light_map        dict    {KAPPA:0, LAMBDA:1}

Metadata:
  channels         list    ['raw_ELP','dif_IgG','dif_IgA','dif_IgM','dif_Kappa','dif_Lambda']
  ch_idx           dict    {raw_ELP:0, dif_IgG:1, ..., dif_Lambda:5}
  feature_cols     list    1800 entries: ['x0_raw_ELP', ..., 'x299_dif_Lambda']
  n_timepoints     int     300
  n_channels       int     6
```

**Dashboard uses:** `X_3d`, `X_ext_3d`, `sample_ids`, `ext_sample_ids`, `y_class9`, `y_ext_class9`

### `data/feature_dictionary.pkl` (4 KB)

```
dict with 4 keys:

  channels        dict (6 entries)
                  Maps channel prefix → description
                  e.g. "dif_IgG" → "IgG difference channel (ELP - IgG)"

  regions         dict (6 entries)
                  Maps region suffix → description
                  e.g. "gamma" → "Gamma (Immunoglobulin zone)"
                  Keys: beta1, beta2, transition, gamma, beta_full, mprotein

  metrics         dict (26 entries)
                  Maps metric suffix → definition + clinical interpretation
                  e.g. "sharp" → "Maximum absolute second derivative. Measures
                  peak sharpness (M-proteins produce narrow, sharp bands)."

  cross_channel   dict (14 entries)
                  Multi-channel derived features
                  e.g. "kl_corr" → "Pearson correlation between Kappa and Lambda.
                  Low correlation suggests monoclonal expression."
                  e.g. "igg_v_igm" → "Differential between IgG and IgM peak intensities."
```

### `data/reflex_matrix.xlsx` (8 KB) — *optional*

Editable reflex testing rules. If absent, hardcoded defaults in `config.py` are used.

| Sheet (rows × cols) | Columns | Description |
|---------------------|---------|-------------|
| Baseline (7 × 2) | test, rationale | Universal tests for all M-protein positive |
| Reflex_Matrix (15 × 5) | class_group, zone, gel_ife, tests, guidance | 5 groups × 3 zones |
| Workflow (3 × 2) | zone, action | Clinical routing per zone |

`tests` column uses semicolon-delimited values. Empty cells (NaN) are handled gracefully.

Generate from defaults: sidebar 📥 Template button, or:
```python
from config import generate_reflex_template
generate_reflex_template("data/reflex_matrix.xlsx")
```

### `results/flow_df.pkl` (100 KB)

```
DataFrame (2219 rows × 9 columns):

  pred_class    str     Predicted 9-class label (e.g. "IGG_KAPPA")
  true_class    str     Ground truth label
  confidence    float   Compound confidence score (0.0–1.0)
  zone          str     HIGH / MEDIUM / LOW
  correct       int     1 if pred == true, else 0
  action        str     Workflow action text
  p_L1          float   L1 probability: p(Positive)
  p_L2          float   L2 max class probability (NaN for negatives)
  p_L3          float   L3 max class probability (NaN for negatives)

Index: integer (0–2218), maps to dataset sample position
```

### `results/L4_shap_dense_full.pkl` (15.7 MB)

```
dict with 8 keys:

  feature_names   list (399)     Feature name strings
                                 e.g. ['raw_ELP_mean','raw_ELP_std',...,'any_mp_100']

  L1              dict           Internal L1 (Binary) SHAP
    shap_matrix     ndarray (2219, 399) float32   SHAP values
    X_matrix        ndarray (2219, 399) float32   Feature values
    sample_indices  ndarray (2219,) int64          Dataset row indices
    base_value      ndarray (1,) float64           E[f(x)] = -0.120

  L2              dict           Internal L2 (Heavy Chain) SHAP — positives only
    shap_matrix     ndarray (1043, 399) float32
    X_matrix        ndarray (1043, 399) float32
    sample_indices  ndarray (1043,) int64          First: [6, 10, 13, ...]
    base_value      ndarray (4,) float64           Per-class: [IgG, IgA, IgM, Free]
                                                   = [0.003, 0.001, -0.010, -0.002]

  L3              dict           Internal L3 (Light Chain) SHAP — positives only
    shap_matrix     ndarray (1043, 399) float32
    X_matrix        ndarray (1043, 399) float32
    sample_indices  ndarray (1043,) int64
    base_value      ndarray (1,) float64           E[f(x)] = -0.762

  L1_external     dict           External L1 — all 498 samples
    shap_matrix     ndarray (498, 399) float32
    X_matrix        ndarray (498, 399) float32
    sample_indices  ndarray (498,) int64
    base_value      ndarray (1,) float64           Same as L1: -0.120

  L2_external     dict           External L2 — 223 positives
    shap_matrix     ndarray (223, 399) float32
    X_matrix        ndarray (223, 399) float32
    sample_indices  ndarray (223,) int64
    base_value      ndarray (4,) float64           Same as L2
    threshold_used  float                          0.4722 (frozen L1 threshold)

  L3_external     dict           External L3 — 223 positives
    shap_matrix     ndarray (223, 399) float32
    X_matrix        ndarray (223, 399) float32
    sample_indices  ndarray (223,) int64
    base_value      ndarray (1,) float64           Same as L3: -0.762
    threshold_used  float                          0.4722

  metadata        dict
    L1_threshold    float    0.472 (Youden-optimized)
    internal_n      int      2219
    internal_pos_n  int      1043
    external_n      int      498
    external_pos_n  int      223
    seed            int      42
    computed_at     str      "2026-03-22 13:15"
```

**Note on base_value:** L1 and L3 are binary classifiers → scalar base. L2 is 4-class → one base value per class. The dashboard selects the element matching the predicted heavy chain (e.g., IgG → index 0).

### `results/L4_ext_validation_results.pkl` (20 KB)

```
dict with 4 keys:

  ext_pred       ndarray (498,) str     Predicted class strings
                                        9 unique values (same as internal)

  ext_l1_proba   ndarray (498,) float32 L1 p(Positive), range [0.011..0.999]

  ext_l2_proba   ndarray (498, 4) float32  L2 class probabilities
                                           Columns: [IgG, IgA, IgM, Free]

  ext_l3_proba   ndarray (498,) float32 L3 p(Lambda), range [0.000..0.980]
```

**Note:** `ext_pred` contains string labels (e.g., "IGG_KAPPA"), not integer indices. `ext_l3_proba` is p(Lambda); p(Kappa) = 1 - p(Lambda).

---

## Configuration

### Confidence Zone Thresholds

Defined in `config.py`, matching Supplemental Table S8 of the manuscript:

| Zone | Threshold | Internal (OOF, n=2219) | External (n=498) |
|------|-----------|------------------------|-------------------|
| HIGH | ≥ 0.70 | n=1380, 62.2%, acc=96.2% | n=254, 51.0%, acc=98.8% |
| MEDIUM | 0.30–0.70 | n=602, 27.1%, acc=82.9% | n=169, 33.9%, acc=87.6% |
| LOW | < 0.30 | n=237, 10.7%, acc=45.6% | n=75, 15.1%, acc=46.7% |

The dashboard recomputes these from the frozen probabilities rather than reading stored zone
labels. `compute_cascade_confidence` indexes the L2/L3 arrays by **position within `pos_idx`**,
so `data_loader.load_all_data()` and `inference.run_frozen_cascade()` both slice to
`l2[pos_idx]` / `l3[pos_idx]` before calling `inference.cohort_confidence()`.

Confidence and zone are computed *downstream* of the classification and never feed back into
it, so they affect triage routing only — never the predicted class. External accuracy is
0.8715 (434/498) regardless of how the zones are cut.

### Compound Confidence Formula

```
conf_L1 = |p₁ − τ| / max(τ, 1−τ)     where τ = 0.472 (Youden threshold)

Positive: conf_compound = conf_L1 × max(P_L2) × 2|p₃ − 0.5|
Negative: conf_compound = conf_L1
```

### AI Interpretation (Optional)

Requires an Anthropic API key. Three methods (checked in order):

1. **Streamlit secrets:** `.streamlit/secrets.toml`
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
2. **Environment variable:**
   ```bash
   export ANTHROPIC_API_KEY="sk-ant-..."
   ```
3. **Sidebar input:** Enter directly in the app (session-only)

Without an API key, the app uses template-based interpretations — it never crashes.

---

## Label Conventions

| Internal Code | Display Label | Class Index |
|---------------|---------------|-------------|
| NEGATIVE | Negative | 8 |
| IGG_KAPPA | IgG-κ | 4 |
| IGG_LAMBDA | IgG-λ | 5 |
| IGA_KAPPA | IgA-κ | 2 |
| IGA_LAMBDA | IgA-λ | 3 |
| IGM_KAPPA | IgM-κ | 6 |
| IGM_LAMBDA | IgM-λ | 7 |
| FREE_KAPPA | Free-κ | 0 |
| FREE_LAMBDA | Free-λ | 1 |

**Note:** External dataset has 8 classes (no IGM_LAMBDA, n=0).

---

## Sidebar Controls

| Control | Options | Description |
|---------|---------|-------------|
| Direct ID Search | text | Patient ID lookup |
| Data Source | ALL / Internal / External | Cohort filter |
| M-Protein Type | ALL + 9 classes | Isotype filter |
| Confidence Zone | ALL / HIGH / MEDIUM / LOW | Zone filter |
| Result | ALL / Correct / Incorrect | Correctness filter |
| Conformal Set Size | ALL / 1 / 2+ / 3+ | Prediction set cardinality |
| L2/L3 SHAP toggle | on/off | Show only patients with full SHAP coverage |
| Mask Patient ID | on/off | Replace first 4 digits with XXXX |
| Publication Mode | on/off | Hide API key UI and deployment artifacts |
| AI Interpretation | on/off | Enable LLM / template interpretation |
| Interpretation Mode | Research / Clinical | Include or hide ground truth in prompt |
| Reflex Rules | upload / reload / download | Custom Excel-based rules |

---

## Citation

The cascade classifier, the analysis code, and the reproducibility package live in the model
repository and are archived with a persistent identifier:

- Code: https://github.com/ditopcu/CDS-IT-mprotein-cascade
- Archive: [10.5281/zenodo.19279916](https://doi.org/10.5281/zenodo.19279916)

The manuscript citation will be added once the paper is published.

---

## License

Not yet determined. Until a licence is added, no permissions beyond viewing the source are
granted.
