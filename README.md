# IFE M-Protein Clinical Decision Support Dashboard

Interactive web-based Clinical Decision Support (CDS) system for cascade M-protein classification from 6-channel capillary zone electrophoresis immunotyping (CZE-IT) signals.

Built with [Streamlit](https://streamlit.io/) · No GPU required · Runs on pre-computed model outputs

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/<your-username>/ife-mprotein-cds.git
cd ife-mprotein-cds
pip install -r requirements.txt

# 2. Place data files (see "Data Files" section below)

# 3. Run
streamlit run app.py
```

---

## Project Structure

```
ife-mprotein-cds/
│
├── app.py                             # Main Streamlit application
├── config.py                          # Constants, thresholds, reflex matrix
├── data_loader.py                     # Data I/O, SHAP retrieval, feature NLP
├── plotting.py                        # Plotly (web) + Matplotlib (PDF) visuals
├── pdf_export.py                      # Two-page ReportLab PDF generator
├── llm_interpret.py                   # LLM clinical interpretation module
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
│
├── data/                              # ⬇ NOT in repo — user-provided
│   ├── dataset.pkl                    #   Signals + labels (37.4 MB)
│   ├── feature_dictionary.pkl         #   Feature descriptions (4 KB)
│   └── reflex_matrix.xlsx             #   (Optional) Custom reflex rules (8 KB)
│
└── results/                           # ⬇ NOT in repo — user-provided
    ├── flow_df.pkl                    #   Internal validation results (100 KB)
    ├── L4_shap_dense_full.pkl         #   SHAP matrices + base values (15.7 MB)
    └── L4_ext_validation_results.pkl  #   External validation results (20 KB)
```

---

## Source Code (6 files, ~2000 lines total)

### `app.py` (~560 lines)
Main Streamlit orchestration with three-layer information architecture:

- **Layer 1 — Executive Summary:** Classification result (large font), conformal prediction set (badges), confidence zone (traffic light), reflex test recommendation (from decision matrix), 6-channel signal trace (interactive Plotly)
- **Layer 2 — Interpretation & Evidence:** AI clinical interpretation (LLM or template), cumulative SHAP waterfall plots (3 levels) with top-3 feature explanation cards
- **Layer 3 — Deep Dive:** 6-channel spatial SHAP overlay (expandable), interactive feature ranking bar charts (expandable)

Three tabs: Clinical Report, Reflex Rules viewer, Debug & Prompts inspector.

### `config.py` (~120 lines)
All constants and the reflex testing decision matrix:

| Constant | Value | Description |
|----------|-------|-------------|
| `ZONE_THRESHOLDS` | HIGH ≥ 0.70, MEDIUM ≥ 0.30 | Confidence zone boundaries |
| `ZONE_COLORS` | `#1A9641` / `#F46D43` / `#D73027` | Traffic-light colors |
| `SHAP_POS_COLOR` | `#B2182B` | Red — pushes toward predicted class |
| `SHAP_NEG_COLOR` | `#2166AC` | Blue — pushes against predicted class |
| `CP_ALPHA` | 0.05 | Conformal prediction significance level |
| `PDF_DPI` | 300 | PDF export resolution |

Also contains: `pretty()` label formatter (IGG_KAPPA → IgG-κ), `UNIVERSAL_BASELINE` (7 tests), `REFLEX_MATRIX` (15 isotype × zone profiles), `reflex_group()`, `get_reflex()`, `generate_reflex_template()`.

### `data_loader.py` (~220 lines)
Data I/O with `@st.cache_resource`:

- `load_all_data()` → `(ds, shap_d, feat_dict, master_df)` — merges internal + external into unified DataFrame
- `get_patient_signal(ds, row)` → `ndarray (6, 300)` signal array
- `get_patient_shap(shap_d, level, sig_idx, is_ext, n=8)` → top-N `[(feat, value, shap)]` tuples
- `get_patient_shap_full(...)` → full 399-feature `(names, shap_vals, x_vals)` for spatial overlay
- `get_human_readable_parts(feat, val, shap, feat_dict)` → `(title, clinical_paragraph, shap_val)` NLP explanation
- `build_conformal_set(row)` → list of class labels in the prediction set
- `load_reflex_rules(xlsx_path, uploaded_file)` → `(baseline, matrix, source_label)` with fallback chain

### `plotting.py` (~450 lines)
Dual-engine visualizations:

| Function | Engine | Purpose |
|----------|--------|---------|
| `plotly_signal_faceted()` | Plotly | 6-row interactive signal with hover |
| `mpl_signal_combined()` | Matplotlib | 6-channel overlay for PDF |
| `mpl_signal_faceted()` | Matplotlib | 6-row static version |
| `plotly_shap_waterfall()` | Plotly | Horizontal bar chart with hover |
| `mpl_shap_waterfall()` | Matplotlib | Static bar chart for PDF |
| `mpl_shap_waterfall_cumulative()` | Matplotlib | True waterfall: E[f(x)] → cumulative → f(x) |
| `mpl_shap_6channel_overlay()` | Matplotlib | 399 SHAP values as colored region fills on signal |
| `render_conformal_set_html()` | HTML | Badge list with zone-colored action indicator |

### `pdf_export.py` (~330 lines)
Two-page ReportLab PDF:

- **Page 1:** Classification, conformal set, zone, reflex (from matrix with Gel IFE color-coded), signal trace, 3× SHAP waterfalls, AI interpretation (compact)
- **Page 2:** Universal baseline panel (7 tests), XAI textual explanations per cascade level

### `llm_interpret.py` (~320 lines)
LLM integration with dual-mode prompts (Research: GT visible / Clinical: GT hidden), API key fallback chain (`st.secrets` → `os.environ` → sidebar), session-state caching, template fallback, publication mode. Model: `claude-sonnet-4-20250514`, max_tokens: 512.

---

## Data Files

> **Not included in the repository.** These files contain patient data and model outputs from the cascade training pipeline.

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
    base_value      ndarray (1,) float64           Same as L1: -0.120

  L2_external     dict           External L2 — 223 positives
    shap_matrix     ndarray (223, 399) float32
    base_value      ndarray (4,) float64           Same as L2
    threshold_used  float                          0.472 (Youden-optimized L1 threshold)

  L3_external     dict           External L3 — 223 positives
    shap_matrix     ndarray (223, 399) float32
    base_value      ndarray (1,) float64           Same as L3: -0.762
    threshold_used  float                          0.472

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

Defined in `config.py`, aligned with manuscript Table 3:

| Zone | Threshold | Internal (OOF, n=2219) | External (n=498) |
|------|-----------|------------------------|-------------------|
| HIGH | ≥ 0.70 | n=1380, 62.2%, acc=96.2% | n=179, 35.9%, acc=97.2% |
| MEDIUM | 0.30–0.70 | n=602, 27.1%, acc=82.9% | n=167, 33.5%, acc=91.6% |
| LOW | < 0.30 | n=237, 10.7%, acc=45.6% | n=152, 30.5%, acc=70.4% |

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

## Screenshots

> Enable **📸 Publication Mode** in the sidebar to hide API key UI and deployment artifacts.

High-resolution capture: Chrome DevTools → Device toolbar → Custom device (Width: 1400, Height: 8000, DPR: 4) → Ctrl+Shift+P → "Capture screenshot"

---

## Citation

```
[Citation to be added after publication]
```

---

## License

[License to be determined]
