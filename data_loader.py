"""
data_loader.py – All data I/O: dataset, SHAP, feature dictionary, master DataFrame.
"""
import pickle
import os
import numpy as np
import pandas as pd
import streamlit as st
from config import (DATA_DIR, RES_DIR, D9, ZONE_THRESHOLDS,
                    DATASET_FILE, FEATURE_DICT, FLOWDF_FILE,
                    SHAP_FILE, EXTVAL_FILE)


@st.cache_resource
def load_all_data():
    """Return (ds, shap_d, feat_dict, master_df)."""
    with open(DATASET_FILE, 'rb') as f:
        ds = pickle.load(f)
    with open(SHAP_FILE, 'rb') as f:
        shap_d = pickle.load(f)
    with open(FEATURE_DICT, 'rb') as f:
        feat_dict = pickle.load(f)

    # ── Internal data ──────────────────────────────────
    with open(FLOWDF_FILE, 'rb') as f:
        raw = pickle.load(f)
    if isinstance(raw, pd.DataFrame):
        df_int = raw.copy()
    else:
        df_int = raw.get('flow_df', None)
        if df_int is None:
            for v in raw.values():
                if isinstance(v, pd.DataFrame):
                    df_int = v.copy(); break

    df_int['source']     = 'Internal'
    df_int['sig_idx']    = df_int.index
    df_int['patient_id'] = [str(ds['sample_ids'][i]) for i in df_int.index]

    # ── External data ──────────────────────────────────
    with open(EXTVAL_FILE, 'rb') as f:
        ext = pickle.load(f)

    ext_pred_class = [D9.get(p, str(p)) for p in ext['ext_pred']]
    ext_true_class = [D9.get(t, str(t)) for t in ds['y_ext_class9']]
    ext_correct    = [int(p == t) for p, t in zip(ext_pred_class, ext_true_class)]

    l1_p   = ext['ext_l1_proba']
    l2_max = np.max(ext['ext_l2_proba'], axis=1)
    l3_max = np.maximum(ext['ext_l3_proba'], 1 - ext['ext_l3_proba'])

    ext_conf = []
    for i, pc in enumerate(ext_pred_class):
        if pc == 'NEGATIVE':
            ext_conf.append(1.0 - l1_p[i])
        else:
            ext_conf.append(l1_p[i] * l2_max[i] * l3_max[i])

    ext_zone = [
        'HIGH' if c >= ZONE_THRESHOLDS['HIGH']
        else 'MEDIUM' if c >= ZONE_THRESHOLDS['MEDIUM']
        else 'LOW'
        for c in ext_conf
    ]

    df_ext = pd.DataFrame({
        'pred_class': ext_pred_class, 'zone': ext_zone, 'confidence': ext_conf,
        'true_class': ext_true_class, 'correct': ext_correct,
        'action': 'External validation', 'source': 'External',
        'sig_idx': range(len(ext['ext_pred'])),
        'patient_id': [str(x) for x in ds['ext_sample_ids']],
        'p_L1': l1_p, 'p_L2': l2_max, 'p_L3': l3_max,
    })

    master_df = pd.concat([df_int, df_ext], axis=0)
    master_df.set_index('patient_id', inplace=True)
    return ds, shap_d, feat_dict, master_df


# ── Helper: get signal array for a patient ────────────────
def get_patient_signal(ds, row):
    """Return (6, T) signal array."""
    idx_int = int(row.name) if isinstance(row.name, str) else row.name
    if row['source'] == 'External':
        pos = np.where(ds['ext_sample_ids'] == idx_int)[0][0]
        return ds['X_ext_3d'][pos]
    else:
        pos = np.where(ds['sample_ids'] == idx_int)[0][0]
        return ds['X_3d'][pos]


# ── SHAP retrieval ────────────────────────────────────────
def get_patient_shap(shap_d, level, sig_idx, is_ext, n=8):
    """Return list of (feat_name, feat_value, shap_value) tuples or status string."""
    key = f"{level}_external" if is_ext else level
    if key not in shap_d:
        return "MISSING_KEY"
    si = np.array(shap_d[key].get('sample_indices', []))
    matches = np.where(si == sig_idx)[0]
    if len(matches) == 0:
        return "NOT_IN_INDEX"
    p  = matches[0]
    sv = shap_d[key]['shap_matrix'][p]
    xv = shap_d[key]['X_matrix'][p]
    feat_names = shap_d.get('feature_names', [f"Feature_{i}" for i in range(len(sv))])
    top = np.argsort(np.abs(sv))[::-1][:n]
    return [(feat_names[j], float(xv[j]), float(sv[j])) for j in top]

def get_patient_shap_full(shap_d, level, sig_idx, is_ext):
    """Return full (feat_names, shap_values_399, x_values_399) or None."""
    key = f"{level}_external" if is_ext else level
    if key not in shap_d:
        return None
    si = np.array(shap_d[key].get('sample_indices', []))
    matches = np.where(si == sig_idx)[0]
    if len(matches) == 0:
        return None
    p = matches[0]
    sv = shap_d[key]['shap_matrix'][p]
    xv = shap_d[key]['X_matrix'][p]
    feat_names = shap_d.get('feature_names', [f"Feature_{i}" for i in range(len(sv))])
    return feat_names, sv, xv

# ── NLP feature explanation ───────────────────────────────
def get_human_readable_parts(feat_name, val, shap_val, feat_dict):
    """Map raw feature name → (title, clinical_paragraph, shap_value)."""
    # Cross-channel features
    for cc_key in feat_dict.get("cross_channel", {}).keys():
        if cc_key in feat_name:
            desc = feat_dict["cross_channel"][cc_key]
            parts = desc.split('. ', 1)
            def_part = parts[0].lower()
            clin_part = parts[1] if len(parts) > 1 else ""
            reg_desc = ""
            for reg in feat_dict.get("regions", {}).keys():
                if reg in feat_name:
                    reg_desc = f" specifically in the {feat_dict['regions'][reg]}"
                    break
            paragraph = f"This evaluates the {def_part}{reg_desc}. {clin_part}"
            return feat_name, paragraph, shap_val

    ch_desc, reg_desc = "entire chromatogram", "entire signal"
    metric_def, metric_clin = "signal measurement", ""

    for ch in feat_dict.get("channels", {}).keys():
        if feat_name.startswith(ch):
            ch_desc = feat_dict["channels"][ch]; break

    for reg in feat_dict.get("regions", {}).keys():
        if f"_{reg}" in feat_name or f"{reg}_" in feat_name:
            reg_desc = feat_dict["regions"][reg]; break

    for met in sorted(feat_dict.get("metrics", {}).keys(), key=len, reverse=True):
        if met in feat_name:
            full = feat_dict["metrics"][met]
            parts = full.split('. ', 1)
            metric_def  = parts[0].lower()
            metric_clin = parts[1] if len(parts) > 1 else ""
            break

    paragraph = (f"This evaluates the {ch_desc} within the {reg_desc}. "
                 f"Specifically, it measures the {metric_def}. {metric_clin}")
    return feat_name, paragraph, shap_val


# ── Conformal prediction set (simple heuristic) ──────────
def build_conformal_set(row):
    """Build conformal prediction set based on zone/confidence.
    Returns list of candidate class labels."""
    pred = row['pred_class']
    zone = row['zone']
    if zone == 'HIGH':
        return [pred]
    elif zone == 'MEDIUM':
        return sorted(set([pred, 'NEGATIVE']))
    else:
        # LOW zone: include nearest plausible alternatives
        candidates = {pred, 'NEGATIVE'}
        # Add light-chain variants
        if 'KAPPA' in pred:
            candidates.add(pred.replace('KAPPA', 'LAMBDA'))
        elif 'LAMBDA' in pred:
            candidates.add(pred.replace('LAMBDA', 'KAPPA'))
        else:
            candidates.update(['FREE_KAPPA', 'FREE_LAMBDA'])
        return sorted(candidates)[:5]

# ═══════════════════════════════════════════════════════════════
#  REFLEX RULES LOADER (Excel → dict, fallback to config.py)
# ═══════════════════════════════════════════════════════════════

def load_reflex_rules(xlsx_path="data/reflex_matrix.xlsx", uploaded_file=None):
    """
    Load reflex rules from Excel. Returns (baseline, matrix, source_label).
    Priority: uploaded_file > xlsx_path > config.py hardcoded fallback.
    """
    import pandas as pd
    from config import UNIVERSAL_BASELINE, REFLEX_MATRIX

    source = None
    xls = None

    # 1. Try uploaded file first
    if uploaded_file is not None:
        try:
            xls = pd.ExcelFile(uploaded_file)
            source = f"📤 Uploaded: {uploaded_file.name}"
        except Exception as e:
            st.warning(f"Could not read uploaded file: {e}. Falling back.")

    # 2. Try default path
    if xls is None and os.path.exists(xlsx_path):
        try:
            xls = pd.ExcelFile(xlsx_path)
            source = f"📁 {xlsx_path}"
        except Exception as e:
            st.warning(f"Could not read {xlsx_path}: {e}. Falling back.")

    # 3. Fallback to hardcoded
    if xls is None:
        return UNIVERSAL_BASELINE, REFLEX_MATRIX, "⚠️ Hardcoded defaults (no Excel found)"

    # Parse sheets
    try:
        # Baseline
        df_base = pd.read_excel(xls, sheet_name="Baseline")
        baseline = list(zip(df_base["test"].tolist(), df_base["rationale"].tolist()))

        # Matrix
        df_mat = pd.read_excel(xls, sheet_name="Reflex_Matrix")
        matrix = {}
        for _, r in df_mat.iterrows():
            grp = str(r["class_group"]).strip()
            zone = str(r["zone"]).strip().upper()
            tests_raw = r.get("tests", "")
            if pd.isna(tests_raw) or str(tests_raw).strip().lower() == "nan":
                tests = []
            else:
                tests = [t.strip() for t in str(tests_raw).split(";") if t.strip()]

            if grp not in matrix:
                matrix[grp] = {}
            matrix[grp][zone] = {
                "gel_ife": str(r["gel_ife"]).strip(),
                "tests": tests,
                "guidance": "" if pd.isna(r.get("guidance", "")) else str(r["guidance"]).strip(),
            }

        return baseline, matrix, source

    except Exception as e:
        st.warning(f"Error parsing Excel sheets: {e}. Falling back to defaults.")
        return UNIVERSAL_BASELINE, REFLEX_MATRIX, "⚠️ Hardcoded defaults (Excel parse error)"
