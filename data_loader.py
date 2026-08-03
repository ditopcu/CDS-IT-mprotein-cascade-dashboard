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
                    SHAP_FILE, EXTVAL_FILE,
                    ALIC_DATASET_FILE, ALIC_FLOWDF_FILE, ALIC_SHAP_FILE,
                    L1_THRESHOLD, CONFORMAL_PROB_THR)
import inference as inf

# source name → SHAP-dict key suffix
SRC_SUFFIX = {'Internal': '', 'External': '_external', 'Alicante': '_alicante'}

# internal OOF probabilities for the split-conformal set (paper's out-of-fold preds)
_OOF_DIRS = [os.path.join("revision", "repor"), "oof_internal"]


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
    # Internal confidence/zone are the paper's isotonic-calibrated values already in
    # flow_df (HIGH 62.2%) — keep them. Only add the split-conformal set (from OOF P9).
    df_int['conformal_set'] = _internal_conformal_sets(df_int)

    # ── External data ──────────────────────────────────
    with open(EXTVAL_FILE, 'rb') as f:
        ext = pickle.load(f)

    ext_pred_class = [D9.get(p, str(p)) for p in ext['ext_pred']]
    ext_true_class = [D9.get(t, str(t)) for t in ds['y_ext_class9']]
    ext_correct    = [int(p == t) for p, t in zip(ext_pred_class, ext_true_class)]

    # Canonical compound confidence + zone (Task 1) — replaces the old simplified formula
    # whose floor never dropped below 0.30 (so it could never assign a LOW zone).
    l1_p    = np.asarray(ext['ext_l1_proba'], float)
    l2_full = np.asarray(ext['ext_l2_proba'], float)      # (N,4) [IgG,IgA,IgM,Free]
    l3_lam  = np.asarray(ext['ext_l3_proba'], float)      # P(lambda)
    pos_idx = np.where(l1_p >= L1_THRESHOLD)[0]
    ext_conf, ext_zone = inf.cohort_confidence(l1_p, l2_full[pos_idx], l3_lam[pos_idx], pos_idx)
    # Split-conformal prediction set (Task 2)
    P9_ext   = inf.build_p9(l1_p, l2_full, l3_lam)
    ext_sets = inf.conformal_sets(P9_ext, CONFORMAL_PROB_THR)
    l2_max = l2_full.max(axis=1)
    l3_max = np.maximum(l3_lam, 1 - l3_lam)               # kept for the probability display

    df_ext = pd.DataFrame({
        'pred_class': ext_pred_class, 'zone': list(ext_zone), 'confidence': list(ext_conf),
        'true_class': ext_true_class, 'correct': ext_correct,
        'action': 'External validation', 'source': 'External',
        'sig_idx': range(len(ext['ext_pred'])),
        'patient_id': [str(x) for x in ds['ext_sample_ids']],
        'p_L1': l1_p, 'p_L2': l2_max, 'p_L3': l3_max,
        'conformal_set': ext_sets,
    })

    frames = [df_int, df_ext]

    # ── Alicante cohort (optional — merged only if the files exist) ──────
    if ALIC_DATASET_FILE.exists() and ALIC_FLOWDF_FILE.exists():
        with open(ALIC_DATASET_FILE, 'rb') as f:
            ds_alic = pickle.load(f)
        ds['X_alic_3d']      = ds_alic['X_alic_3d']
        ds['alic_sample_ids'] = np.asarray(ds_alic['alic_sample_ids'])

        with open(ALIC_FLOWDF_FILE, 'rb') as f:
            df_alic = pickle.load(f).copy()
        df_alic['source']     = 'Alicante'
        df_alic['sig_idx']    = df_alic.index
        df_alic['patient_id'] = [str(ds['alic_sample_ids'][i]) for i in df_alic.index]

        # Canonical confidence/zone + split-conformal set (same code path as External),
        # using the full per-level probabilities stored in the Alicante dataset artifact.
        if all(k in ds_alic for k in ('alic_l1', 'alic_l2', 'alic_l3')):
            al1 = np.asarray(ds_alic['alic_l1'], float)
            al2 = np.asarray(ds_alic['alic_l2'], float)
            al3 = np.asarray(ds_alic['alic_l3'], float)
            apos = np.where(al1 >= L1_THRESHOLD)[0]
            ac, az = inf.cohort_confidence(al1, al2[apos], al3[apos], apos)
            df_alic['confidence']    = list(ac)
            df_alic['zone']          = list(az)
            df_alic['conformal_set'] = inf.conformal_sets(inf.build_p9(al1, al2, al3),
                                                          CONFORMAL_PROB_THR)
        else:
            df_alic['conformal_set'] = [_heuristic_conformal_set(r) for _, r in df_alic.iterrows()]
        frames.append(df_alic)

        if ALIC_SHAP_FILE.exists():
            with open(ALIC_SHAP_FILE, 'rb') as f:
                shap_alic = pickle.load(f)
            for k, v in shap_alic.items():
                if k.endswith('_alicante'):
                    shap_d[k] = v

    master_df = pd.concat(frames, axis=0)
    master_df.set_index('patient_id', inplace=True)
    return ds, shap_d, feat_dict, master_df


# ── Helper: get signal array for a patient ────────────────
def get_patient_signal(ds, row):
    """Return (6, T) signal array."""
    idx_int = int(row.name) if isinstance(row.name, str) else row.name
    if row['source'] == 'External':
        pos = np.where(ds['ext_sample_ids'] == idx_int)[0][0]
        return ds['X_ext_3d'][pos]
    elif row['source'] == 'Alicante':
        pos = np.where(ds['alic_sample_ids'] == idx_int)[0][0]
        return ds['X_alic_3d'][pos]
    else:
        pos = np.where(ds['sample_ids'] == idx_int)[0][0]
        return ds['X_3d'][pos]


# ── SHAP retrieval ────────────────────────────────────────
def get_patient_shap(shap_d, level, sig_idx, src_suffix='', n=8):
    """Return list of (feat_name, feat_value, shap_value) tuples or status string.

    src_suffix: '' (Internal), '_external', or '_alicante' — see SRC_SUFFIX.
    """
    key = f"{level}{src_suffix}"
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

def get_patient_shap_full(shap_d, level, sig_idx, src_suffix=''):
    """Return full (feat_names, shap_values_399, x_values_399) or None."""
    key = f"{level}{src_suffix}"
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


# ── Split-conformal prediction set ───────────────────────
@st.cache_resource
def _internal_P9():
    """Internal (2219,9) OOF P9 for the split-conformal set, or None if unavailable."""
    for d in _OOF_DIRS:
        if os.path.isdir(d):
            P9 = inf.build_internal_P9_from_oof(d)
            if P9 is not None:
                return P9
    return None


def _internal_conformal_sets(df_int):
    """Per-row split-conformal set for the internal cohort (paper procedure via OOF P9).

    Full mode: flow_df index is 0..2218 and aligns to the OOF P9 by position. If the OOF
    is missing or misaligned (e.g. DEMO_MODE's reindexed subset), fall back per row to the
    zone-based heuristic so the app still renders.
    """
    P9 = _internal_P9()
    idx = list(df_int['sig_idx'])
    if P9 is not None and len(P9) >= len(df_int) and max(idx) < len(P9) and idx == list(range(len(df_int))):
        return [inf.conformal_set(P9[i], CONFORMAL_PROB_THR) for i in idx]
    return [_heuristic_conformal_set(r) for _, r in df_int.iterrows()]


def build_conformal_set(row):
    """Return the split-conformal prediction set for a patient (list of 9-class labels).

    Uses the precomputed real set (P9 ≥ CONFORMAL_PROB_THR) attached at load time; falls
    back to the zone heuristic only when no precomputed set is present.
    """
    cs = row.get('conformal_set') if hasattr(row, 'get') else None
    if isinstance(cs, (list, tuple)) and len(cs) > 0:
        return list(cs)
    return _heuristic_conformal_set(row)


def _heuristic_conformal_set(row):
    """Legacy zone/confidence heuristic — fallback only (DEMO internal, missing OOF)."""
    pred = row['pred_class']
    zone = row['zone']
    if zone == 'HIGH':
        return [pred]
    elif zone == 'MEDIUM':
        return sorted(set([pred, 'NEGATIVE']))
    else:
        candidates = {pred, 'NEGATIVE'}
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
