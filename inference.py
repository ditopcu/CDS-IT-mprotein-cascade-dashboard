"""
inference.py — single source of truth for the FROZEN cascade math used by the dashboard.

Nothing here retrains, refits, or recalibrates. It only:
  • wraps the frozen algorithm code vendored byte-identically in ./cascade_src
    (confidence.compute_cascade_confidence, features.extract_all_features,
     cascade.run_external_cascade),
  • computes the paper's compound confidence + zone (Task 1),
  • builds the study's split-conformal prediction set (Task 2), and
  • runs the frozen 5-fold ensemble end-to-end on user-uploaded signals (Task 4).

Frozen facts (see memory cclm-revision-m1):
  L1 threshold  τ  = 0.47219881415367126   (reported 0.47; the 0.47147757 in the old
                                             frozen config was a stale MiniRocket leftover)
  Conformal: split-conformal, LAC score = 1 − p(true), α = 0.05, calibrated on the CLEAN
             external cohort (seed 42, 70% calibration) → q_hat = 0.8165 → prob threshold
             1 − q_hat = 0.1835.  Prediction set = { class j : P9[j] ≥ 0.1835 }.
  Deployed model = 5-fold ENSEMBLE (per-level predict_proba averaging), never a single refit.

Models are loaded from ./pkl and md5-verified against config.MODEL_MD5 (the publication
lineage) before use; a mismatch raises ModelIntegrityError and the upload path is disabled.
"""
import os
import sys
import hashlib
import numpy as np

# numpy>=2 removed np.trapz (cascade features.py uses it) — shim before importing cascade_src.*
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

APP_ROOT   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(APP_ROOT, "pkl")          # publication-lineage models (md5-gated)

if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)
from cascade_src.constants import CLASS9_NAMES, L2_CLASSES, CHANNELS   # noqa: E402
from cascade_src.confidence import compute_cascade_confidence          # noqa: E402

# ── frozen constants ─────────────────────────────────────────────────────────
TAU = 0.47219881415367126                       # frozen L1 threshold
HEAVY_MAP = {"IGG": 0, "IGA": 1, "IGM": 2, "FREE": 3}
_C2I = {c: i for i, c in enumerate(CLASS9_NAMES)}
_NEG = _C2I["NEGATIVE"]
ZONE_HIGH, ZONE_MED = 0.70, 0.30
# Study conformal probability threshold (1 − q_hat). Frozen fallback; recomputed
# from the external cohort by conformal_threshold_from_external() when available.
CONFORMAL_PROB_THR_DEFAULT = 0.1835

PRETTY = {"IGG_KAPPA": "IgG-κ", "IGG_LAMBDA": "IgG-λ", "IGA_KAPPA": "IgA-κ",
          "IGA_LAMBDA": "IgA-λ", "IGM_KAPPA": "IgM-κ", "IGM_LAMBDA": "IgM-λ",
          "FREE_KAPPA": "FLC-κ", "FREE_LAMBDA": "FLC-λ", "NEGATIVE": "Negative"}


def zone_of(conf):
    return "HIGH" if conf >= ZONE_HIGH else ("MEDIUM" if conf >= ZONE_MED else "LOW")


# ── 9-class composite probability P9 (verbatim from revision/conformal_figure.py) ──
def build_p9(l1, l2, l3):
    """P9[(N,9)] from cascade components. l1(N,), l2(N,4)=[IGG,IGA,IGM,FREE], l3(N,)=P(lambda).

    P(class) = P(pos)·P(heavy|pos)·P(light|heavy); NEGATIVE = 1 − P(pos); row-normalized.
    Uses the frozen τ for the positive/negative branch.
    """
    l1 = np.asarray(l1, float); l2 = np.asarray(l2, float); l3 = np.asarray(l3, float)
    N = len(l1)
    P = np.zeros((N, 9))
    for i in range(N):
        p = float(l1[i])
        if p >= TAU:
            ll = float(l3[i]); lk = 1 - ll
            for hi, h in enumerate(L2_CLASSES):
                for lname, lv in (("KAPPA", lk), ("LAMBDA", ll)):
                    key = f"{h}_{lname}"
                    if key in _C2I:
                        P[i, _C2I[key]] = p * l2[i, hi] * lv
            P[i, _NEG] = 1 - p
        else:
            P[i, _NEG] = 1 - p
            for j in range(9):
                if j != _NEG:
                    P[i, j] = (p / 8 if p > 0 else 0)
        s = P[i].sum()
        P[i] /= s if s > 0 else 1
    return P


def conformal_threshold_from_external(P_ext, y_ext, alpha=0.05, cal_fraction=0.70, seed=42):
    """Split-conformal prob threshold (1 − q_hat) calibrated on the CLEAN external cohort.

    LAC non-conformity = 1 − P9[true]; q_hat = quantile(cal_scores, 1−alpha, 'higher').
    Deterministic (fixed seed) → matches the study (q_hat 0.8165, thr 0.1835).
    """
    y_ext = np.asarray([str(x) for x in y_ext])
    rng = np.random.RandomState(seed)
    idx = np.arange(len(y_ext)); rng.shuffle(idx)
    cal = idx[:int(cal_fraction * len(y_ext))]
    scores = np.array([1 - P_ext[i, _C2I[y_ext[i]]] if y_ext[i] in _C2I else 1.0 for i in cal])
    q_hat = np.quantile(scores, 1 - alpha, method="higher")
    return float(1 - q_hat), float(q_hat)


def build_internal_P9_from_oof(oof_dir):
    """Internal (2219,9) calibrated-cascade P9 from the frozen OOF probability pkls.

    OOF probabilities are the paper's out-of-fold predictions (no in-sample leakage);
    positive-only L2/L3 are scattered back to full length by pos_idx. Returns None on
    any failure (caller falls back to a heuristic set).
    """
    import pickle
    K = "XGBoost-Peak-Optuna"

    def _pick(d, *cands):                        # OOF file versions vary in the L1 model key
        for c in cands:
            if c in d:
                return d[c]
        raise KeyError(cands)
    try:
        l1o = pickle.load(open(os.path.join(oof_dir, "L1_oof_predictions.pkl"), "rb"))
        l2o = pickle.load(open(os.path.join(oof_dir, "L2_oof_predictions.pkl"), "rb"))
        l3o = pickle.load(open(os.path.join(oof_dir, "L3_oof_predictions.pkl"), "rb"))
        p1 = _pick(l1o["oof_proba"], K, "xgb_peak_optuna", "ensemble")
        p1 = p1[:, 1] if np.ndim(p1) == 2 else np.asarray(p1)
        pos = np.asarray(l2o["pos_idx"], int); N = len(p1)
        l2c = list(l2o.get("classes", l2o.get("class_names")))
        l3c = list(l3o.get("classes", l3o.get("class_names")))
        l2p = np.asarray(l2o["oof_prob"][K], float)
        l3pr = l3o["oof_proba"][K]; l3pr = l3pr[:, 1] if np.ndim(l3pr) == 2 else np.asarray(l3pr)
        l2_full = np.zeros((N, 4)); l2_full[pos] = l2p[:, [l2c.index(h) for h in L2_CLASSES]]
        l3_full = np.zeros(N);       l3_full[pos] = (l3pr if l3c.index("LAMBDA") == 1 else 1 - l3pr)
        return build_p9(p1, l2_full, l3_full)
    except Exception:
        return None


def conformal_set(P9_row, prob_thr):
    """Prediction set for one sample: {class : P9 ≥ thr}; if empty, {argmax}. Returns raw labels."""
    s = [CLASS9_NAMES[j] for j in range(9) if P9_row[j] >= prob_thr]
    if not s:
        s = [CLASS9_NAMES[int(np.argmax(P9_row))]]
    return s


def conformal_sets(P9, prob_thr):
    return [conformal_set(P9[i], prob_thr) for i in range(len(P9))]


# ── compound confidence + zone (Task 1) — canonical compute_cascade_confidence ──
def cohort_confidence(l1_proba, l2_pos, l3_pos, pos_idx):
    """Return (confidence[N], zone[N]) via the canonical formula.

    l1_proba (N,); l2_pos (P,4) and l3_pos (P,)=P(lambda) are the positive-only slices
    aligned to pos_idx (the samples that have L2/L3 available).
    """
    df = compute_cascade_confidence(np.asarray(l1_proba, float),
                                    np.asarray(l2_pos, float),
                                    np.asarray(l3_pos, float),
                                    TAU, np.asarray(pos_idx, int))
    conf = df["cascade_conf"].to_numpy(float)
    zone = np.array([zone_of(c) for c in conf])
    return conf, zone


# ═══════════════════════════════════════════════════════════════════════════════
#  UPLOAD PIPELINE (Task 4) — frozen forward inference on user-supplied signals
# ═══════════════════════════════════════════════════════════════════════════════
LANE_ALIASES = {   # accepted curve_name spellings → canonical lane
    "ELP": "ELP", "IGG": "IgG", "IGA": "IgA", "IGM": "IgM",
    "KAPPA": "Kappa", "K": "Kappa", "LAMBDA": "Lambda", "LAMDA": "Lambda", "L": "Lambda",
    "IgG": "IgG", "IgA": "IgA", "IgM": "IgM", "Kappa": "Kappa", "Lambda": "Lambda",
}
LANE_ORDER = ["ELP", "IgG", "IgA", "IgM", "Kappa", "Lambda"]  # dif channels built from these


class UploadError(ValueError):
    """Raised for malformed / non-conforming uploads (shown to the user, never a crash)."""


def _lanes_to_sample(lanes):
    """lanes: dict canonical-lane -> (300,) array → (6,300) [raw_ELP, dif_IgG..dif_Lambda]."""
    elp = lanes["ELP"]
    return np.array([elp,
                     elp - lanes["IgG"], elp - lanes["IgA"], elp - lanes["IgM"],
                     elp - lanes["Kappa"], elp - lanes["Lambda"]])


def example_signal_df():
    """One fully-filled SYNTHETIC example sample (no patient data) in the accepted long format.

    Downloadable template so users see the exact schema. ELP is a plausible proteinogram;
    the antiserum lanes equal ELP (difference channels = 0 → a benign 'negative' example).
    """
    import pandas as pd
    x = np.arange(300)

    def g(mu, sig, amp):
        return amp * np.exp(-0.5 * ((x - mu) / sig) ** 2)

    elp = (1 + g(40, 9, 1800) + g(68, 12, 240) + g(95, 15, 380)
             + g(150, 18, 300) + g(230, 32, 260))          # albumin, α1, α2, β, γ
    gamma = g(230, 32, 1.0)                                 # broad γ envelope (polyclonal Ig)
    # antiserum lanes = ELP with a small BROAD γ-region reduction (polyclonal immunosubtraction),
    # per-lane factors so difference channels are non-degenerate → a benign 'negative-like' example.
    frac = {"IgG": 0.30, "IgA": 0.10, "IgM": 0.08, "Kappa": 0.20, "Lambda": 0.18}
    lanes = {"ELP": np.round(elp, 1)}
    for name in LANE_ORDER[1:]:
        lanes[name] = np.round(elp * (1 - frac[name] * gamma), 1)
    rows = []
    for name in LANE_ORDER:
        for xi in range(300):
            rows.append({"sample_id": "EXAMPLE_001", "curve_name": name,
                         "x": int(xi), "y": float(lanes[name][xi])})
    return pd.DataFrame(rows)


# Reference format, repeated in every rejection message so the user can self-correct
# without the app ever quoting their file back at them.
_FORMAT_HINT =("Expected a long-format .xlsx with columns sample_id, curve_name, x, y; "
                "per sample exactly 6 curves (ELP, IgG, IgA, IgM, Kappa, Lambda) × 300 "
                "points (x = 0…299), numeric y. Use the example template above.")


def parse_long_format_excel(file_like, max_bytes=None, max_samples=None, max_rows=None):
    """Long-format signal Excel → (X_3d (N,6,300), sample_ids).

    Columns: sample_id, curve_name, x (0-299), y. Exactly 6 curves × 300 points per sample.

    PRIVACY CONTRACT:
      • Read fully in memory from the given buffer — nothing is written to disk.
      • Only the four signal columns are retained; every other column / sheet / header
        field (names, accession numbers, dates) is dropped before any downstream use.
      • Rejection messages are FIXED strings: they never echo a value, identifier, count
        or filename from the uploaded file.
    """
    import pandas as pd
    from config import UPLOAD_MAX_BYTES, UPLOAD_MAX_SAMPLES, UPLOAD_MAX_ROWS
    max_bytes   = UPLOAD_MAX_BYTES   if max_bytes   is None else max_bytes
    max_samples = UPLOAD_MAX_SAMPLES if max_samples is None else max_samples
    max_rows    = UPLOAD_MAX_ROWS    if max_rows    is None else max_rows

    size = getattr(file_like, "size", None)
    if size is not None and size > max_bytes:
        raise UploadError(f"File too large (limit {max_bytes // (1024 * 1024)} MB). "
                          "Please split the upload into smaller files.")
    try:
        raw = pd.read_excel(file_like)
    except Exception:
        raise UploadError("The file could not be read as an Excel workbook. " + _FORMAT_HINT)
    if len(raw) > max_rows:
        raise UploadError(f"Too many rows (limit {max_rows:,}). "
                          "Please split the upload into smaller files.")

    cols = {str(c).lower().strip(): c for c in raw.columns}
    need = ["sample_id", "curve_name", "x", "y"]
    if any(c not in cols for c in need):
        raise UploadError("Required column(s) missing. " + _FORMAT_HINT)
    df = raw[[cols["sample_id"], cols["curve_name"], cols["x"], cols["y"]]].copy()
    df.columns = ["sample_id", "curve_name", "x", "y"]           # discard all other columns
    del raw                                                       # drop identity-bearing columns
    df["lane"] = df["curve_name"].astype(str).str.strip().map(
        lambda s: LANE_ALIASES.get(s, LANE_ALIASES.get(s.upper())))
    if df["lane"].isna().any():
        raise UploadError("One or more curve_name values are not recognized. " + _FORMAT_HINT)
    if not np.issubdtype(df["y"].dtype, np.number):
        df["y"] = pd.to_numeric(df["y"], errors="coerce")
        if df["y"].isna().any():
            raise UploadError("Column 'y' must be numeric. " + _FORMAT_HINT)
    if df["sample_id"].nunique() > max_samples:
        raise UploadError(f"Too many samples in one file (limit {max_samples}). "
                          "Please split the upload into smaller files.")

    X, ids = [], []
    for sid, g in df.groupby("sample_id", sort=False):
        lanes = {}
        for lane, gl in g.groupby("lane"):
            gl = gl.sort_values("x")
            if len(gl) != 300:
                raise UploadError("A sample has a curve with the wrong number of points. "
                                  + _FORMAT_HINT)
            lanes[lane] = gl["y"].to_numpy(float)
        if any(l not in lanes for l in LANE_ORDER):
            raise UploadError("A sample is missing one or more of the 6 required curves. "
                              + _FORMAT_HINT)
        X.append(_lanes_to_sample(lanes)); ids.append(str(sid))
    if not X:
        raise UploadError("No complete samples found in the file. " + _FORMAT_HINT)
    return np.asarray(X), ids


_MODELS = None


class ModelIntegrityError(RuntimeError):
    """Raised when a model artifact's md5 does not match the published lineage."""


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_md5_report():
    """[(level, filename, expected_md5, actual_md5_or_None, ok)] for the integrity panel."""
    from config import MODEL_FILES, MODEL_MD5
    rows = []
    for lv in ("L1", "L2", "L3"):
        p = os.path.join(MODELS_DIR, MODEL_FILES[lv])
        actual = _md5(p) if os.path.exists(p) else None
        rows.append((lv, MODEL_FILES[lv], MODEL_MD5[lv], actual, actual == MODEL_MD5[lv]))
    return rows


def cascade_src_md5_report():
    """[(filename, expected_md5, actual_md5_or_None, ok)] for the vendored algorithm code."""
    from config import CASCADE_SRC_MD5
    src_dir = os.path.join(APP_ROOT, "cascade_src")
    rows = []
    for fn, exp in CASCADE_SRC_MD5.items():
        p = os.path.join(src_dir, fn)
        actual = _md5(p) if os.path.exists(p) else None
        rows.append((fn, exp, actual, actual == exp))
    return rows


def verify_models():
    """Raise ModelIntegrityError unless all three artifacts match the published lineage."""
    for lv, fn, exp, actual, ok in model_md5_report():
        if actual is None:
            raise ModelIntegrityError(f"model integrity check failed: {lv} artifact missing")
        if not ok:
            raise ModelIntegrityError(
                f"model integrity check failed: {lv} md5 {actual} != expected {exp}")


def _load_models():
    """Load the md5-verified 5-fold ensembles from pkl/. Verification happens BEFORE unpickling."""
    global _MODELS
    if _MODELS is None:
        import pickle
        from config import MODEL_FILES
        verify_models()
        _MODELS = tuple(
            pickle.load(open(os.path.join(MODELS_DIR, MODEL_FILES[lv]), "rb"))
            for lv in ("L1", "L2", "L3"))
    return _MODELS


def run_frozen_cascade(X_3d, prob_thr, with_shap=True):
    """Forward pass of the frozen 5-fold ensemble on (N,6,300) signals.

    Returns a list of per-sample dicts: pred, l1/l2/l3, P9, confidence, zone,
    conformal_set(+size), and (optional) SHAP top-feature lists per level.
    """
    from cascade_src.features import extract_all_features
    from cascade_src.cascade import run_external_cascade
    L1, L2, L3 = _load_models()
    feat = extract_all_features(np.asarray(X_3d, float), channels=CHANNELS, verbose=False)
    Xf = feat.values
    pred, l1p, l2p, l3p = run_external_cascade(L1, L2, L3, Xf, l1_threshold=TAU)

    pos_idx = np.where(l1p >= TAU)[0]
    conf, zone = cohort_confidence(l1p, l2p[pos_idx], l3p[pos_idx], pos_idx)
    P9 = build_p9(l1p, l2p, l3p)

    shap_by_level = _upload_shap(L1, L2, L3, Xf, list(feat.columns), pos_idx, pred) if with_shap else None

    out = []
    for i in range(len(X_3d)):
        cset = conformal_set(P9[i], prob_thr)
        out.append({
            "pred_class": pred[i],
            "l1_proba": float(l1p[i]),
            "l2_proba": l2p[i].tolist(),
            "l3_proba_lambda": float(l3p[i]),
            "confidence": float(conf[i]),
            "zone": zone[i],
            "P9": P9[i],
            "conformal_set": cset,
            "conformal_set_size": len(cset),
            "review_required": len(cset) >= 2,
            "shap": (shap_by_level[i] if shap_by_level else None),
        })
    return out, feat


def _upload_shap(L1, L2, L3, Xf, feat_names, pos_idx, pred, top_n=8):
    """TreeSHAP (5-fold averaged) top features per level for each uploaded sample. Best-effort."""
    try:
        import shap
    except Exception:
        return None

    def fold_mean(models, X):
        return np.mean([np.asarray(shap.TreeExplainer(m).shap_values(X)) for m in models], axis=0)

    def topk(sv_row, xv_row):
        order = np.argsort(np.abs(sv_row))[::-1][:top_n]
        return [(feat_names[j], float(xv_row[j]), float(sv_row[j])) for j in order]

    try:
        l1_sv = fold_mean(L1, Xf)                                    # (N,F)
        res = [{"L1": topk(l1_sv[i], Xf[i]), "L2": "NOT_IN_INDEX", "L3": "NOT_IN_INDEX"}
               for i in range(len(Xf))]
        if len(pos_idx):
            Xp = Xf[pos_idx]
            l2_arr = fold_mean(L2, Xp)                               # (P,F,4) or (P,F)
            l3_sv = fold_mean(L3, Xp)                                # (P,F)
            for k, i in enumerate(pos_idx):
                hi = HEAVY_MAP.get(pred[i].split("_")[0], 0)
                l2row = l2_arr[k, :, hi] if l2_arr.ndim == 3 else l2_arr[k]
                res[i]["L2"] = topk(l2row, Xp[k])
                res[i]["L3"] = topk(l3_sv[k], Xp[k])
        return res
    except Exception:
        return None
