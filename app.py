"""
app.py – IFE M-Protein Clinical Decision Support Dashboard
3-Layer Architecture:
  Layer 1: Executive Summary (4-column, instant decision)
  Layer 2: Evidence (SHAP waterfalls + explanations)
  Layer 3: Deep Dive (spatial SHAP, interactive charts)
"""
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import (
    ZONE_COLORS, CP_ALPHA, pretty, reflex_group,
    INTENDED_USE_NOTICE, UPLOAD_MAX_BYTES, UPLOAD_MAX_SAMPLES,
)
from data_loader import (
    load_all_data, get_patient_signal, get_patient_shap,
    get_patient_shap_full, get_human_readable_parts,
    build_conformal_set, load_reflex_rules,
    attach_user_rows, get_user_result, USER_SOURCE,
)
from plotting import (
    plotly_signal_faceted, mpl_signal_combined,
    plotly_shap_waterfall, mpl_shap_waterfall,
    mpl_shap_waterfall_cumulative, mpl_shap_6channel_overlay,
    render_conformal_set_html,
)
from pdf_export import create_pdf
import inference as inf
from llm_interpret import (
    render_api_key_sidebar, render_interpretation_section,
    build_prompt, is_available, generate_template_interpretation,
    MODE_RESEARCH, MODE_CLINICAL,
)

# ═══════════════════════════════════════════════════════════
#  PAGE CONFIG & STYLING
# ═══════════════════════════════════════════════════════════
st.set_page_config(layout="wide", page_title="IFE M-Protein CDS")
st.markdown("""
<style>
    .block-container {
        padding: 0.75rem 1.5rem 1rem 1.5rem !important;
        max-width: 1100px !important;
        margin: 0 auto !important;
    }
    .stMainBlockContainer { max-width: 1100px !important; }
    /* The header is fixed-position by default; on newer Streamlit builds it overlaps the
       first element and clips the top of the intended-use banner. Taking it out of fixed
       positioning puts it back in normal flow, so the banner always renders in full. */
    header[data-testid="stHeader"] {
        height: 2rem !important;
        position: relative !important;
    }
    [data-testid="stExpander"] {
        border: 1px solid #E0E0E0 !important;
        border-radius: 6px !important;
        margin-bottom: 0.3rem !important;
    }
    h1 { margin: 0.1rem 0 !important; font-size: 1.7rem !important; }
    h4 { margin: 0.1rem 0 !important; }
    [data-testid="stMarkdownContainer"] p { margin-bottom: 0.15rem !important; }
    [data-testid="stHorizontalBlock"] { gap: 0.4rem !important; }
    hr { margin: 0.2rem 0 !important; }
    [data-testid="stAlert"] { padding: 0.3rem 0.6rem !important; margin-bottom: 0.2rem !important; }
    [data-testid="column"] { padding: 0 0.2rem !important; }
    [data-testid="stTabs"] button { font-size: 14px !important; font-weight: 600 !important; }
    [data-testid="stExpander"] h2, [data-testid="stExpander"] h3 {
        font-size: 1rem !important; margin: 0 !important;
    }
    /* The regulatory notice must stay on screen, not scroll away. Streamlit wraps every
       element in its own container, and a sticky box can only travel inside its parent's
       bounds — sticking the banner div itself does nothing, because its wrapper is exactly
       as tall as the banner. So the wrapper is what gets pinned: it is a direct child of
       the page-level vertical block, which spans the full scroll height. The opaque
       background keeps content from showing through as it scrolls underneath. */
    [data-testid="stElementContainer"]:has(.intended-use-banner) {
        position: sticky;
        top: 0;
        z-index: 999;
        background: #FFFFFF;
        padding-bottom: 0.15rem;
    }
    .intended-use-banner {
        background: #FFF4E5;
        border: 1px solid #E8A33D;
        border-left: 5px solid #C77700;
        border-radius: 5px;
        padding: 0.45rem 0.75rem;
        margin: 0 0 0.5rem 0;
        color: #5A3A00;
        font-size: 0.86rem;
        line-height: 1.35;
    }
    .intended-use-banner strong { color: #8A4B00; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  INTENDED-USE NOTICE (permanent, non-dismissible)
# ═══════════════════════════════════════════════════════════
def render_intended_use_banner():
    """Regulatory posture banner. Rendered once, above the tab bar, and pinned there.

    Deliberately plain HTML with no close control — it must not be dismissible. Because it
    sits above the tabs and is sticky, it stays visible on every tab and at every scroll
    position, so no per-tab copy is needed.
    """
    lead, _, rest = INTENDED_USE_NOTICE.partition(". ")
    st.markdown(
        f'<div class="intended-use-banner">⚠️ <strong>{lead}.</strong> {rest}</div>',
        unsafe_allow_html=True)


render_intended_use_banner()

# ═══════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════
ds, shap_d, feat_dict, master_df = load_all_data()

# Samples uploaded in this session are appended to a per-session copy of the cohort so
# they can be browsed alongside it. load_all_data() is cached and shared across sessions,
# so the uploaded rows must never be written into the object it returns.
master_df = attach_user_rows(master_df)

if "_reflex_cache" not in st.session_state:
    _up = st.session_state.get("_reflex_upload", None)
    _baseline, _matrix, _reflex_src = load_reflex_rules(uploaded_file=_up)
    st.session_state["_reflex_cache"] = (_baseline, _matrix, _reflex_src)
_baseline, _matrix, _reflex_src = st.session_state["_reflex_cache"]

# ═══════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.header("🧬 CDS Navigation")
    search_id = st.text_input("Direct ID Search (e.g. 7179609)")
    st.divider()

    st.subheader("Scenario Explorer")
    _sources = ["ALL", "Internal", "External"]
    if (master_df['source'] == 'Alicante').any():
        _sources.append("Alicante")
    if (master_df['source'] == USER_SOURCE).any():
        _sources.append(USER_SOURCE)
    f_source = st.selectbox("Data Source", _sources)
    f_type   = st.selectbox("M-Protein Type", ["ALL"] + sorted(master_df['pred_class'].unique().tolist()))
    f_zone   = st.selectbox("Confidence Zone", ["ALL", "HIGH", "MEDIUM", "LOW"])
    f_res    = st.selectbox("Result", ["ALL", "Correct", "Incorrect"])
    f_cpset  = st.selectbox("Conformal Set Size", ["ALL", "1 (Singleton)", "2+", "3+"])
    req_shap = st.toggle("Show only patients with L2/L3 SHAP", value=False)

    subset = master_df.copy()
    if f_source != "ALL":    subset = subset[subset['source'] == f_source]
    if f_type != "ALL":      subset = subset[subset['pred_class'] == f_type]
    if f_zone != "ALL":      subset = subset[subset['zone'] == f_zone]
    if f_res == "Correct":   subset = subset[subset['correct'] == 1]
    if f_res == "Incorrect": subset = subset[subset['correct'] == 0]

    if f_cpset != "ALL":
        subset['_cp_size'] = subset.apply(lambda r: len(build_conformal_set(r)), axis=1)
        if f_cpset == "1 (Singleton)": subset = subset[subset['_cp_size'] == 1]
        elif f_cpset == "2+":          subset = subset[subset['_cp_size'] >= 2]
        elif f_cpset == "3+":          subset = subset[subset['_cp_size'] >= 3]
        subset = subset.drop(columns=['_cp_size'])

    if req_shap and f_type != "NEGATIVE":
        valid_int  = set(shap_d.get('L2', {}).get('sample_indices', []))
        valid_ext  = set(shap_d.get('L2_external', {}).get('sample_indices', []))
        valid_alic = set(shap_d.get('L2_alicante', {}).get('sample_indices', []))
        # Uploaded samples always carry SHAP — it is computed live during inference rather
        # than looked up in the precomputed pickles, so this availability filter must not
        # drop them.
        subset = subset[
            (subset['source'] == USER_SOURCE) |
            ((subset['source'] == 'Internal') & (subset['sig_idx'].isin(valid_int))) |
            ((subset['source'] == 'External') & (subset['sig_idx'].isin(valid_ext)))  |
            ((subset['source'] == 'Alicante') & (subset['sig_idx'].isin(valid_alic)))
        ]

    st.success(f"Found **{len(subset)}** matches.")
    if not subset.empty:
        selected = st.selectbox("Select Patient:", subset.index.tolist())
        if not search_id:
            st.session_state.active_id = str(selected)
        if st.button("🎲 Random Patient"):
            st.session_state.active_id = str(subset.sample(1).index[0])
    else:
        st.warning("No patients match these criteria.")

    st.divider()
    mask_id   = st.toggle("Mask Patient ID", value=True)
    pub_mode  = st.toggle("📸 Publication Mode", value=False)
    # Off by default: the interpretation calls an external API, so it is opt-in per session.
    enable_ai = st.toggle("🤖 AI Interpretation", value=False)
    ai_mode   = st.radio(
        "Interpretation Mode",
        ["🔬 Research (GT visible)", "🏥 Clinical (GT hidden)"],
        index=0, horizontal=True,
    )
    llm_mode = MODE_RESEARCH if "Research" in ai_mode else MODE_CLINICAL
    # The container is created unconditionally so the sidebar always has the same number
    # of children. Rendering the API-key block directly under an `if` changed the element
    # count on every AI-Interpretation toggle, which shifted every widget below it; the
    # frontend could not match the old nodes to the new ones and left stale, greyed-out
    # duplicates of the Reflex Rules buttons on screen.
    _api_key_slot = st.container()
    if enable_ai and not pub_mode:
        with _api_key_slot:
            render_api_key_sidebar()

    st.divider()
    st.subheader("📋 Reflex Rules")
    uploaded_reflex = st.file_uploader("Upload custom reflex_matrix.xlsx", type=["xlsx"], key="_reflex_upload")
    col_reload, col_download = st.columns(2)
    with col_reload:
        if st.button("🔄 Reload", key="_reload_reflex"):
            st.session_state.pop("_reflex_cache", None)
            st.rerun()
    with col_download:
        from config import generate_reflex_template
        import io
        _rbuf = io.BytesIO()                       # built in memory — nothing hits disk
        generate_reflex_template(_rbuf)
        st.download_button("📥 Template", data=_rbuf.getvalue(), file_name="reflex_matrix.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ═══════════════════════════════════════════════════════════
#  MAIN PANEL
# ═══════════════════════════════════════════════════════════
def _check_model_integrity():
    """Verify every frozen artifact the upload path depends on. Returns models_ok.

    The per-file md5 tables used to be rendered here. They are no longer shown, but the
    check itself still runs and still gates the upload path — a mismatch has to disable
    inference, not merely go unreported.
    """
    try:
        models = inf.model_md5_report()
        inf.cascade_src_md5_report()
    except Exception:
        st.error("❌ Model integrity check failed — the frozen model artifacts could not be read. "
                 "The upload path is disabled. Precomputed cohort results are unaffected.")
        return False

    models_ok = all(ok for *_, ok in models)
    if not models_ok:
        st.error("❌ Model integrity check failed — the deployed model does not match the "
                 "published lineage. The upload path is disabled. Precomputed cohort results "
                 "are unaffected.")
    return models_ok


def _render_upload_section(feat_dict):
    """Upload de-identified signals → run the FROZEN cascade → same outputs."""
    # No banner here — the page-level one is sticky and sits above the tab bar, so it stays
    # on screen for this tab too. Rendering a second copy inside the tab only duplicated it.
    st.markdown(
        "Upload your own **de-identified** capillary immunotyping signals to evaluate the "
        "frozen model on external data — an independent validation path.")
    st.info(
        "🔒 **Data handling.** Upload **de-identified signal data only** — no names, no patient "
        "or accession numbers, no dates. The file is processed **transiently in memory**: it is "
        "**never written to disk, never logged, and never used for training**. Only the four "
        "signal columns are read; every other column, sheet and header field is discarded before "
        "processing. Results live only in your browser session and disappear when you close the "
        "tab or press *Clear uploaded data*. The deployed model is **frozen and md5-verified** "
        "(5-fold XGBoost-Peak-Optuna ensemble; forward inference only).")
    st.caption(
        "**Format — long-format Excel (.xlsx):** columns `sample_id, curve_name, x, y`. "
        "Per sample: exactly **6 curves** (ELP, IgG, IgA, IgM, Kappa, Lambda) × **300 points** "
        f"(`x` = 0…299). One row per (curve, point). Up to **{UPLOAD_MAX_SAMPLES} samples** and "
        f"**{UPLOAD_MAX_BYTES // (1024 * 1024)} MB** per file. `sample_id` is your own label — "
        "use a study code, never a patient identifier.")

    models_ok = _check_model_integrity()

    import io
    _tmpl = inf.example_signal_df()
    _buf = io.BytesIO(); _tmpl.to_excel(_buf, index=False, sheet_name="signals")
    st.download_button("📄 Download example template (.xlsx)", _buf.getvalue(),
                       file_name="cds_upload_template.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       help="One fully-filled example sample (EXAMPLE_001) showing the exact format.")

    if st.session_state.get("_upload_cache"):
        if st.button("🗑️ Clear uploaded data", key="_clear_upload",
                     help="Discard the uploaded signals and all derived results from this session."):
            st.session_state.pop("_upload_cache", None)
            st.session_state.pop("_ext_upload_sel", None)
            st.rerun()

    if not models_ok:
        return

    up = st.file_uploader("De-identified signal Excel (.xlsx)", type=["xlsx", "xls"],
                          key="_ext_upload_file")
    if up is None:
        return
    # Session cache key: hash of (name, size) so the filename itself is never retained.
    import hashlib
    sig_key = hashlib.sha256(f"{up.name}:{up.size}".encode()).hexdigest()[:16]
    cache = st.session_state.setdefault("_upload_cache", {})
    if sig_key not in cache:
        try:
            X, ids = inf.parse_long_format_excel(up)   # validates cols, 6 curves, 300 pts, numeric y
        except inf.UploadError as e:
            # UploadError messages are fixed strings by contract — they never echo file content.
            st.error(f"❌ Upload rejected: {e}")
            return
        except Exception:
            st.error("❌ The file could not be processed. Please check it against the example "
                     "template above and try again.")
            return
        st.caption(f"✓ Validated: {len(ids)} sample(s), 6 curves × 300 points each, numeric signals.")
        with st.spinner(f"Running the frozen cascade on {len(ids)} sample(s)…"):
            try:
                results, _ = inf.run_frozen_cascade(X, inf.CONFORMAL_PROB_THR_DEFAULT, with_shap=True)
            except inf.ModelIntegrityError:
                st.error("❌ Model integrity check failed — inference was not run.")
                return
            except Exception:
                st.error("❌ Inference could not be completed for this file.")
                return
        # One upload per session: a new file replaces the previous one rather than
        # accumulating, so the cohort view never mixes two uploads.
        cache.clear()
        cache[sig_key] = {"ids": ids, "X": X, "results": results}
        # The cohort list is assembled at the top of the script, which has already run by
        # the time we get here — without a rerun the new samples would not appear under
        # Data Source until the next unrelated interaction.
        st.session_state["_upload_completed"] = len(ids)
        st.rerun()

    data = cache[sig_key]
    ids, X, results = data["ids"], data["X"], data["results"]

    _just_done = st.session_state.pop("_upload_completed", None)
    if _just_done:
        st.success(
            f"✅ Upload complete — {_just_done} sample(s) evaluated with the frozen model "
            f"and added to the cohort list as source **“{USER_SOURCE}”**. To open a full "
            f"clinical report for one of them, use the sidebar: **Data Source → "
            f"{USER_SOURCE}**, then pick it under *Select Patient*.")
    else:
        st.success(f"✅ {len(ids)} sample(s) evaluated with the frozen model.")

    # summary table + download
    summ = pd.DataFrame([{
        "sample_id": ids[i], "prediction": pretty(r["pred_class"]),
        "confidence": round(r["confidence"], 3), "zone": r["zone"],
        "conformal_set": " ; ".join(pretty(c) for c in r["conformal_set"]),
        "set_size": r["conformal_set_size"],
        "action": "Requires expert review" if r["review_required"] else "Auto-verify",
    } for i, r in enumerate(results)])
    st.dataframe(summ, use_container_width=True, hide_index=True)
    st.download_button("📥 Download results (CSV)", summ.to_csv(index=False).encode(),
                       file_name="cds_upload_results.csv", mime="text/csv")

    sel = st.selectbox("Inspect sample", list(range(len(ids))),
                       format_func=lambda i: f"{ids[i]} — {pretty(results[i]['pred_class'])}",
                       key="_ext_upload_sel")
    r = results[sel]
    a, b, c = st.columns([1, 1, 1.2])
    with a:
        st.metric("Prediction", pretty(r["pred_class"]))
        st.caption(f"L1 p={r['l1_proba']:.3f}")
    with b:
        st.metric("Confidence zone", r["zone"], f"conf {r['confidence']:.3f}")
    with c:
        st.markdown("**Conformal set** " + " ".join(f"`{pretty(x)}`" for x in r["conformal_set"]))
        if r["review_required"]:
            st.warning(f"Set = {r['conformal_set_size']} → requires expert review")
        else:
            st.success("Singleton → auto-verify")

    # Explicit keys throughout: Streamlit derives a chart's identity from its contents, so
    # the same uploaded sample rendered here and in the patient report — which is now
    # possible, since uploads join the cohort list — would collide on one auto-generated id.
    st.plotly_chart(plotly_signal_faceted(X[sel], title=f"Signal — {ids[sel]}"),
                    use_container_width=True, key=f"upload_signal_{sel}")

    if r.get("shap"):
        with st.expander("Model decision support (SHAP top features)", expanded=False):
            st.info("⚠ SHAP attributions are **regional** (fixed β1/β2/β2–γ/γ windows from an "
                    "averaged training tracing), not exact peak coordinates.")
            for lv in ("L1", "L2", "L3"):
                sd = r["shap"].get(lv)
                st.markdown(f"**{lv}**")
                if not isinstance(sd, list):
                    st.caption("Not evaluated at this level."); continue
                st.plotly_chart(plotly_shap_waterfall(sd, lv), use_container_width=True,
                                key=f"upload_shap_{sel}_{lv}")


# ═══════════════════════════════════════════════════════════
#  TOP-LEVEL LAYOUT
#  Two tabs: the per-patient report and the patient-independent
#  upload path. Both tab containers are created up front so the
#  upload tab is already populated before the patient guard below
#  can stop the script — the upload path must stay reachable
#  without selecting a patient.
# ═══════════════════════════════════════════════════════════
tab_patient, tab_upload = st.tabs(
    ["🧬 Patient analysis", "📤 External data upload"])

with tab_upload:
    _render_upload_section(feat_dict)

target_id = search_id if search_id else st.session_state.get('active_id')
if not target_id:
    with tab_patient:
        st.info("Search for a patient or use the Scenario Explorer to begin.")
    st.stop()

with tab_patient:
    try:
        idx_str = str(target_id)
        row = master_df.loc[idx_str]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        is_ext  = row['source'] == 'External'
        shap_suffix = {'External': '_external', 'Alicante': '_alicante'}.get(row['source'], '')
        sig_idx = row['sig_idx']
        signal  = get_patient_signal(ds, row)
        disp_id = "XXXX" + idx_str[4:] if mask_id else idx_str

        # ── Probabilities ──
        def _get_prob(r, *keys):
            for k in keys:
                v = r.get(k, None)
                if v is not None and pd.notnull(v) and float(v) != 0.0:
                    return f"{float(v):.4f}"
            for k in keys:
                v = r.get(k, None)
                if v is not None and pd.notnull(v):
                    return f"{float(v):.4f}"
            return "—"

        p1 = _get_prob(row, 'p_L1', 'l1_prob', 'l1_proba', 'prob_L1', 'L1_prob')
        p2 = _get_prob(row, 'p_L2', 'l2_prob', 'l2_proba', 'prob_L2', 'L2_prob')
        p3 = _get_prob(row, 'p_L3', 'l3_prob', 'l3_proba', 'prob_L3', 'L3_prob')

        cp_set   = build_conformal_set(row)
        set_size = len(cp_set)
        # Operating rule (Task 2): conformal set size ≥ 2 ⇒ requires expert review (not auto-verified).
        review_required = set_size >= 2
        cp_action = "Auto-verify" if set_size == 1 else "Requires expert review"

        # ── SHAP ──
        # An uploaded sample has no entry in the precomputed SHAP pickles — its SHAP was
        # computed live during inference and travels with the result. Reading shap_d with
        # its sig_idx would silently show a cohort patient's explanation instead.
        is_user = row['source'] == USER_SOURCE
        user_res = get_user_result(row) if is_user else None
        if is_user:
            _ushap = (user_res or {}).get('shap') or {}
            shap_l1 = _ushap.get('L1', "MISSING_KEY")
            shap_l2 = _ushap.get('L2', "MISSING_KEY")
            shap_l3 = _ushap.get('L3', "MISSING_KEY")
        else:
            shap_l1 = get_patient_shap(shap_d, 'L1', sig_idx, shap_suffix)
            shap_l2 = get_patient_shap(shap_d, 'L2', sig_idx, shap_suffix)
            shap_l3 = get_patient_shap(shap_d, 'L3', sig_idx, shap_suffix)

        # ── Base values ──
        def _get_base(level_key):
            level_data = shap_d.get(level_key, {})
            if isinstance(level_data, dict) and 'base_value' in level_data:
                bv = level_data['base_value']
                if hasattr(bv, '__len__'):
                    if len(bv) == 1: return float(bv[0])
                    if len(bv) == 4:
                        l2_map = {'IGG': 0, 'IGA': 1, 'IGM': 2, 'FREE': 3}
                        pred = row['pred_class'].split('_')[0] if '_' in row['pred_class'] else row['pred_class']
                        return float(bv[l2_map.get(pred, 0)])
                return float(bv)
            return shap_d.get('base_values', {}).get(level_key, None)

        # No base values for an upload: shap_suffix is '' for it, so these lookups would
        # read the internal cohort's baselines and attach them to someone else's sample.
        base_vals = {'L1': None, 'L2': None, 'L3': None} if is_user else {
            'L1': _get_base('L1' + shap_suffix),
            'L2': _get_base('L2' + shap_suffix),
            'L3': _get_base('L3' + shap_suffix),
        }

        # ── Reflex ──
        grp = reflex_group(row['pred_class'])
        entry = _matrix.get(grp, {}).get(row['zone'], _matrix.get(grp, {}).get('LOW', {}))
        gel_ife = entry.get('gel_ife', 'N/A')
        extra_tests = entry.get('tests', [])
        guidance = entry.get('guidance', '')
        show_baseline = (grp != 'NEGATIVE')

        # ═════════════════════════════════════════════════
        #  HEADER (before tabs — always visible)
        # ═════════════════════════════════════════════════
        # An upload has no reference interpretation, so there is nothing to score it
        # against — show it plainly instead of rendering a red "incorrect" GT badge.
        has_gt = (not is_user) and pd.notnull(row.get('true_class'))
        true_color = "#1A9641" if row['correct'] == 1 else "#D73027"
        if not has_gt:
            st.markdown(
                f"**Analysis Report: {disp_id}** &nbsp;|&nbsp; "
                f"<span style='color:#666;'>no reference interpretation</span> "
                f"&nbsp;|&nbsp; {row['source']}",
                unsafe_allow_html=True,
            )
        elif llm_mode == MODE_RESEARCH:
            st.markdown(
                f"**Analysis Report: {disp_id}** &nbsp;|&nbsp; "
                f"<span style='color:{true_color}; font-weight:bold;'>"
                f"[GT: {pretty(row['true_class'])}]</span> &nbsp;|&nbsp; {row['source']}",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f"**Analysis Report: {disp_id}** &nbsp;|&nbsp; {row['source']}")

        # ═════════════════════════════════════════════════
        #  TABS (immediately visible)
        # ═════════════════════════════════════════════════
        tab_report, tab_rules, tab_debug = st.tabs(["📋 Clinical Report", "📋 Reflex Rules", "🔧 Debug"])

        # ─────────────────────────────────────────────────
        #  TAB 1: CLINICAL REPORT
        # ─────────────────────────────────────────────────
        with tab_report:

            # ── 1-4: Executive Summary (single row) ──
            c1, c2, c3, c4 = st.columns([1.2, 1, 0.8, 1.4])

            with c1:
                gt_html = ""
                if has_gt and llm_mode == MODE_RESEARCH:
                    _ok = row['correct'] == 1
                    _gt_col = "#1A9641" if _ok else "#D73027"
                    _mark = "✓ correct" if _ok else "✗ incorrect"
                    gt_html = (
                        f"<div style='margin-top:5px; padding:3px 8px; border-radius:4px; "
                        f"background:{_gt_col}1A; border-left:3px solid {_gt_col};'>"
                        f"<span style='font-size:10px; color:#888; font-weight:600;'>GROUND TRUTH</span><br>"
                        f"<span style='font-size:18px; font-weight:bold; color:{_gt_col};'>{pretty(row['true_class'])}</span> "
                        f"<span style='font-size:12px; font-weight:600; color:{_gt_col};'>{_mark}</span></div>"
                    )
                st.markdown(
                    f"<div>"
                    f"<span style='font-size:11px; color:#888; font-weight:600;'>1. CLASSIFICATION</span><br>"
                    f"<span style='font-size:32px; font-weight:bold; color:#222; line-height:1.1;'>{pretty(row['pred_class'])}</span>"
                    f"</div>"
                    f"{gt_html}"
                    f"<div style='font-size:11px; color:#555; line-height:1.5; margin-top:5px;'>"
                    f"L1: p={p1}<br>L2: p={p2}<br>L3: p={p3}<br>"
                    f"<b>Conf: {row['confidence']:.4f}</b></div>",
                    unsafe_allow_html=True,
                )

            with c2:
                st.markdown(
                    "<span style='font-size:11px; color:#888; font-weight:600;'>2. CONFORMAL SET "
                    f"<span style='color:#555;'>(n={set_size})</span></span>",
                    unsafe_allow_html=True,
                )
                st.markdown(render_conformal_set_html(cp_set, row['pred_class'], row['zone']), unsafe_allow_html=True)
                if review_required:
                    st.markdown(
                        "<div style='margin-top:3px; padding:2px 7px; border-radius:4px; background:#FDECEA; "
                        "border-left:3px solid #D7503A; font-size:11px; color:#8B2A1E; font-weight:600;'>"
                        "⚠ Set ≥ 2 → requires expert review</div>", unsafe_allow_html=True)

            with c3:
                zone_col = ZONE_COLORS.get(row['zone'], '#000')
                st.markdown(
                    f"<span style='font-size:11px; color:#888; font-weight:600;'>3. ZONE</span><br>"
                    f"<span style='font-size:26px; font-weight:bold; color:{zone_col}; line-height:1.1;'>{row['zone']}</span>",
                    unsafe_allow_html=True,
                )
                (st.success if set_size == 1 else st.warning)(f"{set_size}-class — {cp_action}")
                st.caption("Zone = compound confidence (calibrated); routing decision follows the conformal set size.")

            with c4:
                ife_colors = {"Not required": "#1A9641", "Consider": "#F46D43", "Recommended": "#E65100", "Mandatory": "#D73027"}
                ife_col = ife_colors.get(gel_ife, "#333")
                short_guidance = guidance[:120] + '…' if len(guidance) > 120 else guidance
                baseline_note = (
                    "<div style='background:#FFF8E1; border-left:2px solid #F9A825; "
                    "padding:2px 6px; border-radius:3px; font-size:10px; color:#5D4037; margin-top:3px;'>"
                    "📋 Baseline Panel → Reflex Rules tab</div>"
                ) if show_baseline else ""

                st.markdown(
                    f"<span style='font-size:11px; color:#888; font-weight:600;'>4. REFLEX</span><br>"
                    f"<span style='font-size:11px;'><b>Gel IFE:</b> "
                    f"<span style='color:{ife_col}; font-weight:bold;'>{gel_ife}</span></span><br>"
                    f"<span style='font-size:10px; color:#555;'>{short_guidance}</span>"
                    f"{baseline_note}",
                    unsafe_allow_html=True,
                )
                if extra_tests:
                    st.markdown(
                        "<span style='font-size:10px; color:#444;'>" +
                        " · ".join(extra_tests[:3]) + "</span>",
                        unsafe_allow_html=True,
                    )

            st.caption(f"ℹ️ Rules: {_reflex_src.replace('📁 ', '').replace('📤 ', '').replace('⚠️ ', '')}")
            st.divider()

            # ── 5: Signal Trace ──
            with st.expander("5. Signal Trace (6-Channel)", expanded=True):
                st.plotly_chart(plotly_signal_faceted(signal, title=f"Signal — {disp_id}"),
                                use_container_width=True, key=f"report_signal_{idx_str}")

            # ── 6: AI Interpretation ──
            interp_text, interp_meta = None, None
            if enable_ai:
                with st.expander("6. AI Clinical Interpretation", expanded=True):
                    interp_text, interp_meta = render_interpretation_section(
                        row, cp_set, shap_l1, shap_l2, shap_l3, feat_dict, mode=llm_mode,
                        publication_mode=pub_mode,
                    )

            # ── 7: SHAP Waterfalls ──
            with st.expander("7. Model Decision Support (SHAP)", expanded=True):
                pred_pretty = pretty(row['pred_class'])
                shap_levels = [
                    ("L1 (Binary): POSITIVE", shap_l1, base_vals.get('L1')),
                    ("L2 (Heavy): " + pred_pretty.split('-')[0], shap_l2, base_vals.get('L2')),
                    ("L3 (Light): " + (pred_pretty.split('-')[-1] if '-' in pred_pretty else ""), shap_l3, base_vals.get('L3')),
                ]
                for level_title, shap_data, base_val in shap_levels:
                    if isinstance(shap_data, str) and 'not available' in shap_data.lower():
                        continue
                    col_chart, col_explain = st.columns([3, 2])
                    with col_chart:
                        fig_wf = mpl_shap_waterfall_cumulative(shap_data, level_title, base_value=base_val)
                        st.pyplot(fig_wf, use_container_width=True)
                        plt.close(fig_wf)
                    with col_explain:
                        if isinstance(shap_data, str) or not shap_data:
                            st.caption("No SHAP data available.")
                        else:
                            top3 = sorted(shap_data, key=lambda x: abs(x[2]), reverse=True)[:3]
                            for feat, val, sv in top3:
                                _, paragraph, _ = get_human_readable_parts(feat, val, sv, feat_dict)
                                color = "#B2182B" if sv >= 0 else "#2166AC"
                                short = paragraph[:120] + '...' if len(paragraph) > 120 else paragraph
                                safe_full = paragraph.replace("'", "&#39;").replace('"', '&quot;')
                                st.markdown(
                                    f"<div style='border-left:3px solid {color}; padding:3px 8px; margin-bottom:6px; cursor:help;' "
                                    f"title='{safe_full}'>"
                                    f"<b style='color:{color}; font-size:12px;'>{feat}</b> = {val:.2f} → {sv:+.3f}<br>"
                                    f"<span style='font-size:11px; color:#555;'>{short}</span></div>",
                                    unsafe_allow_html=True,
                                )
                    st.divider()

            # ── Deep Dive Expanders ──
            with st.expander("🔬 6-Channel Spatial SHAP Overlay", expanded=False):
                st.caption("Red = pushes toward class, blue = pushes against.")
                st.info(
                    "⚠ **Attributions are REGIONAL, not exact peak coordinates.** The 399 features use "
                    "**fixed** electrophoretic region windows (β1 / β2 / β2–γ transition / γ) taken from an "
                    "**averaged** training tracing — not per-sample peak detection. Read the overlay at the "
                    "level of these zones, not individual x-positions.")
                for lv, lv_label in [('L1', 'L1: Binary'), ('L2', 'L2: Heavy'), ('L3', 'L3: Light')]:
                    full = get_patient_shap_full(shap_d, lv, sig_idx, shap_suffix)
                    if full is None:
                        st.caption(f"{lv_label}: not available"); continue
                    fn, sv, xv = full
                    fig_ov = mpl_shap_6channel_overlay(signal, fn, sv, level_title=lv_label)
                    st.pyplot(fig_ov, use_container_width=True)
                    plt.close(fig_ov)

            with st.expander("📊 Interactive Feature Ranking", expanded=False):
                s1, s2, s3 = st.columns(3)
                with s1: st.plotly_chart(plotly_shap_waterfall(shap_l1, "L1"), use_container_width=True,
                                         key=f"report_shap_L1_{idx_str}")
                with s2: st.plotly_chart(plotly_shap_waterfall(shap_l2, "L2"), use_container_width=True,
                                         key=f"report_shap_L2_{idx_str}")
                with s3: st.plotly_chart(plotly_shap_waterfall(shap_l3, "L3"), use_container_width=True,
                                         key=f"report_shap_L3_{idx_str}")

            # ── Export ──
            st.divider()
            include_ai_in_pdf = False
            if interp_text:
                include_ai_in_pdf = st.checkbox("Include AI interpretation in PDF", value=True)

            fig_sig_pdf = mpl_signal_combined(signal, title=f"Signal — {disp_id}")
            fig_s1_pdf  = mpl_shap_waterfall(shap_l1, "L1: Binary")
            fig_s2_pdf  = mpl_shap_waterfall(shap_l2, "L2: Heavy Chain")
            fig_s3_pdf  = mpl_shap_waterfall(shap_l3, "L3: Light Chain")

            pdf_buf = create_pdf(
                row=row, disp_id=disp_id, cp_set=cp_set, reflex_text="",
                p1=p1, p2=p2, p3=p3,
                fig_sig=fig_sig_pdf, fig_s1=fig_s1_pdf, fig_s2=fig_s2_pdf, fig_s3=fig_s3_pdf,
                shap_l1=shap_l1, shap_l2=shap_l2, shap_l3=shap_l3, feat_dict=feat_dict,
                ai_interpretation=interp_text if include_ai_in_pdf else None,
            )
            st.download_button("📥 Download PDF Report", data=pdf_buf,
                               file_name=f"CDS_Report_{disp_id}.pdf", mime="application/pdf", type="primary")
            for f in [fig_sig_pdf, fig_s1_pdf, fig_s2_pdf, fig_s3_pdf]:
                plt.close(f)

        # ─────────────────────────────────────────────────
        #  TAB 2: REFLEX RULES
        # ─────────────────────────────────────────────────
        with tab_rules:
            cur_grp = reflex_group(row['pred_class'])
            cur_zone = row['zone']
            st.info(f"Patient: **{pretty(row['pred_class'])}** → Group: **{cur_grp}** | Zone: **{cur_zone}**")

            st.markdown("#### Panel A: Universal Baseline")
            st.dataframe(pd.DataFrame(_baseline, columns=["Test", "Rationale"]),
                         use_container_width=True, hide_index=True)

            st.markdown("#### Panel B: Isotype × Zone Matrix")
            matrix_rows = []
            for mg, zones in _matrix.items():
                for zone, ent in zones.items():
                    is_current = (mg == cur_grp and zone == cur_zone)
                    matrix_rows.append({
                        "Group": mg, "Zone": zone,
                        "Gel IFE": ent.get("gel_ife", ""),
                        "Tests": "; ".join(ent.get("tests", [])) or "—",
                        "Guidance": ent.get("guidance", ""),
                        "★": "◄" if is_current else "",
                    })
            st.dataframe(pd.DataFrame(matrix_rows), use_container_width=True, hide_index=True, height=500,
                         column_config={"★": st.column_config.TextColumn(width="small"),
                                        "Group": st.column_config.TextColumn(width="small"),
                                        "Zone": st.column_config.TextColumn(width="small")})

            st.markdown("#### Panel C: Workflow")
            for z, act in {"HIGH": "Auto-verify & report", "MEDIUM": "Technician verification",
                           "LOW": "Expert referral + full SHAP"}.items():
                icon = "🟢" if z == "HIGH" else "🟡" if z == "MEDIUM" else "🔴"
                bold = " **◄**" if z == cur_zone else ""
                st.markdown(f"{icon} **{z}:** {act}{bold}")

        # ─────────────────────────────────────────────────
        #  TAB 3: DEBUG
        # ─────────────────────────────────────────────────
        with tab_debug:
            st.caption(f"Patient: {idx_str} | Source: {row['source']} | Mode: {ai_mode}")
            d1, d2 = st.columns(2)

            with d1:
                st.markdown("#### Raw Data")
                row_display = row.to_frame(name='Value').copy()
                for col in ['pred_class', 'true_class']:
                    if col in row.index:
                        row_display.loc[col, 'Pretty'] = pretty(row[col])
                st.dataframe(row_display, use_container_width=True, height=350)

            with d2:
                st.markdown("#### SHAP Coverage")
                for level, label in [('L1', 'L1'), ('L2', 'L2'), ('L3', 'L3')]:
                    key_used = f"{level}{shap_suffix}"
                    has_key = key_used in shap_d
                    in_index = False
                    if has_key:
                        si = np.array(shap_d[key_used].get('sample_indices', []))
                        in_index = sig_idx in si
                    icon = "🟢" if in_index else ("🟡" if has_key else "🔴")
                    st.markdown(f"{icon} **{label}** (`{key_used}`): {'Present' if in_index else 'Missing'}")

                st.divider()
                st.markdown("**Base Values:**")
                for lv, bv in base_vals.items():
                    st.markdown(f"`{lv}`: {bv}")

                st.divider()
                for label, data in [("L1", shap_l1), ("L2", shap_l2), ("L3", shap_l3)]:
                    if isinstance(data, str):
                        st.markdown(f"**{label}:** `{data}`")
                    elif data:
                        df_s = pd.DataFrame(data, columns=["Feature", "Value", "SHAP"])
                        df_s["SHAP"] = df_s["SHAP"].map(lambda x: f"{x:+.4f}")
                        st.markdown(f"**{label}:**")
                        st.dataframe(df_s, use_container_width=True, hide_index=True, height=180)

            with st.expander("LLM Prompt Inspector"):
                prompt_text = build_prompt(row, cp_set, shap_l1, shap_l2, shap_l3, feat_dict, mode=llm_mode)
                c1, c2 = st.columns(2)
                c1.metric("Words", f"{len(prompt_text.split()):,}")
                c2.metric("≈ Tokens", f"{int(len(prompt_text.split()) * 1.3):,}")
                st.code(prompt_text, language="text")

    except KeyError:
        st.error(f"Patient ID '{target_id}' not found.")
    except Exception as e:
        st.error(f"Error: {e}")
        import traceback
        st.code(traceback.format_exc())
