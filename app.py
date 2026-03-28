"""
app.py – IFE M-Protein Clinical Decision Support Dashboard
3-Layer Architecture:
  Layer 1: Executive Summary (4-column, instant decision)
  Layer 2: Evidence (SHAP waterfalls + explanations)
  Layer 3: Deep Dive (spatial SHAP, interactive charts)
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import ZONE_COLORS, CP_ALPHA, pretty, reflex_group
from data_loader import (
    load_all_data, get_patient_signal, get_patient_shap,
    get_patient_shap_full, get_human_readable_parts,
    build_conformal_set, load_reflex_rules,
)
from plotting import (
    plotly_signal_faceted, mpl_signal_combined,
    plotly_shap_waterfall, mpl_shap_waterfall,
    mpl_shap_waterfall_cumulative, mpl_shap_6channel_overlay,
    render_conformal_set_html,
)
from pdf_export import create_pdf
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
        padding: 0.5rem 1.5rem 1rem 1.5rem !important;
        max-width: 1100px !important;
        margin: 0 auto !important;
    }
    .stMainBlockContainer { max-width: 1100px !important; }
    header[data-testid="stHeader"] { height: 2rem !important; }
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
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════
ds, shap_d, feat_dict, master_df = load_all_data()

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
    f_source = st.selectbox("Data Source", ["ALL", "Internal", "External"])
    f_type   = st.selectbox("M-Protein Type", ["ALL"] + sorted(master_df['pred_class'].unique().tolist()))
    f_zone   = st.selectbox("Confidence Zone", ["ALL", "HIGH", "MEDIUM", "LOW"])
    f_res    = st.selectbox("Result", ["ALL", "Correct", "Incorrect"])
    f_cpset  = st.selectbox("Conformal Set Size", ["ALL", "1 (Singleton)", "2+", "3+"])
    req_shap = st.toggle("Show only patients with L2/L3 SHAP", value=True)

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
        valid_int = set(shap_d.get('L2', {}).get('sample_indices', []))
        valid_ext = set(shap_d.get('L2_external', {}).get('sample_indices', []))
        subset = subset[
            ((subset['source'] == 'Internal') & (subset['sig_idx'].isin(valid_int))) |
            ((subset['source'] == 'External') & (subset['sig_idx'].isin(valid_ext)))
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
    enable_ai = st.toggle("🤖 AI Interpretation", value=True)
    ai_mode   = st.radio(
        "Interpretation Mode",
        ["🔬 Research (GT visible)", "🏥 Clinical (GT hidden)"],
        index=0, horizontal=True,
    )
    llm_mode = MODE_RESEARCH if "Research" in ai_mode else MODE_CLINICAL
    if enable_ai and not pub_mode:
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
        import tempfile, os
        tmp_path = os.path.join(tempfile.gettempdir(), "reflex_matrix_template.xlsx")
        generate_reflex_template(tmp_path)
        with open(tmp_path, "rb") as f:
            st.download_button("📥 Template", data=f, file_name="reflex_matrix.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ═══════════════════════════════════════════════════════════
#  MAIN PANEL
# ═══════════════════════════════════════════════════════════
target_id = search_id if search_id else st.session_state.get('active_id')
if not target_id:
    st.info("Search for a patient or use the Scenario Explorer to begin.")
    st.stop()

try:
    idx_str = str(target_id)
    row = master_df.loc[idx_str]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    is_ext  = row['source'] == 'External'
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
    cp_action = "Auto-reportable" if set_size == 1 else "Manual review"

    # ── SHAP ──
    shap_l1 = get_patient_shap(shap_d, 'L1', sig_idx, is_ext)
    shap_l2 = get_patient_shap(shap_d, 'L2', sig_idx, is_ext)
    shap_l3 = get_patient_shap(shap_d, 'L3', sig_idx, is_ext)

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

    base_vals = {
        'L1': _get_base('L1_external' if is_ext else 'L1'),
        'L2': _get_base('L2_external' if is_ext else 'L2'),
        'L3': _get_base('L3_external' if is_ext else 'L3'),
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
    true_color = "#1A9641" if row['correct'] == 1 else "#D73027"
    if llm_mode == MODE_RESEARCH:
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
            st.markdown(
                f"<div>"
                f"<span style='font-size:11px; color:#888; font-weight:600;'>1. CLASSIFICATION</span><br>"
                f"<span style='font-size:32px; font-weight:bold; color:#222; line-height:1.1;'>{pretty(row['pred_class'])}</span>"
                f"</div>"
                f"<div style='font-size:11px; color:#555; line-height:1.5;'>"
                f"L1: p={p1}<br>L2: p={p2}<br>L3: p={p3}<br>"
                f"<b>Conf: {row['confidence']:.4f}</b></div>",
                unsafe_allow_html=True,
            )

        with c2:
            st.markdown(
                "<span style='font-size:11px; color:#888; font-weight:600;'>2. CONFORMAL SET</span>",
                unsafe_allow_html=True,
            )
            st.markdown(render_conformal_set_html(cp_set, row['pred_class'], row['zone']), unsafe_allow_html=True)

        with c3:
            zone_col = ZONE_COLORS.get(row['zone'], '#000')
            st.markdown(
                f"<span style='font-size:11px; color:#888; font-weight:600;'>3. ZONE</span><br>"
                f"<span style='font-size:26px; font-weight:bold; color:{zone_col}; line-height:1.1;'>{row['zone']}</span>",
                unsafe_allow_html=True,
            )
            (st.success if set_size == 1 else st.warning)(f"{set_size}-class — {cp_action}")

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
            st.plotly_chart(plotly_signal_faceted(signal, title=f"Signal — {disp_id}"), use_container_width=True)

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
            for lv, lv_label in [('L1', 'L1: Binary'), ('L2', 'L2: Heavy'), ('L3', 'L3: Light')]:
                full = get_patient_shap_full(shap_d, lv, sig_idx, is_ext)
                if full is None:
                    st.caption(f"{lv_label}: not available"); continue
                fn, sv, xv = full
                fig_ov = mpl_shap_6channel_overlay(signal, fn, sv, level_title=lv_label)
                st.pyplot(fig_ov, use_container_width=True)
                plt.close(fig_ov)

        with st.expander("📊 Interactive Feature Ranking", expanded=False):
            s1, s2, s3 = st.columns(3)
            with s1: st.plotly_chart(plotly_shap_waterfall(shap_l1, "L1"), use_container_width=True)
            with s2: st.plotly_chart(plotly_shap_waterfall(shap_l2, "L2"), use_container_width=True)
            with s3: st.plotly_chart(plotly_shap_waterfall(shap_l3, "L3"), use_container_width=True)

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
                key_used = f"{level}_external" if is_ext else level
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