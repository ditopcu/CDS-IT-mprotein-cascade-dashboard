"""
llm_interpret.py – LLM-powered clinical interpretation of SHAP explanations.

Two modes:
  • Research Mode  – Ground truth known; LLM can discuss correctness, error analysis
  • Clinical Mode  – Ground truth hidden; LLM acts as real-world CDS assistant

API key fallback: st.secrets → os.environ → sidebar input
Graceful: returns None if no key or call fails (never crashes app)
"""
import os
import time
import hashlib
import streamlit as st

from config import pretty
from data_loader import get_human_readable_parts

# ─── Constants ────────────────────────────────────────────
MODE_RESEARCH = "research"
MODE_CLINICAL = "clinical"


# ═══════════════════════════════════════════════════════════
#  API Key Management
# ═══════════════════════════════════════════════════════════

def _resolve_api_key():
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY", None)
        if key: return key
    except Exception:
        pass
    key = os.environ.get("ANTHROPIC_API_KEY", None)
    if key: return key
    return st.session_state.get("_anthropic_api_key", None)


def render_api_key_sidebar():
    key = _resolve_api_key()
    if key:
        st.sidebar.success("🔑 API key configured")
        return True
    with st.sidebar.expander("🔑 AI Interpretation API Key", expanded=False):
        st.caption("Enter your Anthropic API key for AI interpretations.")
        user_key = st.text_input("API Key", type="password", key="_api_key_input", placeholder="sk-ant-...")
        if user_key:
            st.session_state["_anthropic_api_key"] = user_key
            st.success("Key saved for this session")
            return True
        st.info("Without a key, template-based explanations are shown.")
    return False


def is_available():
    return _resolve_api_key() is not None


# ═══════════════════════════════════════════════════════════
#  Prompt Construction
# ═══════════════════════════════════════════════════════════

def _build_shap_summary(shap_data, level_name, feat_dict):
    if isinstance(shap_data, str) or not shap_data:
        return f"{level_name}: Not evaluated at this level."
    lines = []
    for feat, val, sv in shap_data:
        _, paragraph, _ = get_human_readable_parts(feat, val, sv, feat_dict)
        direction = "INCREASES" if sv >= 0 else "DECREASES"
        lines.append(
            f"  - {feat} (value={val:.3f}, SHAP={sv:+.4f}, {direction} prediction score)\n"
            f"    Context: {paragraph}"
        )
    return f"{level_name}:\n" + "\n".join(lines)


def build_prompt(row, cp_set, shap_l1, shap_l2, shap_l3, feat_dict, mode=MODE_RESEARCH):
    """
    Build the full prompt. Public so Debug tab can display it.
    mode: 'research' (ground truth visible) or 'clinical' (ground truth hidden)
    """
    shap_text = "\n\n".join([
        _build_shap_summary(shap_l1, "L1 (Binary: Positive vs Negative)", feat_dict),
        _build_shap_summary(shap_l2, "L2 (Heavy Chain Subtype)", feat_dict),
        _build_shap_summary(shap_l3, "L3 (Light Chain: Kappa vs Lambda)", feat_dict),
    ])
    cp_display = ", ".join([pretty(c) for c in cp_set])

    # ── Patient context block differs by mode
    context_lines = [
        f"- Predicted class: {pretty(row['pred_class'])}",
        f"- Confidence: {row['confidence']:.4f}",
        f"- Confidence zone: {row['zone']}",
        f"- Conformal prediction set (α=0.05): [{cp_display}] ({len(cp_set)}-class set)",
    ]

    if mode == MODE_RESEARCH:
        context_lines.append(f"- Ground truth: {pretty(row['true_class'])}")
        context_lines.append(f"- Correct: {'Yes' if row['correct'] == 1 else 'No'}")

    context_block = "\n".join(context_lines)

    # ── Instructions differ by mode
    if mode == MODE_RESEARCH:
        instructions = """Write a concise clinical interpretation paragraph (150-200 words) that:
1. Explains WHY the model predicted this class, grounding reasoning in the SHAP features
2. Highlights the most discriminative signal features (e.g. gamma peak sharpness, channel asymmetries)
3. Comments on the confidence zone and what the conformal prediction set implies for clinical action
4. Since ground truth is known: analyze whether the prediction was correct or incorrect, and if incorrect, discuss which features may have misled the model
5. Use professional clinical laboratory language"""
    else:
        instructions = """Write a concise clinical interpretation paragraph (150-200 words) that:
1. Explains WHY the model predicted this class, grounding reasoning in the SHAP features
2. Highlights the most discriminative signal features (e.g. gamma peak sharpness, channel asymmetries)
3. Comments on the confidence zone and what the conformal prediction set implies for clinical action
4. Provide a clear recommendation: whether the result can be auto-reported or requires manual expert review
5. If confidence is low or the prediction set is large, flag this as requiring additional testing (e.g. immunofixation)
6. Use professional clinical laboratory language appropriate for a pathologist or laboratory scientist"""

    return f"""You are a clinical laboratory AI assistant interpreting the output of a cascade M-protein classifier that analyzes serum protein electrophoresis (SPE) signals across 6 immunofixation channels (ELP, IgG, IgA, IgM, Kappa, Lambda).

MODE: {"RESEARCH (ground truth available)" if mode == MODE_RESEARCH else "CLINICAL (real-world, no ground truth)"}

PATIENT CONTEXT:
{context_block}

SHAP FEATURE IMPORTANCE (top features per cascade level):
{shap_text}

INSTRUCTIONS:
{instructions}

CRITICAL RULES:
- ONLY interpret features actually provided above. Do NOT invent or hallucinate additional findings.
- Do NOT make a clinical diagnosis. This is a model interpretation, not a patient diagnosis.
- Keep it factual and grounded in the SHAP values provided.
- Write in English.
"""


# ═══════════════════════════════════════════════════════════
#  API Call with Caching + Metadata
# ═══════════════════════════════════════════════════════════

def _cache_key(row, shap_l1, shap_l2, shap_l3, mode):
    parts = [str(row.name), str(row['pred_class']), str(row['confidence']),
             str(shap_l1)[:200], str(shap_l2)[:200], str(shap_l3)[:200], mode]
    return hashlib.md5("|".join(parts).encode()).hexdigest()


def generate_interpretation(row, cp_set, shap_l1, shap_l2, shap_l3, feat_dict, mode=MODE_RESEARCH):
    """
    Call Anthropic API.
    Returns (text, metadata_dict, error_msg).
    metadata_dict: {prompt, model, input_tokens, output_tokens, latency_ms, cache_hit}
    """
    api_key = _resolve_api_key()
    prompt = build_prompt(row, cp_set, shap_l1, shap_l2, shap_l3, feat_dict, mode)
    meta = {"prompt": prompt, "model": "claude-sonnet-4-20250514",
            "input_tokens": None, "output_tokens": None,
            "latency_ms": None, "cache_hit": False}

    if not api_key:
        return None, meta, "no_key"

    ck = _cache_key(row, shap_l1, shap_l2, shap_l3, mode)
    cache_store = st.session_state.setdefault("_llm_cache", {})
    meta_store = st.session_state.setdefault("_llm_meta", {})

    if ck in cache_store:
        meta["cache_hit"] = True
        if ck in meta_store:
            meta.update(meta_store[ck])
        meta["cache_hit"] = True
        return cache_store[ck], meta, None

    # ── Make API call
    t0 = time.time()
    payload = {
        "model": meta["model"],
        "max_tokens": 512,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    url = "https://api.anthropic.com/v1/messages"

    try:
        # Try httpx first, then requests
        resp_json = None
        try:
            import httpx
            resp = httpx.post(url, headers=headers, json=payload, timeout=30.0)
            resp_json = resp.json()
            status = resp.status_code
        except ImportError:
            import requests
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp_json = resp.json()
            status = resp.status_code

        latency = int((time.time() - t0) * 1000)
        meta["latency_ms"] = latency

        if status != 200:
            err = resp_json.get("error", {}).get("message", str(resp_json)[:200])
            return None, meta, f"API error ({status}): {err}"

        text = resp_json["content"][0]["text"]
        usage = resp_json.get("usage", {})
        meta["input_tokens"] = usage.get("input_tokens")
        meta["output_tokens"] = usage.get("output_tokens")

        cache_store[ck] = text
        meta_store[ck] = {k: v for k, v in meta.items() if k != "prompt"}
        return text, meta, None

    except Exception as e:
        meta["latency_ms"] = int((time.time() - t0) * 1000)
        return None, meta, f"Request failed: {str(e)}"


# ═══════════════════════════════════════════════════════════
#  Template Fallback
# ═══════════════════════════════════════════════════════════

def generate_template_interpretation(row, cp_set, shap_l1, shap_l2, shap_l3, feat_dict, mode=MODE_RESEARCH):
    pred = pretty(row['pred_class'])
    zone = row['zone']
    conf = row['confidence']

    parts = [f"The model classified this sample as **{pred}** with a compound confidence of {conf:.4f} ({zone} zone)."]

    if len(cp_set) == 1:
        parts.append(f"The conformal prediction set contains only {pred}, suggesting high certainty suitable for auto-reporting.")
    else:
        cp_names = ", ".join([pretty(c) for c in cp_set])
        parts.append(f"The conformal prediction set includes {len(cp_set)} classes ({cp_names}), indicating manual review is recommended.")

    for level_name, shap_data in [("L1 binary", shap_l1), ("L2 heavy chain", shap_l2), ("L3 light chain", shap_l3)]:
        if isinstance(shap_data, str) or not shap_data:
            continue
        top = shap_data[0]
        direction = "positively" if top[2] >= 0 else "negatively"
        parts.append(f"At the {level_name} level, **{top[0]}** (value={top[1]:.2f}) most strongly influenced the prediction {direction} (SHAP={top[2]:+.3f}).")

    if mode == MODE_RESEARCH and row['correct'] == 0:
        true = pretty(row['true_class'])
        parts.append(f"⚠️ Note: The ground truth is {true}, indicating a misclassification.")

    return " ".join(parts)


# ═══════════════════════════════════════════════════════════
#  Web UI Renderer
# ═══════════════════════════════════════════════════════════

def render_interpretation_section(row, cp_set, shap_l1, shap_l2, shap_l3, feat_dict, mode=MODE_RESEARCH, publication_mode=False):
    """Render AI interpretation in web UI. Returns (interp_text, metadata) tuple."""
    mode_label = "🔬 Research" if mode == MODE_RESEARCH else "🏥 Clinical"
    st.caption(f"{mode_label} mode")

    has_key = is_available()
    meta = None

    if has_key:
        col_btn, col_status = st.columns([1, 3])
        with col_btn:
            generate = st.button("Generate AI Interpretation", type="primary", key="_gen_interp")

        active_key = f"{row.name}_{mode}"
        should_show = generate or st.session_state.get("_last_interp_key") == active_key

        if should_show:
            with st.spinner("Generating clinical interpretation..."):
                text, meta, err = generate_interpretation(row, cp_set, shap_l1, shap_l2, shap_l3, feat_dict, mode)

            if text:
                st.session_state["_last_interp_key"] = active_key
                st.session_state["_last_interp_text"] = text
                st.session_state["_last_interp_meta"] = meta
                _render_interp_card(text)
                return text, meta
            elif err:
                with col_status:
                    st.warning(f"Could not generate: {err}")
        else:
            cached = st.session_state.get("_last_interp_text")
            cached_meta = st.session_state.get("_last_interp_meta")
            if cached and st.session_state.get("_last_interp_key") == active_key:
                _render_interp_card(cached)
                return cached, cached_meta

    # Fallback: template-based
    if publication_mode:
        # Clean look: no expander, no API key hints — just the interpretation
        tmpl = generate_template_interpretation(row, cp_set, shap_l1, shap_l2, shap_l3, feat_dict, mode)
        _render_interp_card(tmpl)
        return tmpl, None
    else:
        with st.expander("📝 Template-Based Interpretation" + (" (API key not set)" if not has_key else ""), expanded=not has_key):
            tmpl = generate_template_interpretation(row, cp_set, shap_l1, shap_l2, shap_l3, feat_dict, mode)
            st.markdown(tmpl)
            if not has_key:
                st.caption("💡 Add an Anthropic API key to enable AI-powered interpretations.")
            return tmpl, None

    return None, None

def _render_interp_card(text):
    st.markdown(
        f"<div style='background:#F8F9FA; border-left:4px solid #1565C0; "
        f"padding:16px 20px; border-radius:4px; font-size:14px; "
        f"line-height:1.6; color:#333;'>{text}</div>",
        unsafe_allow_html=True,
    )
    st.caption("⚠️ AI-generated interpretation of model outputs — not a clinical diagnosis.")
