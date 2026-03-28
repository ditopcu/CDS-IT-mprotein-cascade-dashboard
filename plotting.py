"""
plotting.py – All visualization functions.
  • Plotly (interactive)  → web UI
  • Matplotlib (static)   → PDF export
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import (
    CHANNELS, CH_COLORS, PROTEIN_REGIONS,
    SHAP_POS_COLOR, SHAP_NEG_COLOR,
    PLOT_DPI, FIG_BG, GRID_COLOR, FONT_FAMILY,
    ZONE_COLORS,
)


# ═══════════════════════════════════════════════════════════
#  Signal helpers
# ═══════════════════════════════════════════════════════════
def _abs_signals(signal):
    """Convert raw 6-ch signal → absorbance-like representation."""
    elp = signal[0]
    return [elp] + [elp - signal[i] for i in range(1, 6)]


# ═══════════════════════════════════════════════════════════
#  Plotly: 6-channel faceted signal (interactive, web)
# ═══════════════════════════════════════════════════════════
def plotly_signal_faceted(signal, title="Signal Trace"):
    abs_s = _abs_signals(signal)
    elp   = abs_s[0]
    T     = len(elp)

    fig = make_subplots(
        rows=6, cols=1, shared_xaxes=True,
        vertical_spacing=0.015,
        subplot_titles=[None]*6,
    )

    for i, (ch, col) in enumerate(zip(CHANNELS, CH_COLORS)):
        row = i + 1

        # ELP reference in channels 2-6
        if i > 0:
            fig.add_trace(go.Scatter(
                x=list(range(T)), y=elp.tolist(),
                mode='lines', line=dict(color='#AAAAAA', width=0.8, dash='dot'),
                showlegend=False, hoverinfo='skip',
            ), row=row, col=1)

        # Channel trace
        fig.add_trace(go.Scatter(
            x=list(range(T)), y=abs_s[i].tolist(),
            mode='lines', line=dict(color=col, width=1.6),
            name=ch, showlegend=True,
            hovertemplate=f"<b>{ch}</b><br>pos=%{{x}}<br>val=%{{y:.4f}}<extra></extra>",
        ), row=row, col=1)

        # Y-axis label
        fig.update_yaxes(
            title_text=ch, title_font=dict(size=11, color=col),
            title_standoff=5,
            showticklabels=False, showgrid=False,
            zeroline=True, zerolinecolor=GRID_COLOR, zerolinewidth=0.6,
            row=row, col=1,
        )
        fig.update_xaxes(showticklabels=(i == 5), showgrid=False, row=row, col=1)

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.01, font=dict(size=14)),
        height=560, margin=dict(l=70, r=20, t=50, b=30),
        plot_bgcolor='white', paper_bgcolor='white',
        font=dict(family=FONT_FAMILY),
        legend=dict(orientation='h', yanchor='bottom', y=-0.08, x=0.5, xanchor='center'),
        hovermode='x unified',
    )
    return fig


# ═══════════════════════════════════════════════════════════
#  Matplotlib: combined signal (PDF / static export)
# ═══════════════════════════════════════════════════════════
def mpl_signal_combined(signal, title="Signal Trace"):
    abs_s = _abs_signals(signal)
    fig, ax = plt.subplots(figsize=(10, 3), dpi=PLOT_DPI)

    for i, (ch, col) in enumerate(zip(CHANNELS, CH_COLORS)):
        ax.plot(abs_s[i], color=col, lw=1.2, label=ch, zorder=2)

    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.legend(ncol=6, fontsize=7, loc='lower center', bbox_to_anchor=(0.5, -0.35))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig


def mpl_signal_faceted(signal, title="Signal Trace (6-Channel)"):
    abs_s = _abs_signals(signal)
    elp = abs_s[0]

    fig, axes = plt.subplots(6, 1, figsize=(12, 8), dpi=PLOT_DPI, sharex=True)
    fig.subplots_adjust(hspace=0.15, left=0.08)

    for i, (ax, ch, col) in enumerate(zip(axes, CHANNELS, CH_COLORS)):
        ax.axhline(0, color=GRID_COLOR, lw=0.6, zorder=1)
        if i > 0:
            ax.plot(elp, color='#999999', lw=0.6, linestyle='--', alpha=0.45, zorder=2)
        ax.plot(abs_s[i], color=col, lw=1.0, zorder=3)

        ax.tick_params(axis='y', length=0, labelleft=False)
        ax.set_ylabel(ch, rotation=0, labelpad=30, ha='center', va='center',
                       fontsize=12, fontweight='bold', color=col)

        for sp in ['top', 'right', 'left']:
            ax.spines[sp].set_visible(False)
        ax.spines['bottom'].set_visible(i == 5)
        if i == 5:
            ax.spines['bottom'].set_color('#CCCCCC')
            ax.tick_params(axis='x', labelsize=9, colors='#666666')
        else:
            ax.tick_params(axis='x', labelbottom=False)

        y_max = max(np.max(abs_s[i]), np.max(elp)) if i > 0 else np.max(abs_s[i])
        ax.set_ylim(-y_max * 0.05, y_max * 1.25)

    axes[0].set_title(title, fontweight='bold', loc='left', color='#222', pad=10, fontsize=12)
    return fig


# ═══════════════════════════════════════════════════════════
#  Plotly: SHAP waterfall (interactive, web)
# ═══════════════════════════════════════════════════════════
def plotly_shap_waterfall(shap_data, title):
    """Horizontal bar chart with hover showing full feature name + value."""
    if isinstance(shap_data, str) or not shap_data:
        fig = go.Figure()
        msgs = {
            'MISSING_KEY': 'SHAP data not available for this level',
            'NOT_IN_INDEX': 'Patient not evaluated at this cascade level',
        }
        fig.add_annotation(text=msgs.get(shap_data, 'No SHAP data'),
                           xref='paper', yref='paper', x=0.5, y=0.5,
                           showarrow=False, font=dict(size=13, color='#999'))
        fig.update_layout(
            title=dict(text=f"<b>{title}</b>", font=dict(size=12)),
            height=260, margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            plot_bgcolor='white', paper_bgcolor='white',
        )
        return fig

    features, vals, svs = [], [], []
    for feat, val, sv in reversed(shap_data):
        features.append(feat[:28] + '…' if len(feat) > 28 else feat)
        vals.append(val)
        svs.append(sv)

    colors = [SHAP_POS_COLOR if s >= 0 else SHAP_NEG_COLOR for s in svs]
    full_labels = [f"{f} = {v:.2f}" for f, v, _ in reversed(shap_data)]

    fig = go.Figure(go.Bar(
        x=svs, y=features, orientation='h',
        marker_color=colors,
        customdata=list(zip(full_labels, svs)),
        hovertemplate="<b>%{customdata[0]}</b><br>SHAP = %{customdata[1]:+.4f}<extra></extra>",
        text=[f"{s:+.3f}" for s in svs],
        textposition='outside', textfont=dict(size=9),
    ))
    # Expand x range so outside text doesn't clip
    max_abs = max(abs(s) for s in svs) if svs else 0.1
    pad = max_abs * 0.25

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=12)),
        height=300, margin=dict(l=180, r=70, t=40, b=20),
        plot_bgcolor='white', paper_bgcolor='white',
        xaxis=dict(zeroline=True, zerolinecolor=GRID_COLOR, zerolinewidth=1.5,
                   showgrid=True, gridcolor=GRID_COLOR, title='SHAP value',
                   range=[min(svs) - pad, max(svs) + pad]),
        yaxis=dict(tickfont=dict(size=9)),
        font=dict(family=FONT_FAMILY),
    )
    return fig


# ═══════════════════════════════════════════════════════════
#  Matplotlib: SHAP waterfall (PDF)
# ═══════════════════════════════════════════════════════════
def mpl_shap_waterfall(shap_data, title):
    fig, ax = plt.subplots(figsize=(5, 3), dpi=PLOT_DPI)

    if isinstance(shap_data, str) or not shap_data:
        msgs = {
            'MISSING_KEY': 'SHAP data not available',
            'NOT_IN_INDEX': 'Not evaluated at this level',
        }
        ax.text(0.5, 0.5, msgs.get(shap_data, 'No SHAP data'),
                ha='center', va='center', color='#999', fontsize=10)
        ax.axis('off')
        return fig

    features, shap_vals = [], []
    for feat, val, sv in reversed(shap_data):
        lbl = f"{feat[:25]}..={val:.1f}" if len(feat) > 25 else f"{feat}={val:.1f}"
        features.append(lbl)
        shap_vals.append(sv)

    colors = [SHAP_POS_COLOR if sv >= 0 else SHAP_NEG_COLOR for sv in shap_vals]
    bars = ax.barh(features, shap_vals, color=colors, height=0.6)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.axvline(0, color='#CCCCCC', linewidth=1)
    ax.tick_params(axis='y', length=0, labelsize=7, colors='#666666', pad=10)
    ax.tick_params(axis='x', labelsize=7, colors='#666666')
    ax.set_title(title, loc='left', fontweight='bold', fontsize=9, color='#222')

    max_val = max(np.max(np.abs(shap_vals)), 0.01)
    ax.set_xlim(ax.get_xlim()[0] - max_val * 0.35, ax.get_xlim()[1] + max_val * 0.35)
    for bar, sv in zip(bars, shap_vals):
        offset = max_val * 0.05 if sv >= 0 else -max_val * 0.05
        ax.text(bar.get_width() + offset, bar.get_y() + bar.get_height() / 2,
                f"{sv:+.3f}", va='center', ha='left' if sv >= 0 else 'right',
                fontsize=7, color='#222', fontweight='bold')
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════
#  HTML: Conformal prediction set (clean badge list)
# ═══════════════════════════════════════════════════════════
def render_conformal_set_html(cp_set, pred_class, zone):
    """Return HTML string for a clean conformal prediction set display."""
    from config import pretty, ZONE_COLORS
    n = len(cp_set)
    zone_col = ZONE_COLORS.get(zone, '#000')
    action = "Auto-reportable" if n == 1 else "Manual review"

    items_html = ""
    for cls in cp_set:
        is_pred = cls == pred_class
        label = pretty(cls)
        if is_pred:
            items_html += (
                f"<div style='padding:6px 14px; margin:3px 0; background:#F5F5F5; "
                f"border-left:4px solid #333; border-radius:4px; font-weight:bold; "
                f"font-size:15px; color:#222;'>★ {label}</div>"
            )
        else:
            items_html += (
                f"<div style='padding:5px 14px; margin:3px 0; background:#FAFAFA; "
                f"border-left:4px solid #CCC; border-radius:4px; "
                f"font-size:14px; color:#666;'>&nbsp;&nbsp;{label}</div>"
            )

    action_html = (
        f"<div style='margin-top:8px; padding:4px 10px; display:inline-block; "
        f"background:{zone_col}18; border:1px solid {zone_col}; border-radius:4px; "
        f"font-size:13px; font-weight:bold; color:{zone_col};'>"
        f"■ {n}-CLASS SET — {action}</div>"
    )

    return (
        f"<div style='font-family:Inter,Arial,sans-serif;'>"
        f"<div style='font-size:12px; color:#888; margin-bottom:6px;'>"
        f"Significance level: α = 0.05</div>"
        f"<div style='font-size:13px; color:#555; margin-bottom:4px;'>"
        f"Prediction set ({n} {'class' if n == 1 else 'classes'}):</div>"
        f"{items_html}{action_html}</div>"
    )

# ═══════════════════════════════════════════════════════════════
#  SHAP WATERFALL (cumulative, Matplotlib)
# ═══════════════════════════════════════════════════════════════

def mpl_shap_waterfall_cumulative(shap_data, title, base_value=None, n=8):
    """
    True SHAP waterfall: cumulative bars from base_value to f(x).
    shap_data: list of (feat_name, feat_value, shap_value) tuples
    base_value: E[f(x)]. If None, uses 0 with a note.
    Returns matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=150)

    if isinstance(shap_data, str) or not shap_data:
        ax.text(0.5, 0.5, f'{title}\nNo SHAP data available',
                ha='center', va='center', fontsize=10, color='#999')
        ax.set_axis_off()
        return fig

    # Top N by absolute value, sorted descending
    sorted_data = sorted(shap_data, key=lambda x: abs(x[2]), reverse=True)[:n]
    # Reverse for bottom-to-top plotting (most important at top)
    sorted_data = sorted_data[::-1]

    base = base_value if base_value is not None else 0.0
    has_base = base_value is not None

    feats  = [f'{d[0]}={d[1]:.1f}' for d in sorted_data]
    values = [d[2] for d in sorted_data]

    # Calculate cumulative positions (bottom to top)
    # Start from base, accumulate from least to most important
    cumsum = base
    starts = []
    for v in values:
        starts.append(cumsum)
        cumsum += v
    final_value = cumsum

    # Colors
    colors = ['#B2182B' if v >= 0 else '#2166AC' for v in values]

    y_pos = np.arange(len(feats))

    # Draw bars
    bars = ax.barh(y_pos, values, left=starts, color=colors, height=0.6,
                   edgecolor='white', linewidth=0.5)

    # Value labels on bars
    for i, (v, s) in enumerate(zip(values, starts)):
        x_text = s + v + (0.02 * abs(final_value - base) if v >= 0 else -0.02 * abs(final_value - base))
        ha = 'left' if v >= 0 else 'right'
        ax.text(x_text, i, f'{v:+.3f}', va='center', ha=ha,
                fontsize=7, fontweight='bold', color='#333')

    # Base value line
    ax.axvline(base, color='#CCCCCC', linewidth=1, linestyle=':', zorder=0)

    # Final value annotation
    ax.annotate(f'f(x)={final_value:.3f}', xy=(final_value, len(feats) - 0.5),
                fontsize=8, fontweight='bold', color='#C0392B',
                ha='left' if final_value >= base else 'right')

    # Base value annotation
    base_label = f'E[f(x)]={base:.3f}' if has_base else f'base=0'
    ax.annotate(base_label, xy=(base, -0.8), fontsize=7, color='#888',
                ha='center', style='italic')

    # Feature labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feats, fontsize=7.5)
    ax.set_xlabel('SHAP value (log-odds)', fontsize=8, color='#555')

    # Title
    ax.set_title(title, fontsize=10, fontweight='bold', loc='left', pad=8)

    # Tufte style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(left=False, labelsize=7)
    ax.tick_params(axis='x', labelsize=7, colors='#888')

    # Note if base approximated
    if not has_base:
        ax.text(0.99, 0.01, 'base value approximated; relative contributions are exact',
                transform=ax.transAxes, fontsize=5.5, ha='right', va='bottom',
                color='#AAAAAA', style='italic')

    plt.tight_layout()
    return fig

# ═══════════════════════════════════════════════════════════════
#  6-CHANNEL SHAP OVERLAY (Spatial Feature Attribution)
# ═══════════════════════════════════════════════════════════════

def mpl_shap_6channel_overlay(signal, feat_names, shap_vals, level_title="L1"):
    """
    6-channel signal with region-colored SHAP overlay.
    signal: (6, 300) array
    feat_names: list of 399 feature names
    shap_vals: (399,) array of SHAP values
    Returns matplotlib Figure.
    """
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch

    CHANNELS = ['raw_ELP', 'dif_IgG', 'dif_IgA', 'dif_IgM', 'dif_Kappa', 'dif_Lambda']
    CH_DISPLAY = {
        'raw_ELP': 'ELP (Ref)', 'dif_IgG': 'IgG', 'dif_IgA': 'IgA',
        'dif_IgM': 'IgM', 'dif_Kappa': 'Kappa', 'dif_Lambda': 'Lambda',
    }
    CH_PREFIXES = {
        'raw_ELP': ['raw_ELP'], 'dif_IgG': ['dif_IgG'], 'dif_IgA': ['dif_IgA'],
        'dif_IgM': ['dif_IgM'], 'dif_Kappa': ['dif_Kappa', 'kl_'], 'dif_Lambda': ['dif_Lambda'],
    }
    REGIONS = {
        'beta1': (133, 171), 'beta2': (171, 194),
        'transition': (194, 211), 'gamma': (211, 263),
    }
    POS_COLOR = '#B2182B'
    NEG_COLOR = '#2166AC'

    fn_arr = np.array(feat_names)
    shap_abs_max = np.abs(shap_vals).max() if len(shap_vals) > 0 else 1.0

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for ch_i, ch_name in enumerate(CHANNELS):
        ax = axes.flat[ch_i]
        ax.grid(False)

        elp_sig = signal[0]
        dif_sig = signal[ch_i]
        plot_sig = elp_sig if ch_name == 'raw_ELP' else elp_sig - dif_sig

        y_min, y_max = plot_sig.min(), plot_sig.max()
        y_range = (y_max - y_min) if y_max != y_min else 1.0
        ax.set_ylim(y_min - y_range * 0.05, y_max + y_range * 0.12)

        # Channel mask
        prefixes = CH_PREFIXES[ch_name]
        ch_mask = np.zeros(len(fn_arr), dtype=bool)
        for pfx in prefixes:
            ch_mask |= np.array([fn.startswith(pfx) for fn in fn_arr])
        ch_features = fn_arr[ch_mask]
        ch_shap = shap_vals[ch_mask]

        # Region SHAP sums
        rgn_sums = {}
        for rname, (r0, r1) in REGIONS.items():
            rmask = np.array([rname in fn for fn in ch_features])
            rgn_sums[rname] = ch_shap[rmask].sum() if rmask.sum() > 0 else 0.0

        # Region fill
        for rname, (r0, r1) in REGIONS.items():
            rs = rgn_sums[rname]
            intensity = np.clip(abs(rs) / (shap_abs_max + 1e-9), 0, 1)
            color = POS_COLOR if rs > 0 else NEG_COLOR
            rgba = mcolors.to_rgba(color, alpha=intensity * 0.45)
            ax.axvspan(r0, r1, color=rgba, zorder=0)

            if abs(rs) >= 0.001:
                x_frac = ((r0 + r1) / 2) / 300.0
                text_col = POS_COLOR if rs > 0 else NEG_COLOR
                # Alternate y position to avoid overlap
                rgn_idx = list(REGIONS.keys()).index(rname)
                y_off = 1.06 if rgn_idx % 2 == 0 else 1.01
                ax.text(x_frac, y_off, f'{rs:+.3f}', transform=ax.transAxes,
                        ha='center', va='bottom', fontsize=6, color=text_col,
                        fontweight='bold', clip_on=False, rotation=0)

        # Signal lines
        ax.fill_between(range(300), plot_sig, alpha=0.08, color='#333', zorder=2)
        ax.plot(plot_sig, color='#333', lw=1.4, zorder=3, alpha=0.85)
        if ch_name != 'raw_ELP':
            ax.plot(elp_sig, color='#E67E22', lw=1.0, zorder=4, alpha=0.6, ls='--')

        # Region dividers
        ax.axvline(211, color='#888', lw=0.6, ls=':', zorder=1, alpha=0.5)
        ax.axvline(263, color='#888', lw=0.6, ls=':', zorder=1, alpha=0.5)

        # Top-3 features box
        if len(ch_shap) > 0:
            top_i = np.argsort(np.abs(ch_shap))[::-1][:3]
            lines = []
            for ti in top_i:
                fn = ch_features[ti]; sv = ch_shap[ti]
                short = fn
                for pfx in prefixes:
                    short = short.replace(pfx + '_', '')
                lines.append(f'{"▲" if sv > 0 else "▼"} {short}: {sv:+.3f}')
            ax.text(0.02, 0.04, '\n'.join(lines), transform=ax.transAxes, fontsize=6.5, va='bottom',
                    bbox=dict(boxstyle='square,pad=0.3', fc='white', ec='none', alpha=0.85), zorder=6)

        ax.set_xlim(0, 300)
        ax.tick_params(labelsize=7)
        ch_total = ch_shap.sum()
        ax.set_xlabel(f'{CH_DISPLAY[ch_name]}   ΣSHAP={ch_total:+.3f}', fontsize=9, fontweight='bold')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Legend
    legend_els = [
        Patch(fc=mcolors.to_rgba(POS_COLOR, 0.45), ec='none', label='SHAP > 0 (→ positive/class)'),
        Patch(fc=mcolors.to_rgba(NEG_COLOR, 0.45), ec='none', label='SHAP < 0 (→ negative/other)'),
        plt.Line2D([0], [0], color='#333', lw=1.4, label='Channel Signal'),
        plt.Line2D([0], [0], color='#E67E22', lw=1.0, ls='--', label='ELP Reference'),
    ]
    fig.legend(handles=legend_els, loc='lower center', ncol=4, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f'Spatial SHAP Attribution: {level_title}', fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    return fig