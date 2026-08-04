"""
pdf_export.py – Professional 2-page PDF report.
Page 1: Executive Summary (decision at a glance)
Page 2: Evidence (Baseline Panel + SHAP XAI explanations)
"""
import io
from datetime import date
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader

from config import (
    ZONE_COLORS, PDF_DPI, PDF_MARGIN as M,
    SHAP_POS_COLOR, SHAP_NEG_COLOR, pretty,
    reflex_group, UNIVERSAL_BASELINE, REFLEX_MATRIX,
    INTENDED_USE_NOTICE,
)
from data_loader import get_human_readable_parts


# ─── Helpers ──────────────────────────────────────────────

def _wrap(c, text, x, y, max_w, font, size, lh):
    """Word-wrap text, return new y position."""
    c.setFont(font, size)
    words = text.split()
    lines, cur = [], []
    for w in words:
        test = ' '.join(cur + [w])
        if c.stringWidth(test, font, size) <= max_w:
            cur.append(w)
        else:
            lines.append(' '.join(cur)); cur = [w]
    if cur:
        lines.append(' '.join(cur))
    for line in lines:
        c.drawString(x, y, line)
        y -= lh
    return y


def _draw_fig(c, fig, x, y, w, h):
    """Place a Matplotlib figure on the canvas via an in-memory PNG.

    Figures are patient-derived, so they are never written to a temp file — rendering
    through BytesIO leaves no image on disk.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=PDF_DPI)
    buf.seek(0)
    c.drawImage(ImageReader(buf), x, y, width=w, height=h, preserveAspectRatio=True)


def _heading(c, text, x, y):
    """Section heading with thin underline."""
    c.setFont('Helvetica-Bold', 11)
    c.setFillColor(HexColor('#333333'))
    c.drawString(x, y, text)
    c.setStrokeColor(HexColor('#DDDDDD'))
    c.line(x, y - 3, x + c.stringWidth(text, 'Helvetica-Bold', 11), y - 3)
    return y - 18


def _footer(c, text, y=20):
    """Page footer: the intended-use notice above the page label. On every page."""
    from reportlab.lib.pagesizes import A4 as _A4
    max_w = _A4[0] - 2 * M
    c.setFillColor(HexColor('#8A4B00'))
    # Wrap upward from the page label so the notice never collides with it.
    n_lines = 1
    while c.stringWidth(INTENDED_USE_NOTICE, 'Helvetica-Bold', 6.5) / n_lines > max_w:
        n_lines += 1
    _wrap(c, INTENDED_USE_NOTICE, M, y + 9 + (n_lines - 1) * 7.5, max_w,
          'Helvetica-Bold', 6.5, 7.5)
    c.setFont('Helvetica', 7)
    c.setFillColor(HexColor('#BBBBBB'))
    c.drawString(M, y, text)


# ─── Main Export ──────────────────────────────────────────

def create_pdf(row, disp_id, cp_set, reflex_text,
               p1, p2, p3,
               fig_sig, fig_s1, fig_s2, fig_s3,
               shap_l1, shap_l2, shap_l3, feat_dict,
               ai_interpretation=None):

    buf = io.BytesIO()
    W, H = A4
    W_content = W - 2 * M
    x2 = W / 2 + 10
    c = canvas.Canvas(buf, pagesize=A4)
    today = date.today().strftime('%Y-%m-%d')

    # ══════════════════════════════════════════════════
    #  PAGE 1: EXECUTIVE SUMMARY
    # ══════════════════════════════════════════════════

    # ── Ground truth tag ──
    true_col = '#1A9641' if row['correct'] == 1 else '#D73027'
    c.setFont('Helvetica-Bold', 9)
    c.setFillColor(HexColor(true_col))
    c.drawString(M, H - 35, f'[GROUND TRUTH: {pretty(row["true_class"])}]')

    # ── Title bar ──
    c.setFont('Helvetica-Bold', 16)
    c.setFillColor(HexColor('#222222'))
    c.drawString(M, H - 55, 'Clinical Decision Support System')
    c.setFont('Helvetica', 9)
    c.setFillColor(HexColor('#888888'))
    c.drawRightString(W - M, H - 40, 'Cascade M-Protein Classifier v1.0')
    c.drawRightString(W - M, H - 52, f'Date: {today}')
    c.setFillColor(HexColor('#666666'))
    c.drawString(M, H - 68, f'Sample ID: {disp_id}')
    c.setStrokeColor(HexColor('#CCCCCC'))
    c.line(M, H - 76, W - M, H - 76)

    # ── 1. Classification Result ──
    y = _heading(c, '1. Classification Result', M, H - 95)
    y -= 6
    c.setFont('Helvetica-Bold', 22)
    c.setFillColor(HexColor('#222222'))
    c.drawString(M, y, pretty(row['pred_class']))
    y -= 26
    c.setFont('Helvetica', 9)
    c.setFillColor(HexColor('#555555'))
    c.drawString(M, y, f'L1 (Binary): p(Positive) = {p1}'); y -= 12
    c.drawString(M, y, f'L2 (Heavy):  p(-) = {p2}'); y -= 12
    c.drawString(M, y, f'L3 (Light):  p(-) = {p3}'); y -= 16
    c.setFont('Helvetica-Bold', 10)
    c.setFillColor(HexColor('#C62828'))
    c.drawString(M, y, f'Compound Confidence: {row["confidence"]:.4f}')

    # ── 2. Conformal Prediction Set ──
    y2 = _heading(c, '2. Conformal Prediction Set', x2, H - 95)
    c.setFont('Helvetica', 9)
    c.setFillColor(HexColor('#888888'))
    c.drawString(x2, y2, 'Significance level: \u03b1 = 0.05'); y2 -= 14
    set_size = len(cp_set)
    c.setFillColor(HexColor('#555555'))
    c.setFont('Helvetica-Bold', 9)
    c.drawString(x2, y2, f'Prediction set ({set_size} classes):'); y2 -= 13
    c.setFont('Helvetica', 9)
    c.setFillColor(HexColor('#444444'))
    for cls in cp_set:
        c.drawString(x2 + 12, y2, f'\u2022  {pretty(cls)}'); y2 -= 11
    y2 -= 3
    sc = '#1A9641' if set_size == 1 else '#D73027'
    c.setFillColor(HexColor(sc))
    c.setFont('Helvetica-Bold', 10)
    act_label = "Auto-reportable" if set_size == 1 else "Manual review"
    c.drawString(x2, y2, f'\u25a0 {set_size}-CLASS SET \u2014 {act_label}')

    # ── Fixed row 2 anchor ──
    ROW2_Y = H - 260
    c.setStrokeColor(HexColor('#EEEEEE'))
    c.line(M, ROW2_Y + 15, W - M, ROW2_Y + 15)

    # ── 3. Confidence Zone & Action ──
    y3 = _heading(c, '3. Confidence Zone & Action', M, ROW2_Y)
    zone_col = ZONE_COLORS.get(row['zone'], '#000')
    c.setFont('Helvetica-Bold', 16)
    c.setFillColor(HexColor(zone_col))
    c.drawString(M, y3, f'ZONE: {row["zone"]}'); y3 -= 14
    c.setFont('Helvetica', 9)
    c.setFillColor(HexColor('#444444'))
    if row['zone'] == 'HIGH':
        c.drawString(M, y3, '\u2022 Auto-verify & Report')
    elif row['zone'] == 'MEDIUM':
        c.drawString(M, y3, '\u2022 Technician verification required')
    else:
        c.drawString(M, y3, '\u2022 Expert review required')
    y3 -= 11
    if row['pred_class'] != 'NEGATIVE':
        c.drawString(M, y3, '\u2022 See reflex recommendations \u2192')

    # ── 4. Reflex Test Recommendation (from matrix) ──
    grp = reflex_group(row['pred_class'])
    matrix_entry = REFLEX_MATRIX.get(grp, {}).get(row['zone'], {})
    gel_ife = matrix_entry.get('gel_ife', 'N/A')
    extra_tests = matrix_entry.get('tests', [])
    guidance_text = matrix_entry.get('guidance', '')

    _heading(c, '4. Reflex Test Recommendation', x2, ROW2_Y)
    ry = ROW2_Y - 22

    # Gel IFE with color
    ife_colors = {
        'Not required': '#1A9641', 'Consider': '#F46D43',
        'Recommended': '#E65100', 'Mandatory': '#D73027',
    }
    c.setFont('Helvetica-Bold', 9)
    c.setFillColor(HexColor(ife_colors.get(gel_ife, '#333')))
    c.drawString(x2, ry, f'Gel IFE: {gel_ife}'); ry -= 13

    # Extra tests (max 3)
    c.setFont('Helvetica', 8)
    c.setFillColor(HexColor('#444444'))
    for t in extra_tests[:3]:
        c.drawString(x2, ry, f'\u2022 {t}'); ry -= 10

    # Guidance (wrapped)
    if guidance_text:
        ry -= 2
        c.setFont('Helvetica-Oblique', 7.5)
        c.setFillColor(HexColor('#555555'))
        ry = _wrap(c, guidance_text, x2, ry, W / 2 - M - 15, 'Helvetica-Oblique', 7.5, 9)

    # Baseline cross-reference
    if grp != 'NEGATIVE':
        ry -= 6
        c.setFont('Helvetica-Oblique', 7)
        c.setFillColor(HexColor('#888888'))
        c.drawString(x2, ry, '\u2192 Universal Baseline Panel (7 tests) \u2014 see page 2')

    # ── 5. Signal Trace ──
    SIG_TOP = ROW2_Y - 75
    _draw_fig(c, fig_sig, M, SIG_TOP - 150, W_content, 150)

    # ── 6. SHAP Waterfalls (3 side-by-side) ──
    SHAP_TOP = SIG_TOP - 170
    _heading(c, '6. Model Decision Support (SHAP Waterfalls)', M, SHAP_TOP)
    SHAP_TOP -= 5
    pw = W_content / 3
    for i, fs in enumerate([fig_s1, fig_s2, fig_s3]):
        _draw_fig(c, fs, M + i * pw, SHAP_TOP - 140, pw, 140)

    # ── 7. AI Interpretation (compact, page 1 bottom) ──
    if ai_interpretation:
        ai_y = SHAP_TOP - 150
        c.setStrokeColor(HexColor('#DDDDDD'))
        c.setLineWidth(0.5)
        c.line(M, ai_y, W - M, ai_y)
        ai_y -= 12
        c.setFont('Helvetica-Bold', 9)
        c.setFillColor(HexColor('#1565C0'))
        c.drawString(M, ai_y, '\u25a0 AI Clinical Interpretation')
        ai_y -= 12
        c.setFillColor(HexColor('#444444'))
        ai_y = _wrap(c, ai_interpretation, M, ai_y, W_content,
                     'Helvetica', 7.5, 9.5)
        ai_y -= 6
        c.setFont('Helvetica-Oblique', 6.5)
        c.setFillColor(HexColor('#999999'))
        c.drawString(M, ai_y,
                     'Disclaimer: AI-generated interpretation \u2014 not a clinical diagnosis. '
                     'Always verify with qualified laboratory professionals.')

    _footer(c, f'Page 1  |  Cascade CDS Pipeline  |  {today}')

    # ══════════════════════════════════════════════════
    #  PAGE 2: EVIDENCE (Baseline + XAI)
    # ══════════════════════════════════════════════════
    c.showPage()
    y = H - 40

    # ── Universal Baseline Panel ──
    if grp != 'NEGATIVE':
        c.setFont('Helvetica-Bold', 12)
        c.setFillColor(HexColor('#333333'))
        c.drawString(M, y, 'Universal Baseline Panel')
        c.setFont('Helvetica', 8)
        c.setFillColor(HexColor('#888888'))
        c.drawRightString(W - M, y, 'Recommended for all M-protein positive predictions')
        y -= 8
        c.setStrokeColor(HexColor('#DDDDDD'))
        c.line(M, y, W - M, y)
        y -= 14

        for test, rationale in UNIVERSAL_BASELINE:
            c.setFillColor(HexColor('#333333'))
            c.setFont('Helvetica-Bold', 8)
            c.drawString(M + 4, y, f'\u2022 {test}')
            tw = c.stringWidth(f'\u2022 {test}  ', 'Helvetica-Bold', 8)
            c.setFont('Helvetica', 7)
            c.setFillColor(HexColor('#666666'))
            max_rat_w = W_content - tw - 10
            rat_text = rationale
            while c.stringWidth(rat_text, 'Helvetica', 7) > max_rat_w and len(rat_text) > 20:
                rat_text = rat_text[:-4] + '...'
            c.drawString(M + 4 + tw, y, f'\u2014 {rat_text}')
            y -= 12

        y -= 8
        c.setStrokeColor(HexColor('#DDDDDD'))
        c.line(M, y, W - M, y)
        y -= 16

    # ── XAI Textual Explanations ──
    c.setFont('Helvetica-Bold', 14)
    c.setFillColor(HexColor('#333333'))
    c.drawString(M, y, 'XAI Textual Explanations')
    c.setFont('Helvetica', 9)
    c.setFillColor(HexColor('#888888'))
    c.drawRightString(W - M, y, f'Sample: {disp_id}')
    y -= 8
    c.setStrokeColor(HexColor('#DDDDDD'))
    c.line(M, y, W - M, y)
    y -= 22

    def write_shap(data, title, yp):
        """Write SHAP feature explanations for one cascade level."""
        c.setFont('Helvetica-Bold', 10)
        c.setFillColor(HexColor('#333333'))
        c.drawString(M, yp, title)
        yp -= 18
        if isinstance(data, str) or not data:
            c.setFont('Helvetica', 9)
            c.setFillColor(HexColor('#999999'))
            c.drawString(M + 10, yp, 'Evaluation skipped or no significant features.')
            return yp - 25
        for feat, val, sv in data:
            tf, para, _ = get_human_readable_parts(feat, val, sv, feat_dict)
            col = HexColor(SHAP_POS_COLOR if sv >= 0 else SHAP_NEG_COLOR)
            imp = "Increased" if sv >= 0 else "Decreased"
            c.setFont('Helvetica-Bold', 9)
            c.setFillColor(col)
            c.drawString(M + 8, yp, f'\u25a0 {tf}')
            c.setFont('Helvetica-Oblique', 8)
            w = c.stringWidth(f'\u25a0 {tf} ', 'Helvetica-Bold', 9)
            c.drawString(M + 8 + w, yp, f'({imp} by {abs(sv):.3f})')
            yp -= 12
            c.setFillColor(HexColor('#555555'))
            yp = _wrap(c, f'Value: {val:.2f} | {para}',
                       M + 16, yp, W_content - 24, 'Helvetica', 8, 10)
            yp -= 10
            if yp < 70:
                _footer(c, f'XAI Report  |  {today}')
                c.showPage()
                yp = H - 50
        return yp - 12

    y = write_shap(shap_l1, 'L1: Binary Classifier Explanations', y)
    y = write_shap(shap_l2, 'L2: Heavy Chain Classifier Explanations', y)
    y = write_shap(shap_l3, 'L3: Light Chain Classifier Explanations', y)

    _footer(c, f'Interpretability Report  |  {today}')
    c.save()
    buf.seek(0)
    return buf
