"""What the exported PDF is and is not allowed to say.

Two bugs are guarded here:

  * Clinical mode hid the reference class on screen while create_pdf() printed it anyway,
    so the downloaded report disclosed exactly what the screen had withheld.
  * The PDF read the reflex matrix and the universal baseline straight from config while
    the screen rendered whatever load_reflex_rules() returned, so a custom
    reflex_matrix.xlsx made the report contradict the screen.
"""
import matplotlib.pyplot as plt
import pandas as pd
import pypdf
import pytest

import config
from pdf_export import create_pdf


# A deliberate misclassification: the reference class must be findable as a string that
# appears nowhere else in the report, so "absent" means absent and not merely shadowed by
# an identical prediction. 'Free-' occurs in no reflex rule or baseline entry.
ROW = pd.Series({
    "pred_class": "IGG_KAPPA",
    "true_class": "FREE_LAMBDA",
    "correct": 0,
    "confidence": 0.8123,
    "zone": "HIGH",
    "source": "External",
})
GT_NEEDLE = "Free"


def _figs():
    out = []
    for _ in range(4):
        fig, ax = plt.subplots(figsize=(2, 1))
        ax.plot([0, 1], [0, 1])
        out.append(fig)
    return out


def _pdf_text(**kwargs):
    figs = _figs()
    shap = [("ELP_gamma_area", 1.5, 0.4), ("IgG_beta_max", 2.0, -0.2)]
    buf = create_pdf(
        row=kwargs.pop("row", ROW), disp_id="XXXX609", cp_set=kwargs.pop("cp_set", ["IGG_KAPPA"]),
        reflex_text="", p1="0.9000", p2="0.8000", p3="0.7000",
        fig_sig=figs[0], fig_s1=figs[1], fig_s2=figs[2], fig_s3=figs[3],
        shap_l1=shap, shap_l2=shap, shap_l3=shap, feat_dict={},
        **kwargs)
    for f in figs:
        plt.close(f)
    reader = pypdf.PdfReader(buf)
    return "\n".join(p.extract_text() or "" for p in reader.pages)


def test_research_mode_prints_the_reference_class():
    text = _pdf_text(show_ground_truth=True)
    assert "GROUND TRUTH" in text
    assert GT_NEEDLE in text


def test_clinical_mode_pdf_contains_no_ground_truth():
    text = _pdf_text(show_ground_truth=False)
    assert "GROUND TRUTH" not in text
    assert GT_NEEDLE not in text
    # Withheld is a different claim from "there is none" — the report must say which.
    assert "withheld" in text


def test_upload_without_reference_says_so_rather_than_withheld():
    row = ROW.copy()
    row["true_class"] = None
    row["correct"] = float("nan")
    text = _pdf_text(row=row, show_ground_truth=True)
    assert "no reference interpretation" in text
    assert "withheld" not in text
    assert "GROUND TRUTH" not in text


def test_default_is_not_a_silent_leak_path():
    # Callers that forget the flag get the Research behaviour; the app always passes it.
    import inspect
    sig = inspect.signature(create_pdf)
    assert sig.parameters["show_ground_truth"].default is True


def test_custom_reflex_matrix_reaches_the_pdf():
    custom = {"IgG": {"HIGH": {"gel_ife": "Mandatory",
                               "tests": ["SITE SPECIFIC TEST ALPHA"],
                               "guidance": "Local protocol ZZZ applies."}}}
    text = _pdf_text(reflex_matrix=custom)
    assert "SITE SPECIFIC TEST ALPHA" in text
    assert "Local protocol ZZZ applies." in text
    # and the config default for the same cell is gone
    assert "Mayo MGUS risk model" not in text


def test_config_matrix_is_the_fallback_not_the_override():
    text = _pdf_text()
    default_guidance = config.REFLEX_MATRIX["IgG"]["HIGH"]["guidance"]
    assert default_guidance.split("(")[0].strip()[:30] in text


def test_custom_baseline_reaches_the_pdf():
    text = _pdf_text(universal_baseline=[("SITE PANEL ONE", "because the site says so")])
    assert "SITE PANEL ONE" in text
    assert "(1 tests)" in text                    # the page-1 cross-reference counts them
    assert "b2-microglobulin" not in text and "microglobulin" not in text
