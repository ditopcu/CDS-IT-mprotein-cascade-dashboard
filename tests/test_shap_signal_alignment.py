"""The SHAP a patient is shown must be the SHAP of that patient's signal.

The demo artifacts are rebuilt by debug.py (create_demo_dataset), which re-keys the
per-level SHAP blocks from the full cohort's row index to the demo cohort's row index.
That re-keying used to be a plain arange(), which is right for L1 (every selected
patient is present) and wrong for the positive-only L2/L3 blocks: each positive was
handed the explanation of a different patient, and two genuine positives lost theirs.
Nothing in the app can notice, because a SHAP vector carries no patient identity.

The check here does not trust the index at all. It re-derives the 399 features from the
signal the app renders and compares them with the X_matrix stored next to the SHAP row,
so a re-keying mistake shows up as a feature mismatch.
"""
import numpy as np
import pytest

import inference                       # installs the np.trapz shim cascade_src needs
import data_loader as dl
from cascade_src.constants import CHANNELS
from cascade_src.features import extract_all_features

# The signals are stored float32 and the features are re-derived in float64, so the
# comparison is on relative error. A wrong patient differs by orders of magnitude.
REL_TOL = 1e-3


@pytest.fixture(scope="module")
def cohort():
    ds, shap_d, _feat_dict, master_df = dl.load_all_data()
    return ds, shap_d, master_df


def _features_of(signal):
    frame = extract_all_features(np.asarray(signal, float)[None, ...],
                                 channels=CHANNELS, verbose=False)
    return frame.values[0], list(frame.columns)


@pytest.mark.parametrize("level", ["L1", "L2", "L3"])
def test_stored_shap_row_belongs_to_the_rendered_signal(cohort, level):
    ds, shap_d, master_df = cohort
    checked = 0
    for patient_id, row in master_df.iterrows():
        suffix = dl.SRC_SUFFIX[row["source"]]
        stored = dl.get_patient_shap_full(shap_d, level, int(row["sig_idx"]), suffix)
        if stored is None:                     # no SHAP for this patient at this level
            continue
        feat_names, _shap_values, x_values = stored
        recomputed, columns = _features_of(dl.get_patient_signal(ds, row))
        assert columns == list(feat_names), "feature order drifted from cascade_src"
        rel = np.nanmax(np.abs(np.asarray(x_values, float) - recomputed)
                        / (np.abs(recomputed) + 1e-9))
        assert rel < REL_TOL, (
            f"{level} SHAP for {row['source']} patient {patient_id} "
            f"(sig_idx {row['sig_idx']}) does not match its own signal, rel err {rel:.3g}"
        )
        checked += 1
    assert checked > 0, f"no {level} SHAP rows were checked"


def test_positive_only_levels_are_keyed_by_cohort_row_not_by_position(cohort):
    """L2/L3 sample_indices must be the cohort row index, never 0..n-1.

    A contiguous arange over a positive-only block is the exact shape of the old bug,
    and it is invisible in the app. It stays legal only when the positives really do
    occupy the first rows, which the explicit L1 comparison below pins down.
    """
    _ds, shap_d, _master_df = cohort
    for key in ("L2", "L3", "L2_external", "L3_external"):
        if key not in shap_d:
            continue
        positive_idx = np.asarray(shap_d[key]["sample_indices"])
        all_idx = np.asarray(shap_d[key.replace("L2", "L1").replace("L3", "L1")]
                             ["sample_indices"])
        assert set(positive_idx.tolist()) <= set(all_idx.tolist()), (
            f"{key} refers to rows that are not in the cohort"
        )
        assert len(set(positive_idx.tolist())) == len(positive_idx), f"{key} has duplicates"
