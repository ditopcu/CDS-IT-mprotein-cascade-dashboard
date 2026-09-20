#!/usr/bin/env python3
"""
create_demo_dataset.py
─────────────────────
Select 54 representative patients (27 internal + 27 external) from the full
dataset — one per (cohort × class × zone) combination — anonymize IDs, and
save mini *_demo.pkl files for Streamlit Cloud deployment.

Coverage: 2 cohorts × 9 classes × 3 zones = 54 patients

Run from the app root:
    python create_demo_dataset.py

Inputs  (must exist):
    data/dataset.pkl
    results/flow_df.pkl
    results/L4_shap_dense_full.pkl
    results/L4_ext_validation_results.pkl

Outputs (created):
    data/dataset_demo.pkl
    results/flow_df_demo.pkl
    results/ext_flow_df_demo.pkl
    results/L4_shap_dense_full_demo.pkl
    results/L4_ext_validation_results_demo.pkl
    demo_selection_report.csv

Non-patient files (copy as-is to repo):
    data/feature_dictionary.pkl   — no change needed
    data/reflex_matrix.xlsx       — no change needed
"""

import pickle
import copy
import sys
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

# The progress output is full of box drawing and check marks, and a default Windows
# console is cp1254 here, which cannot encode them. Without this the run dies on its
# first status line, after loading the pickles but before writing anything.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ─── CONFIG ──────────────────────────────────────────────────────────────────
ANON_START  = 9_000_001
HIGH_THRESH = 0.70
LOW_THRESH  = 0.30

CLASSES_9 = [
    "IGG_KAPPA", "IGG_LAMBDA",
    "IGA_KAPPA", "IGA_LAMBDA",
    "IGM_KAPPA", "IGM_LAMBDA",
    "FREE_KAPPA", "FREE_LAMBDA",
    "NEGATIVE",
]

ZONES = ["HIGH", "MEDIUM", "LOW"]

# Full grid: 2 cohorts × 9 classes × 3 zones = 54 slots
SELECTION_GRID = list(product(["int", "ext"], CLASSES_9, ZONES))


def load_pkl(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pkl(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    mb = path.stat().st_size / 1024 / 1024
    print(f"  ✓ Saved {path}  ({mb:.2f} MB)")


def assign_zone(confidence: float) -> str:
    if confidence >= HIGH_THRESH:
        return "HIGH"
    elif confidence >= LOW_THRESH:
        return "MEDIUM"
    else:
        return "LOW"


def compound_conf(l1_p, l2_p, l3_p, pred, threshold):
    """Compute compound confidence score for a single sample."""
    conf_l1 = abs(l1_p - threshold) / max(threshold, 1 - threshold)
    if pred == "NEGATIVE":
        return conf_l1
    max_l2 = float(np.max(l2_p)) if l2_p is not None else 1.0
    l3_factor = 2 * abs(l3_p - 0.5)
    return conf_l1 * max_l2 * l3_factor


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    root = Path(".")
    dataset_path = root / "data"    / "dataset.pkl"
    flowdf_path  = root / "results" / "flow_df.pkl"
    shap_path    = root / "results" / "L4_shap_dense_full.pkl"
    extval_path  = root / "results" / "L4_ext_validation_results.pkl"

    for p in [dataset_path, flowdf_path, shap_path, extval_path]:
        if not p.exists():
            print(f"✗ Missing: {p}")
            sys.exit(1)
    print("✓ All source files found.\n")

    # ─── Load ────────────────────────────────────────────────────────────
    print("Loading data...")
    ds      = load_pkl(dataset_path)
    flow_df = load_pkl(flowdf_path)
    shap_d  = load_pkl(shap_path)
    ext_val = load_pkl(extval_path)

    n_int = len(ds["sample_ids"])
    n_ext = len(ds["ext_sample_ids"])
    print(f"  Internal: {n_int}  |  External: {n_ext}")
    print(f"  dataset.pkl keys ({len(ds)}): {sorted(ds.keys())}")
    print(f"  flow_df shape: {flow_df.shape},  columns: {list(flow_df.columns)}")
    print()

    # ─── Prepare internal lookup ─────────────────────────────────────────
    int_flow = flow_df.copy()
    if "zone" not in int_flow.columns:
        int_flow["zone"] = int_flow["confidence"].apply(assign_zone)
    if "correct" not in int_flow.columns:
        int_flow["correct"] = int_flow["pred_class"] == int_flow["true_class"]

    # ─── Prepare external lookup ─────────────────────────────────────────
    l1_thr = shap_d.get("metadata", {}).get("L1_threshold", 0.4722)

    ext_pred    = np.array(ext_val["ext_pred"])
    ext_l1_prob = np.array(ext_val["ext_l1_proba"])
    ext_l2_prob = np.array(ext_val["ext_l2_proba"])
    ext_l3_prob = np.array(ext_val["ext_l3_proba"])
    ext_true    = np.array(ds["y_ext_class9"])

    ext_conf = np.array([
        compound_conf(ext_l1_prob[i], ext_l2_prob[i], ext_l3_prob[i],
                       ext_pred[i], l1_thr)
        for i in range(n_ext)
    ])

    ext_df = pd.DataFrame({
        "pred_class":  ext_pred,
        "true_class":  ext_true,
        "confidence":  ext_conf,
        "zone":        [assign_zone(c) for c in ext_conf],
        "correct":     ext_pred == ext_true,
    })

    # ─── Select patients ─────────────────────────────────────────────────
    selected_int = []   # positional indices into internal arrays
    selected_ext = []   # positional indices into external arrays
    report_rows  = []   # for summary table
    skipped      = []

    print("═══ PATIENT SELECTION (2 × 9 × 3 = 54 slots) ═══\n")

    for cohort, cls, zone in SELECTION_GRID:
        df    = int_flow if cohort == "int" else ext_df
        taken = selected_int if cohort == "int" else selected_ext
        tag   = "INT" if cohort == "int" else "EXT"

        # Exact match: true_class + zone
        mask = (df["true_class"] == cls) & (df["zone"] == zone)
        cands = [c for c in df.index[mask] if c not in taken]

        # Prefer correct predictions when available
        if cands:
            correct_cands = [c for c in cands if df.loc[c, "correct"]]
            if correct_cands:
                cands = correct_cands

        level = "exact"
        if not cands:
            # Relax: same class, adjacent zone
            adjacent = {"HIGH": ["MEDIUM"], "MEDIUM": ["HIGH", "LOW"],
                        "LOW": ["MEDIUM"]}
            for adj_z in adjacent[zone]:
                mask2 = (df["true_class"] == cls) & (df["zone"] == adj_z)
                cands = [c for c in df.index[mask2] if c not in taken]
                if cands:
                    level = f"relaxed→{adj_z}"
                    break

        if not cands:
            # Further relax: same class, any zone
            mask3 = df["true_class"] == cls
            cands = [c for c in df.index[mask3] if c not in taken]
            level = "any_zone"

        if not cands:
            skipped.append((cohort, cls, zone))
            print(f"  ⚠ SKIP  {tag:3s}  {cls:14s}  {zone:6s}  "
                  f"— no candidates left")
            continue

        chosen = cands[0]
        taken.append(chosen)
        row = df.loc[chosen]
        correct_str = "✓" if row["correct"] else "✗"

        report_rows.append({
            "cohort": tag, "slot_class": cls, "slot_zone": zone,
            "actual_zone": row["zone"], "correct": correct_str,
            "confidence": f'{row["confidence"]:.3f}',
            "idx": chosen, "match": level,
        })

        print(f"  {tag:3s}  {cls:14s}  {zone:6s}  →  idx={chosen:5d}  "
              f"conf={row['confidence']:.3f}  {correct_str}  [{level}]")

    n_sel_int = len(selected_int)
    n_sel_ext = len(selected_ext)
    n_total   = n_sel_int + n_sel_ext

    print(f"\n  Selected: {n_sel_int} internal + {n_sel_ext} external "
          f"= {n_total}")
    if skipped:
        print(f"  Skipped:  {len(skipped)} slots (insufficient data)")
        for s in skipped:
            print(f"    {s}")
    print()

    # ─── Anonymized ID maps ──────────────────────────────────────────────
    anon_int = {old: ANON_START + i for i, old in enumerate(selected_int)}
    anon_ext = {old: ANON_START + n_sel_int + i
                for i, old in enumerate(selected_ext)}

    int_idx = np.array(selected_int)
    ext_idx = np.array(selected_ext)

    # ─── Build dataset_demo.pkl ──────────────────────────────────────────
    print("Building dataset_demo.pkl...")
    demo_ds = {}

    # Pre-compute positive counts for mapping positive-only arrays
    full_pos_mask = np.array(ds.get("pos_mask", np.zeros(n_int, dtype=bool)),
                             dtype=bool)
    full_ext_pos_mask = np.array(ds.get("ext_pos_mask",
                                        np.zeros(n_ext, dtype=bool)),
                                 dtype=bool)
    n_int_pos = int(np.sum(full_pos_mask))
    n_ext_pos = int(np.sum(full_ext_pos_mask))

    # Cumulative index mapping for positive-only arrays
    if n_int_pos > 0:
        pos_cumsum = np.cumsum(full_pos_mask) - 1
    if n_ext_pos > 0:
        ext_pos_cumsum = np.cumsum(full_ext_pos_mask) - 1

    for key, val in ds.items():
        if val is None:
            demo_ds[key] = None
            continue

        if isinstance(val, np.ndarray) and val.ndim >= 1:
            # Internal full-size arrays
            if val.shape[0] == n_int:
                demo_ds[key] = val[int_idx]
                print(f"  {key}: {val.shape} → {demo_ds[key].shape}  [int]")
                continue

            # External full-size arrays
            if val.shape[0] == n_ext:
                demo_ds[key] = val[ext_idx]
                print(f"  {key}: {val.shape} → {demo_ds[key].shape}  [ext]")
                continue

            # Internal positive-only arrays
            if n_int_pos > 0 and val.shape[0] == n_int_pos:
                sel_pos_idx = [int(pos_cumsum[si])
                               for si in selected_int if full_pos_mask[si]]
                if sel_pos_idx:
                    demo_ds[key] = val[np.array(sel_pos_idx)]
                    print(f"  {key}: {val.shape} → {demo_ds[key].shape}"
                          f"  [int-pos]")
                    continue

            # External positive-only arrays
            if n_ext_pos > 0 and val.shape[0] == n_ext_pos:
                sel_pos_idx = [int(ext_pos_cumsum[si])
                               for si in selected_ext
                               if full_ext_pos_mask[si]]
                if sel_pos_idx:
                    demo_ds[key] = val[np.array(sel_pos_idx)]
                    print(f"  {key}: {val.shape} → {demo_ds[key].shape}"
                          f"  [ext-pos]")
                    continue

        # Scalar / dict / list / unmatched → keep as-is
        demo_ds[key] = copy.deepcopy(val)
        kind = type(val).__name__
        if isinstance(val, np.ndarray):
            kind += f" {val.shape}"
        elif isinstance(val, (dict, list)):
            kind += f" ({len(val)})"
        print(f"  {key}: as-is ({kind})")

    # Overwrite IDs with anonymized
    demo_ds["sample_ids"] = np.array(list(anon_int.values()), dtype=np.int64)
    demo_ds["ext_sample_ids"] = np.array(list(anon_ext.values()),
                                          dtype=np.int64)

    # Recompute pos_mask / ext_pos_mask for demo subset
    if "y_class9" in demo_ds:
        demo_ds["pos_mask"] = np.array(demo_ds["y_class9"]) != "NEGATIVE"
    if "y_ext_class9" in demo_ds:
        demo_ds["ext_pos_mask"] = (
            np.array(demo_ds["y_ext_class9"]) != "NEGATIVE"
        )

    save_pkl(demo_ds, root / "data" / "dataset_demo.pkl")

    # ─── Build flow_df_demo.pkl ──────────────────────────────────────────
    print("\nBuilding flow_df_demo.pkl...")
    demo_flow = int_flow.loc[selected_int].copy().reset_index(drop=True)
    demo_flow.insert(0, "sample_id", list(anon_int.values()))
    print(f"  Shape: {demo_flow.shape}")
    cols_show = ["sample_id", "true_class", "pred_class", "zone",
                 "correct", "confidence"]
    cols_show = [c for c in cols_show if c in demo_flow.columns]
    print(demo_flow[cols_show].to_string())
    save_pkl(demo_flow, root / "results" / "flow_df_demo.pkl")

    # ─── Build ext_flow_df_demo.pkl ──────────────────────────────────────
    print("\nBuilding ext_flow_df_demo.pkl...")
    demo_ext_flow = ext_df.loc[selected_ext].copy().reset_index(drop=True)
    demo_ext_flow.insert(0, "sample_id", list(anon_ext.values()))
    print(f"  Shape: {demo_ext_flow.shape}")
    cols_show2 = ["sample_id", "true_class", "pred_class", "zone",
                  "correct", "confidence"]
    cols_show2 = [c for c in cols_show2 if c in demo_ext_flow.columns]
    print(demo_ext_flow[cols_show2].to_string())
    save_pkl(demo_ext_flow, root / "results" / "ext_flow_df_demo.pkl")

    # ─── Build SHAP demo ─────────────────────────────────────────────────
    print("\nBuilding L4_shap_dense_full_demo.pkl...")
    demo_shap = {}

    # Every level is offered the full selected cohort. The positive-only levels
    # (L2/L3) then keep whichever of those patients the full SHAP file actually
    # holds. Pre-filtering with a label-derived positive mask was wrong: the full
    # file keys L2/L3 on the TRUE positives for the internal cohort but on the
    # PREDICTED positives for the external one, so a mask built here matches
    # neither convention and drops or invents rows.
    level_map = {
        "L1":          ("int", int_idx),
        "L2":          ("int", int_idx),
        "L3":          ("int", int_idx),
        "L1_external": ("ext", ext_idx),
        "L2_external": ("ext", ext_idx),
        "L3_external": ("ext", ext_idx),
    }

    for level_key, (cohort, orig_indices) in level_map.items():
        if level_key not in shap_d:
            print(f"  {level_key}: not in SHAP file — skip")
            continue

        level = shap_d[level_key]
        demo_level = {}

        level_si = np.array(level.get("sample_indices", []))
        cohort_idx = int_idx if cohort == "int" else ext_idx

        if len(level_si) > 0:
            # Rows are picked by ORIGINAL cohort index, but the demo file has to be
            # keyed by the patient's row in the DEMO cohort, because that is what
            # data_loader passes as sig_idx. The two coincide for L1 only. For the
            # positive-only levels an arange() here silently re-points every
            # explanation at a different patient, so carry the demo index along.
            positions, demo_si = [], []
            for oi in orig_indices:
                pos = np.where(level_si == oi)[0]
                if len(pos) > 0:
                    positions.append(pos[0])
                    demo_si.append(int(np.where(cohort_idx == oi)[0][0]))

            positions = (np.array(positions) if positions
                         else np.array([], dtype=int))

            if len(positions) > 0:
                for mat_key in ["shap_matrix", "X_matrix"]:
                    if mat_key in level:
                        mat = np.array(level[mat_key])
                        demo_level[mat_key] = mat[positions]
                        print(f"  {level_key}.{mat_key}: {mat.shape} → "
                              f"{demo_level[mat_key].shape}")
                demo_level["sample_indices"] = np.array(demo_si, dtype=int)
                print(f"  {level_key}.sample_indices: {list(demo_si)}")
            else:
                print(f"  ⚠ {level_key}: no matching sample_indices")
        else:
            for mat_key in ["shap_matrix", "X_matrix"]:
                if mat_key in level:
                    mat = np.array(level[mat_key])
                    safe = orig_indices[orig_indices < mat.shape[0]]
                    if len(safe) > 0:
                        demo_level[mat_key] = mat[safe]
                        demo_level["sample_indices"] = np.array(
                            [int(np.where(cohort_idx == oi)[0][0]) for oi in safe],
                            dtype=int)
                        print(f"  {level_key}.{mat_key}: {mat.shape} → "
                              f"{demo_level[mat_key].shape}  [positional]")

        for mk in ["base_value", "threshold_used"]:
            if mk in level:
                demo_level[mk] = copy.deepcopy(level[mk])

        demo_shap[level_key] = demo_level

    for gkey in ["feature_names", "metadata"]:
        if gkey in shap_d:
            demo_shap[gkey] = copy.deepcopy(shap_d[gkey])
            print(f"  {gkey}: copied")

    save_pkl(demo_shap, root / "results" / "L4_shap_dense_full_demo.pkl")

    # ─── Build ext_validation_results demo ───────────────────────────────
    print("\nBuilding L4_ext_validation_results_demo.pkl...")
    demo_ext_val = {}
    for key, val in ext_val.items():
        arr = np.array(val)
        if arr.ndim >= 1 and arr.shape[0] == n_ext:
            demo_ext_val[key] = arr[ext_idx]
            print(f"  {key}: {arr.shape} → {demo_ext_val[key].shape}")
        else:
            demo_ext_val[key] = copy.deepcopy(val)
            print(f"  {key}: as-is")

    save_pkl(demo_ext_val, root / "results" /
             "L4_ext_validation_results_demo.pkl")

    # ─── Summary ─────────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("DEMO DATASET CREATION COMPLETE")
    print("═" * 65)
    print(f"  Grid:     2 cohorts × 9 classes × 3 zones = 54 slots")
    print(f"  Filled:   {n_sel_int} internal + {n_sel_ext} external "
          f"= {n_total}")
    print(f"  Skipped:  {len(skipped)}")
    print(f"  ID range: {ANON_START} – {ANON_START + n_total - 1}")

    # Coverage matrix
    print(f"\n── COVERAGE MATRIX ──")
    for cohort_label, cohort_code in [("INTERNAL", "INT"),
                                       ("EXTERNAL", "EXT")]:
        print(f"\n  {cohort_label}:")
        print(f"  {'CLASS':14s}  {'HIGH':>6s}  {'MEDIUM':>6s}  {'LOW':>6s}")
        print(f"  {'─' * 38}")
        for cls in CLASSES_9:
            cells = []
            for z in ZONES:
                found = any(
                    r["cohort"] == cohort_code
                    and r["slot_class"] == cls
                    and r["slot_zone"] == z
                    for r in report_rows
                )
                cells.append("  ✓   " if found else "  ·   ")
            print(f"  {cls:14s}  {''.join(cells)}")

    print(f"\n── FILES CREATED ──")
    for f in [
        "data/dataset_demo.pkl",
        "results/flow_df_demo.pkl",
        "results/ext_flow_df_demo.pkl",
        "results/L4_shap_dense_full_demo.pkl",
        "results/L4_ext_validation_results_demo.pkl",
    ]:
        print(f"  {f}")

    print(f"\n── NEXT STEPS ──")
    print(f"  1. Review selection & coverage above")
    print(f"  2. Apply config.py DEMO_MODE patch")
    print(f"  3. Update .gitignore (allow *_demo.pkl)")
    print(f"  4. git add + commit + push")
    print(f"  5. Deploy on Streamlit Cloud")

    # Save selection report
    if report_rows:
        rpt = pd.DataFrame(report_rows)
        rpt_path = root / "demo_selection_report.csv"
        rpt.to_csv(rpt_path, index=False)
        print(f"\n  ✓ {rpt_path} saved for reference")


if __name__ == "__main__":
    main()