"""
config.py – Central configuration for the IFE M-Protein CDS Dashboard.
All constants, color palettes, label maps, and layout settings live here.
"""
import os
from pathlib import Path

# ─── Demo mode toggle ────────────────────────────────────
# Default: True (Streamlit Cloud uses demo data out of the box)
# Local with full data: set DEMO_MODE=false in environment
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"

# ─── Paths ───────────────────────────────────────────────
DATA_DIR = Path("data")
RES_DIR  = Path("results")

_SUFFIX = "_demo" if DEMO_MODE else ""

DATASET_FILE   = DATA_DIR / f"dataset{_SUFFIX}.pkl"
FEATURE_DICT   = DATA_DIR / "feature_dictionary.pkl"       # no demo variant
FLOWDF_FILE    = RES_DIR  / f"flow_df{_SUFFIX}.pkl"
SHAP_FILE      = RES_DIR  / f"L4_shap_dense_full{_SUFFIX}.pkl"
EXTVAL_FILE    = RES_DIR  / f"L4_ext_validation_results{_SUFFIX}.pkl"

# ─── 9-class label map ───────────────────────────────────
D9 = {
    0: 'NEGATIVE',   1: 'IGG_KAPPA',  2: 'IGG_LAMBDA',
    3: 'IGA_KAPPA',  4: 'IGA_LAMBDA', 5: 'IGM_KAPPA',
    6: 'IGM_LAMBDA', 7: 'FREE_KAPPA', 8: 'FREE_LAMBDA',
}

# ─── Pretty display names ────────────────────────────────
PRETTY_LABELS = {
    'NEGATIVE':    'Negative',
    'IGG_KAPPA':   'IgG-κ',
    'IGG_LAMBDA':  'IgG-λ',
    'IGA_KAPPA':   'IgA-κ',
    'IGA_LAMBDA':  'IgA-λ',
    'IGM_KAPPA':   'IgM-κ',
    'IGM_LAMBDA':  'IgM-λ',
    'FREE_KAPPA':  'Free-κ',
    'FREE_LAMBDA': 'Free-λ',
}

def pretty(label):
    """Convert raw class label to display format."""
    return PRETTY_LABELS.get(label, label)

# ─── Channel definitions ─────────────────────────────────
CHANNELS   = ['ELP', 'IgG', 'IgA', 'IgM', 'Kappa', 'Lambda']
CH_COLORS  = ['#1a1a1a', '#C0392B', '#8E44AD', '#2980B9', '#27AE60', '#E67E22']

# ─── Protein-region boundaries (data-point indices) ──────
#     Used for background shading on signal plots
PROTEIN_REGIONS = [
    ('Albumin',      0,   40, '#E3F2FD'),
    ('α1',          40,   70, '#FFF3E0'),
    ('α2',          70,  110, '#FCE4EC'),
    ('β1',         110,  155, '#E8F5E9'),
    ('β2',         155,  195, '#F3E5F5'),
    ('γ',          195,  300, '#FFF9C4'),
]

# ─── Confidence-zone styling ─────────────────────────────
ZONE_COLORS  = {'HIGH': '#1A9641', 'MEDIUM': '#F46D43', 'LOW': '#D73027'}
ZONE_THRESHOLDS = {'HIGH': 0.70, 'MEDIUM': 0.30}  # >= threshold

# ─── SHAP color palette ─────────────────────────────────
SHAP_POS_COLOR = '#B2182B'
SHAP_NEG_COLOR = '#2166AC'

# ─── Conformal prediction defaults ───────────────────────
CP_ALPHA = 0.05

# ─── PDF settings ────────────────────────────────────────
PDF_DPI      = 300
PDF_FONT     = 'Helvetica'
PDF_MARGIN   = 36

# ─── Plotly / Matplotlib shared settings ─────────────────
PLOT_DPI     = 300
FIG_BG       = '#FAFAFA'
GRID_COLOR   = '#E8E8E8'
FONT_FAMILY  = 'Inter, Segoe UI, Arial, sans-serif'


# ═══════════════════════════════════════════════════════════════
#  REFLEX TESTING DECISION MATRIX (Table S1)
# ═══════════════════════════════════════════════════════════════

# Panel A: Universal baseline — all M-protein positive predictions
UNIVERSAL_BASELINE = [
    ("sFLC with κ/λ ratio", "IMWG SLiM criterion; abnormal in 100% LCMM, 80–95% AL amyloidosis"),
    ("Quantitative Ig (IgG, IgA, IgM)", "Immunoparesis assessment; IgA quantification when β-migration"),
    ("CBC with differential", "Anemia screening (Hb <10 = CRAB); cytopenias"),
    ("CMP (Cr/eGFR, Ca, albumin, hepatic)", "CRAB criteria; ISS staging"),
    ("β2-microglobulin", "ISS/R-ISS staging (Stage III: ≥5.5 mg/L)"),
    ("Serum LDH", "R-ISS staging; elevated = aggressive biology"),
    ("24h urine with UPEP/uIFE", "Bence Jones detection; AL amyloidosis evaluation"),
]

# Panel B: Isotype × Zone reflex matrix
# Keys: class group → zone → {gel_ife, tests, guidance}
# gel_ife levels: "Not required", "Consider", "Recommended", "Mandatory"
REFLEX_MATRIX = {
    "IgG": {
        "HIGH": {
            "gel_ife": "Not required",
            "tests": [],
            "guidance": "Consider BM biopsy based on clinical findings (defer if M-protein <1.5 g/dL, normal sFLC ratio, no CRAB per Mayo MGUS risk model)",
        },
        "MEDIUM": {
            "gel_ife": "Recommended",
            "tests": [],
            "guidance": "Consider BM biopsy; evaluate with hematology",
        },
        "LOW": {
            "gel_ife": "Mandatory",
            "tests": ["Consider IgD/IgE testing (exclude rare isotype)"],
            "guidance": "Hematology referral for BM with FISH panel; imaging (PET-CT or LDWBCT) if indicated",
        },
    },
    "IgA": {
        "HIGH": {
            "gel_ife": "Consider",
            "tests": ["Nephelometric IgA (essential)", "Consider HLC (Hevylite IgA-κ/IgA-λ)"],
            "guidance": "Consider BM biopsy (non-IgG isotype = independent MGUS progression risk factor)",
        },
        "MEDIUM": {
            "gel_ife": "Recommended",
            "tests": ["Nephelometric IgA (essential)", "HLC recommended"],
            "guidance": "Hematology referral recommended for BM evaluation",
        },
        "LOW": {
            "gel_ife": "Mandatory",
            "tests": ["Nephelometric IgA (essential)", "HLC recommended", "Consider IgD/IgE testing"],
            "guidance": "Hematology referral for BM with FISH panel; imaging if indicated",
        },
    },
    "IgM": {
        "HIGH": {
            "gel_ife": "Consider",
            "tests": ["Cryoglobulin screen", "Serum viscosity if IgM >4 g/dL"],
            "guidance": "Evaluate for Waldenström; consider BM with MYD88 L265P, flow cytometry/IHC, CT imaging",
        },
        "MEDIUM": {
            "gel_ife": "Recommended",
            "tests": ["Cryoglobulin screen", "Serum viscosity", "Cold agglutinins if anemia"],
            "guidance": "Hematology referral; BM with MYD88 + flow cytometry/IHC; CT chest/abdomen/pelvis",
        },
        "LOW": {
            "gel_ife": "Mandatory",
            "tests": ["Cryoglobulin screen", "Serum viscosity", "Cold agglutinins",
                       "Consider IgD/IgE testing", "Hepatitis B/C serology"],
            "guidance": "Hematology referral for BM with FISH + MYD88 + flow; CT mandatory",
        },
    },
    "FREE": {
        "HIGH": {
            "gel_ife": "Recommended",
            "tests": ["IgD/IgE testing (mandatory)", "NT-proBNP and troponin (AL screening)",
                       "Renal panel (Cr, eGFR, urinalysis)"],
            "guidance": "Consider BM biopsy; evaluate for AL amyloidosis with organ involvement",
        },
        "MEDIUM": {
            "gel_ife": "Mandatory",
            "tests": ["IgD/IgE testing (mandatory)", "NT-proBNP and troponin",
                       "Renal panel with protein/creatinine ratio"],
            "guidance": "Hematology referral for BM; consider echocardiography if indicated",
        },
        "LOW": {
            "gel_ife": "Mandatory",
            "tests": ["IgD/IgE testing (mandatory)", "NT-proBNP and troponin",
                       "Renal panel with protein/creatinine ratio", "Alkaline phosphatase"],
            "guidance": "Hematology referral for BM with FISH + Congo red; echocardiography recommended; consider fat pad aspirate",
        },
    },
    "NEGATIVE": {
        "HIGH": {
            "gel_ife": "Not required",
            "tests": ["sFLC recommended"],
            "guidance": "No further workup if sFLC normal; repeat SPEP in 3–6 months only if clinical suspicion",
        },
        "MEDIUM": {
            "gel_ife": "Consider",
            "tests": ["sFLC (mandatory)", "Quantitative Ig recommended"],
            "guidance": "If sFLC abnormal: initiate full baseline + gel IFE; technician review of IT pattern",
        },
        "LOW": {
            "gel_ife": "Recommended",
            "tests": ["sFLC (mandatory)", "Quantitative Ig (mandatory)"],
            "guidance": "Full baseline regardless of sFLC; expert review mandatory; consider MASS-FIX if available",
        },
    },
}

# Helper: map 9-class pred_class to reflex group key
def reflex_group(pred_class):
    if pred_class == 'NEGATIVE':
        return 'NEGATIVE'
    if pred_class.startswith('IGG'):
        return 'IgG'
    if pred_class.startswith('IGA'):
        return 'IgA'
    if pred_class.startswith('IGM'):
        return 'IgM'
    if pred_class.startswith('FREE'):
        return 'FREE'
    return 'NEGATIVE'  # fallback


def get_reflex(pred_class, zone):
    """Return (gel_ife, tests, guidance, show_baseline) for a given prediction."""
    grp = reflex_group(pred_class)
    entry = REFLEX_MATRIX.get(grp, {}).get(zone, REFLEX_MATRIX[grp]['LOW'])
    show_baseline = (grp != 'NEGATIVE')
    return entry['gel_ife'], entry['tests'], entry['guidance'], show_baseline

def generate_reflex_template(path="data/reflex_matrix.xlsx"):
    """Generate a template Excel from hardcoded rules. Run once."""
    import pandas as pd

    # Sheet 1: Baseline
    df_base = pd.DataFrame(UNIVERSAL_BASELINE, columns=["test", "rationale"])

    # Sheet 2: Reflex Matrix
    rows = []
    for grp, zones in REFLEX_MATRIX.items():
        for zone, entry in zones.items():
            rows.append({
                "class_group": grp,
                "zone": zone,
                "gel_ife": entry["gel_ife"],
                "tests": "; ".join(entry["tests"]) if entry["tests"] else "",
                "guidance": entry["guidance"],
            })
    df_matrix = pd.DataFrame(rows)

    # Sheet 3: Workflow
    df_workflow = pd.DataFrame([
        {"zone": "HIGH",   "action": "Auto-verify and report with CDS summary"},
        {"zone": "MEDIUM", "action": "Technician verification required before reporting"},
        {"zone": "LOW",    "action": "Expert referral with full SHAP panel; manual review mandatory"},
    ])

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df_base.to_excel(writer, sheet_name="Baseline", index=False)
        df_matrix.to_excel(writer, sheet_name="Reflex_Matrix", index=False)
        df_workflow.to_excel(writer, sheet_name="Workflow", index=False)

    return path