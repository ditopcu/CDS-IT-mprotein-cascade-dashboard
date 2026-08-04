"""
Vendored frozen algorithm code from the model repository
(github.com/ditopcu/CDS-IT-mprotein-cascade, `src/`).

These modules are copied **byte-identically** so the deployed dashboard runs the exact
feature extraction, cascade inference and confidence math used for the published results,
without depending on a sibling checkout at runtime. Do not edit them here; edit upstream
and re-copy. Expected md5s are listed in `config.CASCADE_SRC_MD5` and shown in the app's
Model & integrity panel.

Vendored modules: features.py, cascade.py, confidence.py, constants.py, calibration.py
(the full import closure of the upload inference path).
"""
