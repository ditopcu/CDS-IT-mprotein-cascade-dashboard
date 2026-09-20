"""Put the app package root on sys.path so the tests import the live modules.

The app is a flat set of modules at the repo root, not an installed package, so pytest's
own rootdir insertion is not enough when the tests live in tests/.
"""
import os
import sys

APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

# Keep the tests off the full (gitignored) patient pickles regardless of the caller's
# environment — everything here works against the committed demo artifacts or synthetic
# fixtures. Set before any app module reads it at import time.
os.environ.setdefault("DEMO_MODE", "true")
os.environ.setdefault("ALICANTE", "false")

import matplotlib                                                   # noqa: E402
matplotlib.use("Agg")                                               # headless figures
