#!/usr/bin/env bash
# Build "UV-Vis Kinetics.app" (self-contained, no terminal window, no Python needed to run).
# Run ON A MAC from the folder containing uvvis_kinetics.py:   chmod +x build_mac.sh && ./build_mac.sh
# Result: dist/UV-Vis Kinetics.app  (also dist/UV-Vis-Kinetics.zip for sharing)
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
# The bundled app uses the Tk that ships with this Python, so it must have tkinter.
"$PY" -c "import tkinter" 2>/dev/null || {
  echo "This Python has no tkinter. Install Python from python.org (includes Tk)" >&2
  echo "or run: brew install python-tk" >&2
  exit 1
}

# Isolated build environment: only what the app needs gets bundled.
"$PY" -m venv .buildenv
source .buildenv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet numpy matplotlib pyinstaller

# --windowed : GUI app, no Terminal window     --onedir : faster launch than one-file
pyinstaller --noconfirm --clean --windowed --onedir \
  --name "UV-Vis Kinetics" \
  --osx-bundle-identifier com.local.uvviskinetics \
  --hidden-import matplotlib.backends.backend_tkagg \
  --hidden-import PIL._tkinter_finder \
  --exclude-module matplotlib.tests \
  uvvis_kinetics.py

# Ad-hoc sign so Apple Silicon will launch it, and zip for sharing.
codesign --force --deep --sign - "dist/UV-Vis Kinetics.app" || true
ditto -c -k --keepParent "dist/UV-Vis Kinetics.app" dist/UV-Vis-Kinetics.zip

echo "Built: dist/UV-Vis Kinetics.app"
echo "First launch: right-click > Open (unsigned app). If a downloaded copy is blocked:"
echo "  xattr -dr com.apple.quarantine \"dist/UV-Vis Kinetics.app\""
