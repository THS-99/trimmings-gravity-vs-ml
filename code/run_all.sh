#!/usr/bin/env bash
# Download the raw data, build and validate the panel and regenerate every table and figure.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 code/01_download_comext.py
python3 code/02_download_comexstat.py
python3 code/03_download_wdi.py
python3 code/04_download_cepii.py
python3 code/05_build_panel.py
python3 code/06_validate_panel.py
python3 code/14_descriptives.py
python3 code/08_ppml.py
python3 code/09_ml.py
python3 code/10_evaluate.py
python3 code/11_explainability.py
python3 code/12_rq3.py
python3 code/13_robustness.py
echo "All results regenerated under results/"
