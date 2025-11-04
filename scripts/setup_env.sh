#!/usr/bin/env bash
set -euo pipefail


PY=${1:-"3.10"}


python${PY} -m venv .venv
source .venv/bin/activate


pip install --upgrade pip
pip install -r requirements.txt


# Optional heavy deps
if [[ "${ENABLE_TRANSFORMERS:-0}" == "1" ]]; then
pip install -r requirements-optional.txt
fi


echo "\n[ ok ] Environment ready. To activate: source .venv/bin/activate"