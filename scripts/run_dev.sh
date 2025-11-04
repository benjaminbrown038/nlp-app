#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
export ENV=dev
# set ENABLE_TRANSFORMERS=1 to use HF pipelines (after installing optional reqs)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000