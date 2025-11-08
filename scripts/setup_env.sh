#!/usr/bin/env bash

# Use python3 by default (no args needed)
PYTHON=python3

echo "Using Python executable: $PYTHON"

$PYTHON -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Virtual environment created and dependencies installed!"
