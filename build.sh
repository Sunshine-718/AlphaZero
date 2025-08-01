#!/usr/bin/env bash
set -e

if command -v pip3 &>/dev/null; then
    PIP=pip3
else
    PIP=pip
fi

if command -v python3 &>/dev/null; then
    PYTHON=python3
else
    PYTHON=python
fi

$PIP install -r requirements.txt

$PYTHON setup.py build_ext --inplace

echo "Done."
