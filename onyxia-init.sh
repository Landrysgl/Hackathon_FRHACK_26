#!/bin/bash

echo "===== PYTHON ====="
which python
python --version

echo "===== PIP ====="
python -m pip --version

echo "===== INSTALLATION ====="
python -m pip install --user \
    pandas \
    requests \
    pillow \
    tqdm \
    boto3 \
    dash \
    kaleido \
    streamlit \
    ultralytics

echo "===== VERIFICATION ====="
python -c "import boto3; print('boto3 OK')"
python -c "import dash; print('dash OK')"
python -c "import kaleido; print('kaleido OK')"