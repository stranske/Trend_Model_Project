#!/usr/bin/env bash
set -euo pipefail
python -c "import streamlit as st; print('Streamlit version:', st.__version__)" >/dev/null 2>&1 || pip install streamlit
exec streamlit run streamlit_app/app.py
