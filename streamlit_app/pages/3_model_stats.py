import streamlit as st
import json
import os

st.set_page_config(page_title="Model Stats", page_icon="📊")

st.header("📊 Model Statistics")

# Load training results if available
results_path = "training_results.json"
if os.path.exists(results_path):
    with open(results_path, "r") as f:
        results = json.load(f)
    st.json(results)
else:
    st.warning("No training results found. Please run training first.")
