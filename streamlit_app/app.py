import streamlit as st

st.set_page_config(
    page_title="Visual Anomaly Detection",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Visual Anomaly Detection System")

st.markdown("""
### Welcome to the Industrial Anomaly Detection Dashboard

This application allows you to:
- **Detect anomalies** in images using deep learning models.
- **Analyze batches** of images for quality control.
- **View model statistics** and performance metrics.

#### How to use
1. Navigate to **Upload & Predict** to test single images.
2. Use **Batch Analysis** for processing multiple files.
3. Check **Model Stats** for performance insights.

#### Architecture
- **Backend**: AWS Lambda + TensorFlow Lite
- **Frontend**: Streamlit
- **Models**: CNN, ResNet, Autoencoder
""")

st.sidebar.success("Select a page above.")
