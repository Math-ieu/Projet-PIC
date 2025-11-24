import streamlit as st
from utils.api_client import predict_anomaly
from utils.visualizations import display_prediction
from PIL import Image

st.set_page_config(page_title="Upload & Predict", page_icon="📤")

st.header("📤 Upload Image for Anomaly Detection")

# Category selection
categories = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
selected_category = st.selectbox("Select Category", categories)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    
    if st.button("Predict Anomaly"):
        with st.spinner("Analyzing image..."):
            # Reset file pointer
            uploaded_file.seek(0)
            result = predict_anomaly(uploaded_file, category=selected_category)
            display_prediction(image, result)
