import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def display_prediction(image, result):
    """
    Displays image and prediction result.
    """
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
    with col2:
        st.subheader("Prediction Result")
        if result.get("error"):
            st.error(result["error"])
        else:
            is_anomaly = result.get("is_anomaly", False)
            confidence = result.get("confidence", 0.0)
            
            if is_anomaly:
                st.error(f"ANOMALY DETECTED ({confidence:.2%})")
            else:
                st.success(f"NORMAL ({confidence:.2%})")
            
            st.json(result)
