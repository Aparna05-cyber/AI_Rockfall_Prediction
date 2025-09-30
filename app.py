import streamlit as st
import os

# Title
st.title("Rockfall Prediction DEM")

# Run your prediction script
if st.button("Run Rockfall Prediction"):
    os.system("python rockfall_prediction.py")
    st.success("Prediction complete! Check outputs folder.")

# Show terrain results
if os.path.exists("outputs/terrain_risk_map.html"):
    st.components.v1.html(open("outputs/terrain_risk_map.html", encoding="utf-8").read(), height=600)


if os.path.exists("outputs/heatmap.png"):
    st.image("outputs/heatmap.png", caption="Risk Heatmap")