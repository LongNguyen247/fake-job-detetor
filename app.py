import streamlit as st
from model_utils import predict_job_post

st.title("Fake Job Post Detector")

user_input = st.text_area("Paste Job Description Here:")

if st.button("Analyze Job Post"):
    if user_input.strip():
        label, confidence = predict_job_post(user_input)
        st.write(f"**Prediction:** {label}")
        st.write(f"**Confidence:** {confidence:.2f}")
    else:
        st.write("Please enter job description text.")
