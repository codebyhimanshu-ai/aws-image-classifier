import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="AWS Image Classifier",
    page_icon="🐶",
    layout="centered"
)

st.title("🐱🐶 AWS Image Classifier")

st.write("Upload a Cat or Dog image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    st.image(uploaded_file, use_container_width=True)

    if st.button("Predict"):

        with st.spinner("Predicting..."):

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type,
                )
            }

            response = requests.post(API_URL, files=files)

            if response.status_code == 200:

                result = response.json()

                st.success("Prediction Completed")

                st.write("### Result")

                st.json(result)

            else:

                st.error("API Error")