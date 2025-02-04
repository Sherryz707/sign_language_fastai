import streamlit as st
from streamlit_back_camera_input import back_camera_input
from fastai.learner import load_learner
from PIL import Image
from pathlib import Path
import pathlib
import sys

temp = pathlib.PosixPath
pathlib.WindowsPath = pathlib.PosixPath

# # Load trained FastAI model
learn = load_learner('./model.pkl')

# Streamlit UI
st.title("Live Gesture Recognition")
st.write("Touch the video area to capture an image and detect gestures.")

# Back camera input
image = back_camera_input()

if image:
    # Open the image
    # st.image(image)
    # Convert BytesIO to PIL Image
    img = Image.open(image).convert("RGB")
    print("image",image)
    img = img.resize((224, 224))  # Resize to match model input size

    # Predict using the model
    pred = learn.predict(img)

    # # Display the image with prediction
    st.image(img, caption=f"Prediction: {pred[0]}", use_container_width=True)

    # Show prediction results
    st.success(f"Detected Gesture: {pred[0]}")
