import streamlit as st
from streamlit_back_camera_input import back_camera_input
from fastai.learner import load_learner
from PIL import Image
import io

# Load trained FastAI model
learn = load_learner('./model.pkl')

# Initialize session state if not already
if 'input_sequence' not in st.session_state:
    st.session_state['input_sequence'] = ""
if 'capturing' not in st.session_state:
    st.session_state['capturing'] = True

# Streamlit UI
st.title("Live Gesture Recognition")
st.write("Touch the video area to capture an image and detect gestures.")

# Show current sequence
st.write(f"Current sequence: {st.session_state['input_sequence']}")

# Capture image from back camera
image = back_camera_input()

if image:
    # Convert the BytesIO image to a PIL Image
    img = Image.open(image).convert("RGB")
    img = img.resize((224, 224))  # Resize to match model input size

    # Predict using the model
    pred = learn.predict(img)

    # Show the image and prediction
    st.image(img, caption=f"Prediction: {pred[0]}", use_container_width=True)
    st.success(f"Detected Gesture: {pred[0]}")

    # Concatenate prediction if capturing
    if st.session_state['capturing']:
        st.session_state['input_sequence'] += pred[0]

    # Input controls
    if st.button('Try Again'):
        if len(st.session_state['input_sequence']) > 0:
            st.session_state['input_sequence'] = st.session_state['input_sequence'][:-1]
            st.write(f"Removed last letter. New sequence: {st.session_state['input_sequence']}")

    if st.button('Continue'):
        st.session_state['capturing'] = False
        st.write(f"Final sequence: {st.session_state['input_sequence']}")
        st.session_state['input_sequence'] = ""  # Reset after continue
        st.session_state['capturing'] = True  # Ready to start new input
