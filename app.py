import os
import gdown
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==============================
# CONFIG
# ==============================
MODEL_PATH = "best_model.h5"   # atau "model.h5"
GDRIVE_FILE_ID = "1joIstlMa2sl-BnjwfqQLwr-OxbTQfczm" #https://drive.google.com/file/d/1joIstlMa2sl-BnjwfqQLwr-OxbTQfczm/view?usp=sharing

IMG_SIZE = 224   # sesuaikan dengan model kamu
CLASS_NAMES = ["Cancer", "Normal"]   # sesuaikan label kamu

# ==============================
# DOWNLOAD MODEL FROM GDRIVE
# ==============================
def download_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        st.info("📥 Downloading model from Google Drive... please wait")
        gdown.download(url, MODEL_PATH, quiet=False)
        st.success("✅ Model downloaded successfully!")

# ==============================
# LOAD MODEL (CACHED)
# ==============================
@st.cache_resource
def load_model():
    download_model()
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# ==============================
# IMAGE PREPROCESSING
# ==============================
def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ==============================
# STREAMLIT UI
# ==============================
st.set_page_config(page_title="Oral Cancer Detection", layout="centered")

st.title("🦷 Oral Cancer Detection App")
st.write("Upload an oral image and the model will predict whether it is cancer or normal.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    processed_image = preprocess_image(image)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing image..."):
            prediction = model.predict(processed_image)
            class_index = np.argmax(prediction)
            confidence = float(np.max(prediction))

            st.success(f"✅ Prediction: {CLASS_NAMES[class_index]}")
            st.info(f"📊 Confidence: {confidence:.2%}")
