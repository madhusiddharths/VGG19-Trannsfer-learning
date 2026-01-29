import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
import numpy as np
from PIL import Image
import os

# Set page config
st.set_page_config(page_title="Horse or Human Classifier", layout="centered")

st.title("🐎 Horse or Human Image Classifier")
st.markdown("""
Upload an image and the VGG19 model will tell you if it's a **horse** or a **human**.
""")

# Load the model
MODEL_PATH = 'model_new.h5'

@st.cache_resource
def load_classification_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    else:
        st.error(f"Model file '{MODEL_PATH}' not found. Please train the model first by running `python train.py`.")
        return None

model = load_classification_model()

def predict(img, model):
    # Resize to target size for VGG19
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    preds = model.predict(x)
    return preds

upload = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if upload is not None and model is not None:
    img = Image.open(upload)
    # Ensure image is RGB (converts RGBA/grayscale to RGB)
    img = img.convert("RGB")
    
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    with st.spinner('Thinking...'):
        predictions = predict(img, model)
        # Assuming index 0 is horse and index 1 is human based on typical data gen sorting (H then H?) 
        # Actually ImageDataGenerator sorts alphabetically: 'horses', 'humans'
        classes = ['Horse', 'Human']
        result_idx = np.argmax(predictions[0])
        confidence = predictions[0][result_idx] * 100
        
        st.success(f"Prediction: **{classes[result_idx]}** ({confidence:.2f}% confidence)")
        
        # Display probability bar
        st.write("Confidence Scores:")
        for idx, label in enumerate(classes):
            st.write(f"{label}")
            st.progress(float(predictions[0][idx]))
else:
    if model is not None:
        st.info("Waiting for image upload...")
