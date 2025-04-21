import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

st.set_page_config(page_title="Weld Classifier", layout="centered")

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('weld_classifier.h5')  # Rename based on your actual file
    return model

model = load_model()

# Class names
class_names = [
    'Burn-through', 'Crack', 'Excess Reinforcement', 'Good Welding',
    'Overlap', 'Porosity', 'Spatters', 'Undercut'
]

st.title("🛠️ Weld Classification System")
st.write("Upload a welding image and let the model predict the defect type.")

file = st.file_uploader("📤 Choose a weld image (JPG or PNG)", type=["jpg", "png"])

def import_and_predict(image_data, model):
    size = (224, 224)  # Use your training image size
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image).astype('float32') / 255.0
    img_array = img_array[np.newaxis, ...]
    prediction = model.predict(img_array)
    return prediction

if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    prediction = import_and_predict(image, model)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.success(f"🔍 **Prediction:** {predicted_class}")
    st.info(f"📊 **Confidence:** {confidence:.2%}")
else:
    st.warning("Please upload an image to get a prediction.")
