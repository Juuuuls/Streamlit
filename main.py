import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

st.title("Weld Classifier")

# Upload model file
model_file = st.file_uploader("Upload HDF5 model", type=["h5"])
if model_file is not None:
    model = tf.keras.models.load_model(model_file)
    st.success("Model loaded successfully!")

    # Upload image
    img_file = st.file_uploader("Upload weld image", type=["jpg", "png"])
    if img_file:
        image = Image.open(img_file).convert("RGB")
        st.image(image, use_column_width=True)

        def import_and_predict(image_data, model):
            size = (224, 224)
            image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
            img = np.asarray(image).astype('float32') / 255.0
            img = img[np.newaxis, ...]
            return model.predict(img)

        prediction = import_and_predict(image, model)
        class_names = [
            'Burn-through', 'Crack', 'Excess Reinforcement', 'Good Welding',
            'Overlap', 'Porosity', 'Spatters', 'Undercut'
        ]
        st.success(f"Prediction: {class_names[np.argmax(prediction)]}")
else:
    st.warning("Please upload your HDF5 model to proceed.")
