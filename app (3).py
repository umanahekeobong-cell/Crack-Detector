import streamlit as st
from PIL import Image
import os
import random
import tensorflow as tf
import numpy as np
import glob

# --- Configuration Variables ---
# When deploying from GitHub to Streamlit, paths are relative to the app.py file
MODEL_PATH = 'models/mobilenetv3_transfer.keras' # Path to your trained model
MODEL_TEST_ACCURACY = 81.60 # Placeholder: Replace with actual test accuracy from notebook evaluation
app_class_names = ['Apple_Formalin-mixed', 'Apple_Fresh'] # Explicitly define class names for the app

# --- Load Model (Best Performing: MobileNetV3 Fine-Tuned) ---
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model: {e}. Please ensure '{MODEL_PATH}' is available in the 'models' directory.")
    model = None # Set model to None if loading fails

# --- Prediction Function ---
def predict_image(image, model, class_names, image_size=(128, 128)):
    if model is None:
        return None, None, None, None # Indicate model not loaded

    img_resized = image.resize(image_size)
    img_array = tf.keras.utils.img_to_array(img_resized) # [0, 255] float32
    img_array = np.expand_dims(img_array, axis=0) # (1, H, W, 3)

    # The model was likely trained with MobileNetV3 preprocessing, which scales to [-1, 1]
    preprocessed_img = tf.keras.applications.mobilenet_v3.preprocess_input(img_array)

    predictions = model.predict(preprocessed_img)
    # Return the full softmax probabilities
    softmax_scores = tf.nn.softmax(predictions[0]).numpy()
    predicted_class_index = np.argmax(softmax_scores)
    predicted_class_name = class_names[predicted_class_index]
    confidence = softmax_scores[predicted_class_index] * 100
    return predicted_class_name, predicted_class_index, confidence, softmax_scores


# --- Streamlit UI ---
st.set_page_config(
    page_title="🍎 Apple Disease Classifier",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': 'https://www.extremely.cool-app.com/help', # Placeholder link
        'Report a bug': "https://www.extremely.cool-app.com/bug", # Placeholder link
        'About': "# This is a classifier for apple diseases."
    }
)

st.title("🍎 Apple Disease Classifier")
st.markdown("Upload an image of an apple to classify its health status (Formalin-mixed or Fresh).")
st.markdown("---")

# Section 1: Upload Your Own Image (now the primary section)
st.subheader("1. Upload Your Own Image")
st.markdown("Upload an image from your computer or use a camera if available.")

uploaded_file = st.file_uploader(
    "Choose an image file (JPG, JPEG, PNG, WEBP, GIF, BMP, TIFF)",
    type=["jpg", "jpeg", "png", "webp", "gif", "bmp", "tiff"],
    help="Drag and drop your image here or click to browse files. On mobile, this may allow camera access."
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width='stretch')
        st.success("Image uploaded successfully!")

        if st.button("Classify Uploaded Image"):
            if model:
                _, _, _, softmax_scores = predict_image(image, model, app_class_names, image_size=(128, 128))
                st.subheader("Prediction for Uploaded Image:")

                cols = st.columns(len(app_class_names))
                max_score_index = np.argmax(softmax_scores)

                for i, score in enumerate(softmax_scores):
                    with cols[i]:
                        st.write(f"**{app_class_names[i]}**")
                        display_value = f"{score * 100:.2f}% {'✅' if i == max_score_index else ''}"
                        st.progress(float(score))
                        st.write(display_value)
            else:
                st.error("Model not loaded for prediction.")
    except Exception as e:
        st.error(f"Error loading image: {e}")
else:
    st.info("Awaiting image upload. Please upload a file to proceed with classification.")

st.markdown("---")

# Section 2: Model Performance Summary (renumbered)
st.subheader("2. Model Performance Summary")
st.markdown("Here's a brief overview of the model's performance on a separate test set:")

st.metric(label="Model Test Accuracy", value=f"{MODEL_TEST_ACCURACY:.2f}%")

if MODEL_TEST_ACCURACY > 80:
    st.success("This model shows good overall accuracy. Great job!")
elif MODEL_TEST_ACCURACY > 60:
    st.info("The model has decent accuracy, but there might be room for improvement.")
else:
    st.warning("The model's accuracy is relatively low. Consider further training or model adjustments.")

st.markdown("*(Note: This accuracy score is a static value provided for context. For a live assessment, you would need a more sophisticated evaluation setup.)*")
