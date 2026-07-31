import streamlit as st
from PIL import Image
import os
import random
import tensorflow as tf
import numpy as np
import glob
import sys # Import sys for direct stderr output

# --- Configuration Variables ---
# When deploying from GitHub to Streamlit, paths are relative to the app.py file
MODEL_PATH = 'models/mobilenetv3_transfer.keras' # Path to your trained model
MODEL_TEST_ACCURACY = 81.60 # Placeholder: Replace with actual test accuracy from notebook evaluation
app_class_names = ['Apple_Formalin-mixed', 'Apple_Fresh'] # Explicitly define class names for the app

# --- DIAGNOSTICS START (will print to stderr for logging) ---
print("--- Streamlit App Debug Info ---", file=sys.stderr)
current_dir = os.getcwd()
print(f"Current Working Directory: {current_dir}", file=sys.stderr)
model_full_path_abs = os.path.join(current_dir, MODEL_PATH)
print(f"Expected Model Full Absolute Path: {model_full_path_abs}", file=sys.stderr)
print(f"Checking relative MODEL_PATH: {MODEL_PATH}", file=sys.stderr)

if os.path.exists('models'):
    print("'models' directory exists relative to app.py.", file=sys.stderr)
else:
    print("'models' directory DOES NOT exist relative to app.py.", file=sys.stderr)

if os.path.exists(MODEL_PATH):
    print(f"Model file '{MODEL_PATH}' exists (relative path check successful).", file=sys.stderr)
else:
    print(f"Model file '{MODEL_PATH}' DOES NOT exist (relative path check failed).", file=sys.stderr)

print("--- End Debug Info ---", file=sys.stderr)
# --- DIAGNOSTICS END ---


# --- Load Model (Best Performing: MobileNetV3 Fine-Tuned) ---
import os
import gdown
import streamlit as st
import tensorflow as tf

# Define folder and file path
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "mobilenetv3_transfer.keras")

# PASTE YOUR GOOGLE DRIVE FILE ID HERE
GDRIVE_FILE_ID = "YOUR_FILE_ID_HERE" 

@st.cache_resource
def load_model_from_drive():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model weights..."):
            url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
            
    return tf.keras.models.load_model(MODEL_PATH)

# Load your model
model = load_model_from_drive()

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
                st.error("Model not loaded for prediction. Please check the model path and file integrity.")
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
