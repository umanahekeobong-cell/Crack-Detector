import glob
import json
import os
import sys  # Import sys for direct stderr output
import tempfile
import zipfile
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# --- Configuration Variables ---
MODEL_PATH = "models/mobilenetv3_transfer.keras"  # Path to your trained model
MODEL_TEST_ACCURACY = 81.60  # Static test accuracy score
app_class_names = ["Apple_Formalin-mixed", "Apple_Fresh"]

# --- DIAGNOSTICS START ---
print("--- Streamlit App Debug Info ---", file=sys.stderr)
current_dir = os.getcwd()
print(f"Current Working Directory: {current_dir}", file=sys.stderr)
model_full_path_abs = os.path.join(current_dir, MODEL_PATH)
print(
    f"Expected Model Full Absolute Path: {model_full_path_abs}", file=sys.stderr
)

if os.path.exists("models"):
  print("'models' directory exists relative to app.py.", file=sys.stderr)
else:
  print("'models' directory DOES NOT exist relative to app.py.", file=sys.stderr)

if os.path.exists(MODEL_PATH):
  print(
      f"Model file '{MODEL_PATH}' exists (relative path check successful).",
      file=sys.stderr,
  )
else:
  print(
      f"Model file '{MODEL_PATH}' DOES NOT exist (relative path check failed).",
      file=sys.stderr,
  )

print("--- End Debug Info ---", file=sys.stderr)
# --- DIAGNOSTICS END ---


# --- Custom Model Loader (Bypasses quantization_config Error) ---
@st.cache_resource
def load_fixed_keras_model(model_path):
  """Loads a .keras model while stripping incompatible layer configurations

  (like 'quantization_config') created by newer Keras versions.
  """
  if not os.path.exists(model_path):
    return None

  # Check if the file is a standard Keras zip archive
  if zipfile.is_zipfile(model_path):
    try:
      # Create a temporary modified copy of the zip archive
      with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp_file:
        tmp_path = tmp_file.name

      with (
          zipfile.ZipFile(model_path, "r") as zin,
          zipfile.ZipFile(tmp_path, "w") as zout,
      ):
        for item in zin.infolist():
          buffer = zin.read(item.filename)
          if item.filename == "config.json":
            # Parse JSON and recursively remove 'quantization_config'
            config_data = json.loads(buffer.decode("utf-8"))

            def strip_quantization(obj):
              if isinstance(obj, dict):
                obj.pop("quantization_config", None)
                for v in obj.values():
                  strip_quantization(v)
              elif isinstance(obj, list):
                for item in obj:
                  strip_quantization(item)

            strip_quantization(config_data)
            buffer = json.dumps(config_data).encode("utf-8")

          zout.writestr(item, buffer)

      # Load model from the patched temporary file
      loaded_model = tf.keras.models.load_model(tmp_path, compile=False)

      # Cleanup temp file
      os.remove(tmp_path)
      return loaded_model
    except Exception as e:
      print(f"Fallback to standard load failed: {e}", file=sys.stderr)

  # Fallback to standard loading if not a zip file or if patching is unnecessary
  return tf.keras.models.load_model(model_path, compile=False)


# Load Model
try:
  model = load_fixed_keras_model(MODEL_PATH)
except Exception as e:
  st.error(f"Error loading model: {e}")
  print(f"MODEL LOAD FAILED: {e}", file=sys.stderr)
  model = None


# --- Prediction Function ---
def predict_image(image, model, class_names, image_size=(128, 128)):
  if model is None:
    return None, None, None, None

  img_resized = image.resize(image_size)
  img_array = tf.keras.utils.img_to_array(img_resized)  # [0, 255] float32
  img_array = np.expand_dims(img_array, axis=0)  # (1, H, W, 3)

  # Preprocess inputs using MobileNetV3 expectations
  preprocessed_img = tf.keras.applications.mobilenet_v3.preprocess_input(
      img_array
  )

  predictions = model.predict(preprocessed_img)

  # Extract raw predictions or Apply Softmax
  if predictions.shape[-1] == 1:
    # Handle Binary Sigmoid output if applicable
    prob = float(predictions[0][0])
    softmax_scores = np.array([1 - prob, prob])
  else:
    softmax_scores = tf.nn.softmax(predictions[0]).numpy()

  predicted_class_index = int(np.argmax(softmax_scores))
  predicted_class_name = class_names[predicted_class_index]
  confidence = softmax_scores[predicted_class_index] * 100

  return (
      predicted_class_name,
      predicted_class_index,
      confidence,
      softmax_scores,
  )


# --- Streamlit UI ---
st.set_page_config(
    page_title="🍎 Apple Disease Classifier",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        "Get Help": "https://www.extremely.cool-app.com/help",
        "Report a bug": "https://www.extremely.cool-app.com/bug",
        "About": "# This is a classifier for apple diseases.",
    },
)

st.title("🍎 Apple Disease Classifier")
st.markdown(
    "Upload an image of an apple to classify its health status (Formalin-mixed"
    " or Fresh)."
)
st.markdown("---")

# Section 1: Upload Your Own Image
st.subheader("1. Upload Your Own Image")
st.markdown("Upload an image from your computer or use a camera if available.")

uploaded_file = st.file_uploader(
    "Choose an image file (JPG, JPEG, PNG, WEBP, GIF, BMP, TIFF)",
    type=["jpg", "jpeg", "png", "webp", "gif", "bmp", "tiff"],
    help="Drag and drop your image here or click to browse files.",
)

if uploaded_file is not None:
  try:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.success("Image uploaded successfully!")

    if st.button("Classify Uploaded Image"):
      if model:
        _, _, _, softmax_scores = predict_image(
            image, model, app_class_names, image_size=(128, 128)
        )
        st.subheader("Prediction for Uploaded Image:")

        cols = st.columns(len(app_class_names))
        max_score_index = np.argmax(softmax_scores)

        for i, score in enumerate(softmax_scores):
          with cols[i]:
            st.write(f"**{app_class_names[i]}**")
            display_value = (
                f"{score * 100:.2f}% {'✅' if i == max_score_index else ''}"
            )
            st.progress(float(score))
            st.write(display_value)
      else:
        st.error(
            "Model not loaded. Please ensure the model file exists in the"
            " 'models/' directory."
        )
  except Exception as e:
    st.error(f"Error loading image: {e}")
else:
  st.info(
      "Awaiting image upload. Please upload a file to proceed with"
      " classification."
  )

st.markdown("---")

# Section 2: Model Performance Summary
st.subheader("2. Model Performance Summary")
st.markdown(
    "Here's a brief overview of the model's performance on a separate test"
    " set:"
)

st.metric(label="Model Test Accuracy", value=f"{MODEL_TEST_ACCURACY:.2f}%")

if MODEL_TEST_ACCURACY > 80:
  st.success("This model shows good overall accuracy. Great job!")
elif MODEL_TEST_ACCURACY > 60:
  st.info(
      "The model has decent accuracy, but there might be room for improvement."
  )
else:
  st.warning(
      "The model's accuracy is relatively low. Consider further training or"
      " model adjustments."
  )

st.markdown(
    "*(Note: This accuracy score is a static value provided for context.)*"
)
