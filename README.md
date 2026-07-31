# 🍎 Fresh Apple vs. Formalin-Mixed Apple Classifier

An end-to-end Computer Vision and Deep Learning web application designed to detect formalin contamination in apples using fine-tuned MobileNetV3 architectures and Streamlit.

---

## 📌 Project Overview
Formalin (a dissolved form of formaldehyde) is illegally used in some agricultural supply chains as an artificial preservative to extend the shelf life of fruits. This project provides an accessible, AI-powered diagnostic tool capable of classifying apple images into two categories:
* **`Apple_Fresh`**: Natural, untreated fresh apples.
* **`Apple_Formalin-mixed`**: Apples treated or preserved with formalin.

The core underlying engine utilizes a fine-tuned **MobileNetV3** model trained via Transfer Learning to achieve high accuracy while remaining lightweight enough for real-time edge/web deployment.

---

## 🚀 Key Features
* **Real-time Image Classification:** Upload apple images (`.jpg`, `.png`, `.jpeg`, `.webp`, etc.) for instant predictions.
* **Confidence Scoring:** Visual progress indicators displaying confidence percentages for both classes.
* **Automated Patching:** Dynamic on-the-fly model configuration handling to resolve cross-version Keras serialization issues during deployment.
* **Streamlit Web UI:** Intuitive, browser-accessible user interface for seamless interaction.

---

## 🛠️ Project Structure

```text
├── models/
│   └── mobilenetv3_transfer.keras   # Trained MobileNetV3 model binary
├── app.py                            # Streamlit web application script
├── requirements.txt                  # Environment dependencies
└── README.md                         # Project documentation
