# Crack-Detector
A web app that comes with a model built to detect concrete cracks in bridge images
 Concrete Crack Detection Dataset Preparation

This repository contains the Jupyter Notebook used for preparing and preprocessing a dataset for concrete crack detection. The notebook handles the organization, splitting, and verification of image data into training, validation, and testing sets.

## Project Overview

This project aims to develop a machine learning model for identifying concrete cracks. This specific notebook focuses on the crucial initial step: preparing the image dataset. It takes an raw dataset, renames output directories for clarity, and then systematically splits the images into subsets suitable for model training, validation, and testing.

## Notebook Contents

1.  **Environment Setup**: Imports necessary libraries (e.g., TensorFlow, Keras, Matplotlib, Seaborn, scikit-learn) and sets random seeds for reproducibility. It also verifies GPU availability.
2.  **Path Configuration**: Defines the input path for the raw dataset and the output path for the processed and split dataset. The output directory is named `concrete crack renamed`.
3.  **Dataset Splitting**: Implements a robust method to split the dataset into `train`, `val` (validation), and `test` directories with configurable ratios (default: 70% train, 15% validation, 15% test). This process iterates through categories and classes within the dataset, ensuring a consistent split.
4.  **Split Verification**: After splitting, the notebook verifies the existence of the newly created directories and counts the number of images in each split, confirming the successful distribution of the dataset.

## Setup and Usage

To run this notebook and prepare your own dataset:

1.  **Mount Google Drive**: Ensure your Google Drive is mounted in Google Colab.
2.  **Dataset Placement**: Place your raw dataset (e.g., `extracted_dataset`) in the specified `base_input` path within your Google Drive.
3.  **Run Cells**: Execute the cells in sequential order.
    *   The `base_input` variable should point to the directory containing your raw, unzipped dataset (e.g., `/content/drive/MyDrive/extracted_dataset/`).
    *   The `base_output` variable defines where the processed dataset will be saved (e.g., `/content/drive/MyDrive/concrete crack renamed/`).
4.  **Review Output**: Check the printed outputs for path configurations and image counts in each split to confirm successful data preparation.

## Directory Structure (Output)

After running the notebook, the `concrete crack renamed` directory in your Google Drive will have a structure similar to this:

```
concrete crack renamed/
├── train/
│   ├── Decks_Cracked/
│   ├── Decks_Non-cracked/
│   ├── Pavements_Cracked/
│   └── Pavements_Non-cracked/
├── val/
│   ├── Decks_Cracked/
│   ├── Decks_Non-cracked/
│   ├── Pavements_Cracked/
│   └── Pavements_Non-cracked/
└── test/
    ├── Decks_Cracked/
    ├── Decks_Non-cracked/
    ├── Pavements_Cracked/
    └── Pavements_Non-cracked/
```

Each subdirectory within `train`, `val`, and `test` will contain the respective image files for that class and split.

```
## Project Overview

This project aims to build a machine learning model for detecting concrete cracks in various surfaces (walls, pavements, decks). It leverages TensorFlow/Keras for image classification.

## Getting Started

To replicate the environment and data setup:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/umanahekeobong-cell/Crack-Detector.git
    cd Crack-Detector
    ```

2.  **Dataset Preparation:**
    The dataset (e.g., `archive (1).zip`) should be unzipped into a local directory named `./data`. The notebook will then reorganize this raw data into `train`, `val`, and `test` splits, creating a structured dataset in Google Drive (e.g., `/content/drive/MyDrive/concrete crack renamed/`). This process categorizes images into classes like `Walls_Cracked`, `Pavements_Non-cracked`, etc.

## Data Loading and Preprocessing

*   **Dataset Loading:** Images are loaded efficiently using `tf.keras.utils.image_dataset_from_directory` with specified image dimensions (128x128 pixels) and batch sizes.
