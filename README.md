# Fish Species Classification from Images

## Project Overview

Identifying fish species accurately from images is important for fisheries monitoring, environmental research, and culinary purposes. This project builds a convolutional neural network (CNN) to classify fish species from images into nine categories: Black Sea Sprat, Gilt-Head Bream, Hourse Mackerel, Red Mullet, Red Sea Bream, Sea Bass, Shrimp, Striped Red Mullet, and Trout.

The goal is to provide a lightweight but effective image classifier that can be used in:

- Fisheries population monitoring
- Research and biodiversity studies
- Cooking and seafood supply chain applications
- Educational tools for species identification

The model leverages:

- Image preprocessing with PyTorch-style normalization
- Data augmentation to improve generalization
- Multiple CNN architectures with varying depth, dropout, and batch normalization

---

## Dataset

### Source
https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset

The dataset consists of over 9,000 images of fish across 9 species. Images vary in resolution and background, providing a realistic challenge for species classification.

### Preprocessing & Augmentation

- Resized all images to 224×224 pixels.
- Applied augmentations for training images only:
  - Random horizontal flips
  - Random rotations
  - Color jitter (brightness, contrast, saturation)
- Validation and test images kept original transforms (resize + normalization) for unbiased evaluation.

### Train/Validation/Test Split

- Training: 60%
- Validation: 20%
- Test: 20%

---

## Exploratory Data Analysis (EDA)

#### Key findings from EDA

- Most classes are relatively balanced, with minor variations in sample counts.
- Images contain varied backgrounds and lighting, emphasizing the need for augmentation.
- Shrimp and Trout images are more visually distinct, while some species (e.g., Gilt-Head Bream vs. Red Sea Bream) are harder to differentiate.
- Color and shape features dominate class distinctions.

EDA visualizations include:

- Class distribution histograms
- Sample images per class
- Heatmaps of image pixel intensity ranges

---

## Modeling Approach & Metrics

### Problem Statement

This is a **multi-class image classification** problem:

- Input: Fish image
- Output: One of nine fish species

### Primary Metric

- **Accuracy** — straightforward metric for multi-class classification.
- **Softmax probabilities** used to rank predictions.

### Model Training

Multiple CNN architectures were tested:

- Linear classifier (no hidden layers)
- CNN with 1–3 inner layers
- Variants with and without dropout
- Variants with batch normalization

Hyperparameters tuned:

- Learning rate
- Dropout rate (0–0.4)
- Inner layer size (128–256)
- Number of hidden layers (1–3)

Training used early stopping based on validation accuracy.

### Results

- Best model: 1 inner layer, 128 neurons, 0.0 dropout, with batch normalization
- Test accuracy: ~92%
- Model generalizes well to standard test images but may struggle on unseen web images with unusual backgrounds.

---

## How to Run

### Local

#### Dependencies

For testing the app go into `/services` and use `uv`:

```
uv sync
```

For installing the dependencies required for the notebook/training specifically please run:
```
pip install -r requirements.txt
```

### Training

A pre-trained model is provided (`model/final_model.onnx`) but steps on how to train the model are in the notebook.

### Deployment

Run the FastAPI service locally:

```
uv run uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

You can send a request throguh the `test.py` function in the service folder. Replace the url with a different image url to play around with it.

```
uv run python test.py
```

## Docker

### Build Container
```
docker build -t fish-classifier:v1 .
```

### Run Container
```
docker run -it --rm -p 8080:8080 fish-classifier:v1     
```

Use the same `test.py` file to use the service.

## Known Limitations / Next Steps

### Limitations
- Model may misclassify visually similar species.
- Generalization to images outside the dataset may be poor (different backgrounds, lighting, angles).
- No uncertainty calibration for predictions.

### Potential Improvements
- Increase dataset size with more diverse images.
- Apply more sophisticated augmentation or style transfer.
- Train deeper CNN architectures (ResNet, EfficientNet) for better feature extraction.
- Add a confidence threshold to reject low-confidence predictions.
- Build a real-time web interface or mobile app for field use.