import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load pre-trained ResNet50 model for feature extraction
resnet_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

def preprocess_image(image_path):
    """
    Preprocesses an MRI/CT image for analysis.
    :param image_path: Path to the medical image file.
    :return: Feature vector extracted from the image.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = cv2.resize(image, (224, 224))  # Resize for ResNet50

    # Convert to 3-channel image
    image = np.stack((image,) * 3, axis=-1)

    # Normalize for ResNet
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    # Extract features
    features = resnet_model.predict(image)
    return features.flatten()
