import cv2
import numpy as np
from PIL import Image

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess the image for disease detection:
    1. Convert to grayscale
    2. Apply CLAHE for contrast enhancement
    3. Resize to target size
    4. Normalize pixel values

    Args:
        image_path (str): Path to the input image
        target_size (tuple): Target size for resizing (width, height)

    Returns:
        tuple: (processed_image, grayscale_image, clahe_image)
               processed_image: Final processed image for model input
               grayscale_image: Grayscale version for display
               clahe_image: CLAHE enhanced version for display
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray)

    # Resize to target size
    resized = cv2.resize(clahe_image, target_size)

    # Normalize pixel values to [0, 1]
    normalized = resized.astype(np.float32) / 255.0

    # Convert grayscale and CLAHE images to RGB for display
    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    clahe_rgb = cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2RGB)

    return normalized, gray_rgb, clahe_rgb

def load_and_preprocess_image(image_file, target_size=(224, 224)):
    """
    Load image from file object and preprocess it.

    Args:
        image_file: File object from Flask request
        target_size (tuple): Target size for resizing

    Returns:
        tuple: (processed_image, grayscale_image, clahe_image)
    """
    # Convert file to numpy array
    image = Image.open(image_file)
    image = np.array(image)

    # Convert RGB to BGR if necessary (PIL loads as RGB)
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray)

    # Resize to target size
    resized = cv2.resize(clahe_image, target_size)

    # Normalize pixel values to [0, 1] and convert to RGB for model input
    normalized = resized.astype(np.float32) / 255.0

    # Convert to 3-channel RGB for MobileNetV2
    processed_rgb = cv2.cvtColor((normalized * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0

    # Convert grayscale and CLAHE images to RGB for display
    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    clahe_rgb = cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2RGB)

    return processed_rgb, gray_rgb, clahe_rgb
