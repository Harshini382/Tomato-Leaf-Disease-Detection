"""
Method 1 module - Uses improved feature extraction
"""
import io
import numpy as np
from PIL import Image
import cv2
import pickle
import os
from utils import load_and_preprocess_image

# Disease classes
DISEASE_CLASSES = [
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___healthy',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
]

# Global model
_model_data = None

def load_model():
    """Load the improved model"""
    global _model_data
    if _model_data is not None:
        return _model_data
    
    model_path = 'tomato_disease_model.pkl'
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                _model_data = pickle.load(f)
            print(f"Model loaded: {_model_data.get('feature_type', 'unknown')} type")
            return _model_data
        except Exception as e:
            print(f"Error loading model: {e}")
    return None

def extract_features(img_array):
    """Extract comprehensive features"""
    features = []
    
    # Convert to different color spaces
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    
    # 1. Basic grayscale stats
    features.extend([np.mean(gray), np.std(gray), np.min(gray), np.max(gray),
                    np.median(gray), np.percentile(gray, 25), np.percentile(gray, 75)])
    
    # 2. RGB channel stats
    for i in range(3):
        channel = img_array[:,:,i]
        features.extend([np.mean(channel), np.std(channel)])
    
    # 3. HSV stats
    for i in range(3):
        channel = hsv[:,:,i]
        features.extend([np.mean(channel), np.std(channel)])
    
    # 4. LAB stats
    for i in range(3):
        channel = lab[:,:,i]
        features.extend([np.mean(channel), np.std(channel)])
    
    # 5. CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(gray)
    features.extend([np.mean(clahe_img), np.std(clahe_img)])
    
    # 6. Edge features
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    features.extend([np.mean(sobel_mag), np.std(sobel_mag), np.max(sobel_mag)])
    
    # 7. Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    features.extend([np.mean(np.abs(laplacian)), np.std(laplacian)])
    
    # 8. Local variance
    kernel = np.ones((5,5), np.float32) / 25
    local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    local_var = cv2.filter2D((gray.astype(np.float32) - local_mean)**2, -1, kernel)
    features.extend([np.mean(local_var), np.std(local_var)])
    
    # 9. Histogram
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-8)
    sorted_idx = np.argsort(hist)[::-1][:8]
    for i in sorted_idx:
        features.append(hist[i])
    
    return np.array(features)

def predict_disease_internal(features):
    """Predict using the model"""
    model_data = load_model()
    
    if model_data is None:
        return {'predicted_class': 'Model not loaded', 'confidence': 0.0, 'probabilities': {}}
    
    clf = model_data['classifier']
    scaler = model_data.get('scaler')
    
    # Scale features if scaler exists
    if scaler is not None:
        features = scaler.transform(features.reshape(1, -1))
    else:
        features = features.reshape(1, -1)
    
    # Get prediction
    proba = clf.predict_proba(features)[0]
    pred_idx = np.argmax(proba)
    
    return {
        'predicted_class': DISEASE_CLASSES[pred_idx],
        'confidence': float(proba[pred_idx]),
        'probabilities': {DISEASE_CLASSES[i]: float(p) for i, p in enumerate(proba)}
    }


def run_method1(file_bytes):
    """Run Method 1 pipeline"""
    # Load original image
    original_pil = Image.open(io.BytesIO(file_bytes))
    original_arr = np.array(original_pil)
    
    # Handle alpha channel
    if len(original_arr.shape) == 3 and original_arr.shape[2] == 4:
        original_arr = cv2.cvtColor(original_arr, cv2.COLOR_RGBA2RGB)
    
    # Get preprocessing
    processed, grayscale_rgb, clahe_rgb = load_and_preprocess_image(io.BytesIO(file_bytes))
    
    # Resize to 128x128 for feature extraction
    img_resized = cv2.resize(original_arr, (128, 128))
    
    # Extract features
    features = extract_features(img_resized)
    
    # Predict
    results = predict_disease_internal(features)
    
    return {
        'results': results,
        'original': original_arr,
        'grayscale': grayscale_rgb,
        'processed': clahe_rgb,
        'processed_for_model': clahe_rgb,
        'feature_vector': features,
        'feature_names': [],
        'feature_meta': {}
    }


def get_report_context():
    return {
        'executive_model_architecture': 'Method 1 - Improved Features + Random Forest',
        'executive_feature_extractor': 'Color + Texture + Edge features',
        'executive_classifier': 'Random Forest (500 trees)',
        'executive_target_accuracy': 'High accuracy across all disease classes'
    }
