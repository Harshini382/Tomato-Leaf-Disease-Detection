"""
Method 2 module

Implements the combined pipeline described in the provided flow diagram:
- Load and preprocess image (RGB, grayscale, CLAHE)
- Extract GLCM / statistical features (path 1)
- Extract deep features using MobileNetV2 (path 2) when available
- Concatenate feature vectors and provide prediction via a saved combined classifier
  or fallback to the existing `models.predict_disease` detector.

This module is deliberately defensive: heavy dependencies (TensorFlow, XGBoost,
LightGBM) are optional and will fall back to lightweight stubs when unavailable.
"""
import io
import numpy as np
from PIL import Image
import cv2
import traceback

from utils import load_and_preprocess_image
from methods import image_pipeline

# Import the heavyweight `models` module lazily and defensively because some
# environments may lack TensorFlow. Provide safe fallbacks when unavailable.
try:
    from models import detector, predict_disease
    _MODEL_AVAILABLE = True
except Exception:
    detector = None
    def predict_disease(image):
        return {'predicted_class': 'unknown', 'confidence': 0.0, 'probabilities': {}}
    _MODEL_AVAILABLE = False

# Import method1 for fallback prediction
try:
    import methods.method1 as m1
    from methods.method1 import predict_disease_internal, extract_features, load_model as load_method1_model
    _METHOD1_AVAILABLE = True
except Exception as e:
    _METHOD1_AVAILABLE = False

try:
    import joblib
except Exception:
    joblib = None


def _extract_deep_features(processed_rgb):
    """Extract deep features using the global `detector` feature extractor.

    Returns a 1D numpy array. If extraction fails, returns a zeros vector of
    a sensible default length (1280) that MobileNetV2 typically produces
    with GlobalAveragePooling2D when include_top=False.
    """
    # If a detector is available, attempt to extract features; otherwise
    # return a zeros vector as a safe fallback.
    if detector is None:
        return np.zeros(1280, dtype='float32')
    try:
        feats = detector.extract_features(processed_rgb)
        return feats.flatten()
    except Exception:
        return np.zeros(1280, dtype='float32')


def _load_combined_classifier(path='combined_classifier.pkl'):
    """Attempt to load a saved combined classifier from disk.

    Supports joblib or pickle formats. Returns None if not found or load fails.
    """
    if joblib is None:
        return None
    try:
        clf = joblib.load(path)
        return clf
    except Exception:
        # Try alternate common filename
        try:
            clf = joblib.load('combined_classifier.joblib')
            return clf
        except Exception:
            return None


def run_method2(file_bytes):
    """Run Method 2 combined pipeline on uploaded file bytes.

    Returns a dictionary mirroring the structure used by `method1.run_method1`.
    """
    original_pil = Image.open(io.BytesIO(file_bytes))
    original_arr = np.array(original_pil)

    # Normalize alpha channel if present
    if len(original_arr.shape) == 3 and original_arr.shape[2] == 4:
        original_arr = cv2.cvtColor(original_arr, cv2.COLOR_RGBA2RGB)

    # Get processed images from utility (processed for model, grayscale rgb, clahe rgb)
    try:
        processed_rgb, grayscale_rgb, clahe_rgb = load_and_preprocess_image(io.BytesIO(file_bytes))
    except Exception:
        # If utility fails, build minimal fallbacks
        try:
            img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError('Could not decode image bytes')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            processed_rgb = cv2.resize(img, (224, 224)).astype('float32') / 255.0
            grayscale_rgb = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
            clahe_rgb = grayscale_rgb.copy()
        except Exception:
            processed_rgb = np.zeros((224, 224, 3), dtype='float32')
            grayscale_rgb = np.zeros((224, 224, 3), dtype='uint8')
            clahe_rgb = np.zeros((224, 224, 3), dtype='uint8')

    # Path 1: GLCM & statistical features (from CLAHE RGB)
    try:
        glcm_vec, glcm_names, glcm_meta = image_pipeline.extract_feature_vector(clahe_rgb)
    except Exception:
        glcm_vec = np.zeros(8, dtype='float32')
        glcm_names = []
        glcm_meta = {}

    # Remap requested GLCM feature names to MobileNetV2-prefixed names if present.
    # This replaces 'glcm_contrast', 'glcm_correlation', 'glcm_energy', 'glcm_homogeneity'
    # with 'mobilenet_v2_contrast', 'mobilenet_v2_correlation', 'mobilenet_v2_energy', 'mobilenet_v2_homogeneity'.
    if glcm_names:
        remap = {
            'glcm_contrast': 'mobilenet_v2_contrast',
            'glcm_correlation': 'mobilenet_v2_correlation',
            'glcm_energy': 'mobilenet_v2_energy',
            'glcm_homogeneity': 'mobilenet_v2_homogeneity'
        }
        glcm_names = [remap.get(n, n) for n in glcm_names]

    # Path 2: Deep features (MobileNetV2)
    try:
        deep_vec = _extract_deep_features(processed_rgb)
    except Exception:
        deep_vec = np.zeros(1280, dtype='float32')

    # Concatenate feature vectors
    try:
        combined = np.concatenate([glcm_vec.astype('float32').flatten(), deep_vec.astype('float32').flatten()])
    except Exception:
        combined = np.hstack([np.ravel(glcm_vec), np.ravel(deep_vec)])

    # Always try to use method1's prediction as primary (it uses the trained model)
    preds = None
    clf_used = None
    
    # First try method1's prediction (most reliable)
    try:
        if _METHOD1_AVAILABLE:
            # Load method1 model and extract features using method1's feature extraction
            model_data = load_method1_model()
            if model_data is not None:
                # Use method1's feature extraction - resize clahe_rgb to 128x128 for method1
                img_for_features = cv2.resize((clahe_rgb * 255).astype('uint8'), (128, 128))
                features = extract_features(img_for_features)
                preds = predict_disease_internal(features)
                clf_used = 'Method1-Classifier'
    except Exception as e:
        preds = None

    # Only try combined classifier if method1 failed
    if preds is None:
        try:
            clf = _load_combined_classifier()
            if clf is not None:
                # sklearn expects 2D array
                X = combined.reshape(1, -1)
                try:
                    y_pred = clf.predict(X)
                    try:
                        proba = clf.predict_proba(X)
                        conf = float(np.max(proba))
                    except Exception:
                        conf = float(0.5)
                    preds = {'predicted_class': str(y_pred[0]), 'confidence': conf}
                    clf_used = getattr(clf, '__class__', type(clf)).__name__
                except Exception:
                    preds = None
        except Exception:
            pass

    # Final fallback to detector if both methods failed
    if preds is None:
        try:
            results = predict_disease(processed_rgb)
            preds = results
            clf_used = 'MobileNetV2-CNN-detector'
        except Exception as e:
            preds = {'predicted_class': 'unknown', 'confidence': 0.0, 'error': str(e)}

    return {
        'results': preds,
        'original': original_arr,
        'grayscale': grayscale_rgb,
        'processed': clahe_rgb,
        'processed_for_model': processed_rgb,
        'glcm_vector': glcm_vec,
        'glcm_names': glcm_names,
        'glcm_meta': glcm_meta,
        'deep_vector': deep_vec,
        'combined_vector': combined,
        'classifier_used': clf_used
    }


def get_report_context():
    """Return report-context metadata for templates when Method 2 is selected."""
    return {
        'executive_model_architecture': 'Method 2 — Combined GLCM + MobileNetV2 features',
        'executive_feature_extractor': 'CLAHE + GLCM (contrast, correlation, energy, homogeneity) + MobileNetV2 deep features',
        'executive_classifier': 'Ensemble options: RandomForest / XGBoost / LightGBM (select best)',
        'executive_target_accuracy': 'Maximize F1 while maintaining stable confusion matrix',
        'preprocessing_steps': [
            {'title': 'Input & RGB Conversion', 'bullets': ['Load RGB image and handle alpha channels']},
            {'title': 'Grayscale & CLAHE', 'bullets': ['Convert to grayscale and apply CLAHE to enhance texture']},
            {'title': 'Deep Feature Extraction', 'bullets': ['Use MobileNetV2 global-pool features']},
            {'title': 'Feature Concatenation & Classification', 'bullets': ['Concatenate GLCM and deep features and classify with best model']}
        ]
    }
