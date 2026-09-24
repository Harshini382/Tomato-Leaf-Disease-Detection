"""
Debug script to test the prediction pipeline
"""
import os
import numpy as np
from PIL import Image
import pickle
from methods.image_pipeline import extract_feature_vector
from utils import load_and_preprocess_image
import io

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

def test_prediction(image_path):
    print(f"\n{'='*50}")
    print(f"Testing: {os.path.basename(image_path)}")
    print('='*50)
    
    # Load image
    with open(image_path, 'rb') as f:
        file_bytes = f.read()
    
    # Method 1: Use utils preprocessing (like the web app)
    processed, grayscale_rgb, clahe_rgb = load_and_preprocess_image(io.BytesIO(file_bytes))
    
    # Extract features using the pipeline
    feat_vec, feat_names, meta = extract_feature_vector(clahe_rgb)
    
    print(f"Feature vector type: {type(feat_vec)}")
    if isinstance(feat_vec, dict):
        print(f"Features: {list(feat_vec.keys())}")
        feat_array = np.array(list(feat_vec.values())).reshape(1, -1)
    else:
        print(f"Features shape: {feat_vec.shape}")
        feat_array = feat_vec.reshape(1, -1)
    
    print(f"Feature array shape: {feat_array.shape}")
    
    # Load GLCM model
    with open('tomato_disease_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    print(f"Model feature type: {model_data.get('feature_type', 'unknown')}")
    print(f"Model classes: {model_data.get('classes', 'N/A')}")
    
    clf = model_data['classifier']
    
    # Predict
    proba = clf.predict_proba(feat_array)[0]
    pred_idx = np.argmax(proba)
    
    print(f"\nPrediction: {DISEASE_CLASSES[pred_idx]}")
    print(f"Confidence: {proba[pred_idx]*100:.1f}%")
    print(f"\nAll probabilities:")
    for i, p in enumerate(proba):
        print(f"  {DISEASE_CLASSES[i]}: {p*100:.1f}%")

# Test with different disease images
print("Testing GLCM Model Predictions")
print("="*50)

# Test with one image from each class
train_dir = 'tomato/train'
for class_name in DISEASE_CLASSES[:5]:  # Test first 5 classes
    class_path = os.path.join(train_dir, class_name)
    if not os.path.exists(class_path):
        continue
    
    files = [f for f in os.listdir(class_path) if f.endswith('.JPG')]
    if files:
        test_prediction(os.path.join(class_path, files[0]))
