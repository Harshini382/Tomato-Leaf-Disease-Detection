"""
Test script to verify the tomato disease prediction model
"""
import os
import sys
import numpy as np
from PIL import Image
import pickle

# Add current directory to path
sys.path.insert(0, os.getcwd())

def test_model():
    print("=" * 50)
    print("Testing Tomato Disease Model")
    print("=" * 50)
    
    # Test 1: Check if model file exists
    model_path = 'tomato_disease_model.pkl'
    if not os.path.exists(model_path):
        print(f"ERROR: Model file {model_path} not found!")
        return False
    
    print(f"[OK] Model file exists: {model_path}")
    
    # Test 2: Load and inspect model
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        print(f"[OK] Model loaded successfully")
        print(f"  Keys in model_data: {model_data.keys()}")
        
        # Check what's in the model
        if 'classifier' in model_data:
            classifier = model_data['classifier']
            print(f"  Classifier type: {type(classifier)}")
            
            # Check if classifier has classes
            if hasattr(classifier, 'classes_'):
                classes = classifier.classes_
                print(f"  Number of classes: {len(classes)}")
                print(f"  Classes: {classes}")
            else:
                print("  WARNING: Classifier doesn't have classes_ attribute!")
        
        if 'classes' in model_data:
            print(f"  Model classes: {model_data['classes']}")
            
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return False
    
    # Test 3: Test with a sample image from training data
    print("\n" + "=" * 50)
    print("Testing with sample images")
    print("=" * 50)
    
    train_dir = 'tomato/train'
    if not os.path.exists(train_dir):
        print(f"ERROR: Training directory {train_dir} not found!")
        return False
    
    # Get first class directory
    class_dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    print(f"Found {len(class_dirs)} classes in training data")
    
    # Test with one image from each class
    for class_dir in class_dirs[:3]:  # Test first 3 classes
        class_path = os.path.join(train_dir, class_dir)
        images = [f for f in os.listdir(class_path) if f.endswith('.JPG')]
        
        if not images:
            continue
            
        img_path = os.path.join(class_path, images[0])
        print(f"\nTesting with: {os.path.basename(img_path)}")
        print(f"  Actual class: {class_dir}")
        
        # Load and preprocess image
        img = Image.open(img_path)
        img = img.resize((224, 224))
        img_array = np.array(img)
        
        # Convert to RGB if needed
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array]*3, axis=-1)
        elif img_array.shape[2] == 4:
            img_array = img_array[:,:,:3]
        
        print(f"  Image shape: {img_array.shape}")
        
        # Try prediction using models.py
        try:
            from models import predict_disease
            # Ensure image is float32 and normalized
            img_input = img_array.astype(np.float32) / 255.0
            result = predict_disease(img_input)
            
            print(f"  Predicted: {result.get('predicted_class')}")
            print(f"  Confidence: {result.get('confidence', 0)*100:.1f}%")
            print(f"  Probabilities: {result.get('probabilities', {})}")
            
        except Exception as e:
            print(f"  ERROR during prediction: {e}")
            import traceback
            traceback.print_exc()
    
    return True

if __name__ == '__main__':
    test_model()
