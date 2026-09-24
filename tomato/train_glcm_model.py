"""
Train model using GLCM features - the proper way for this system
"""
import os
import numpy as np
from PIL import Image
import cv2
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from methods.image_pipeline import extract_feature_vector

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

def load_and_extract_features(data_dir, samples_per_class=300):
    """Load images and extract GLCM features"""
    features = []
    labels = []
    
    for idx, class_name in enumerate(DISEASE_CLASSES):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.exists(class_path):
            print(f"Missing: {class_path}")
            continue
            
        print(f"Processing {class_name}...")
        files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        files = files[:samples_per_class]
        
        for i, f in enumerate(files):
            try:
                img_path = os.path.join(class_path, f)
                img = Image.open(img_path).convert('RGB')
                img = img.resize((224, 224))
                img_array = np.array(img)
                
                # Extract GLCM features using the pipeline
                feat_vec, _, _ = extract_feature_vector(img_array)
                
                # Flatten the feature vector
                if isinstance(feat_vec, dict):
                    # Convert dict to flat array
                    feat_vec = np.array(list(feat_vec.values()))
                elif len(feat_vec.shape) > 1:
                    feat_vec = feat_vec.flatten()
                    
                features.append(feat_vec)
                labels.append(idx)
                
            except Exception as e:
                print(f"Error: {f} - {e}")
                continue
        
        print(f"  -> {len([l for l in labels if l == idx])} samples")
    
    return np.array(features), np.array(labels)

def main():
    print("="*50)
    print("Training with GLCM Features")
    print("="*50)
    
    # Load training data
    X, y = load_and_extract_features('tomato/train', samples_per_class=300)
    print(f"\nTotal: {len(X)} samples, {len(np.unique(y))} classes")
    print(f"Feature shape: {X.shape}")
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Train
    print("\nTraining Random Forest...")
    clf = RandomForestClassifier(n_estimators=300, max_depth=50, random_state=42, n_jobs=-1, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    # Evaluate
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    val_acc = accuracy_score(y_val, clf.predict(X_val))
    print(f"Train: {train_acc*100:.1f}%, Val: {val_acc*100:.1f}%")
    
    # Test individual predictions
    print("\nTesting predictions:")
    test_classes = clf.classes_
    test_preds = clf.predict(X_val[:10])
    for i, pred in enumerate(test_preds):
        print(f"  Sample {i}: Predicted={DISEASE_CLASSES[pred]}, Actual={DISEASE_CLASSES[y_val[i]]}")
    
    # Save model with GLCM features
    model_data = {
        'classifier': clf,
        'classes': DISEASE_CLASSES,
        'feature_type': 'glcm'  # Mark as GLCM model
    }
    
    with open('tomato_disease_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("\nModel saved to tomato_disease_model.pkl")
    print(f"Classes: {clf.classes_}")

if __name__ == '__main__':
    main()
