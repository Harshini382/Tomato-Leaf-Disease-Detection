"""
Improved model with more features and better training
"""
import os
import numpy as np
from PIL import Image
import cv2
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

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

def extract_all_features(img_array):
    """Extract comprehensive features from image"""
    features = []
    feature_names = []
    
    # Convert to different color spaces
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    
    # 1. Basic grayscale stats
    features.extend([np.mean(gray), np.std(gray), np.min(gray), np.max(gray),
                    np.median(gray), np.percentile(gray, 25), np.percentile(gray, 75)])
    feature_names.extend(['gray_mean', 'gray_std', 'gray_min', 'gray_max', 
                        'gray_median', 'gray_q25', 'gray_q75'])
    
    # 2. RGB channel stats
    for i, c in enumerate(['R', 'G', 'B']):
        channel = img_array[:,:,i]
        features.extend([np.mean(channel), np.std(channel)])
        feature_names.extend([f'{c}_mean', f'{c}_std'])
    
    # 3. HSV stats
    for i, c in enumerate(['H', 'S', 'V']):
        channel = hsv[:,:,i]
        features.extend([np.mean(channel), np.std(channel)])
        feature_names.extend([f'{c}_mean', f'{c}_std'])
    
    # 4. LAB stats
    for i, c in enumerate(['L', 'a', 'b']):
        channel = lab[:,:,i]
        features.extend([np.mean(channel), np.std(channel)])
        feature_names.extend([f'{c}_mean', f'{c}_std'])
    
    # 5. CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(gray)
    features.extend([np.mean(clahe_img), np.std(clahe_img)])
    feature_names.extend(['clahe_mean', 'clahe_std'])
    
    # 6. Edge features (Sobel)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    features.extend([np.mean(sobel_mag), np.std(sobel_mag), np.max(sobel_mag)])
    feature_names.extend(['edge_mean', 'edge_std', 'edge_max'])
    
    # 7. Laplacian (edge detection)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    features.extend([np.mean(np.abs(laplacian)), np.std(laplacian)])
    feature_names.extend(['laplacian_mean', 'laplacian_std'])
    
    # 8. Local Binary Pattern approximation (using simple filtering)
    # Simple texture measure using local variance
    kernel = np.ones((5,5), np.float32) / 25
    local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    local_var = cv2.filter2D((gray.astype(np.float32) - local_mean)**2, -1, kernel)
    features.extend([np.mean(local_var), np.std(local_var)])
    feature_names.extend(['local_var_mean', 'local_var_std'])
    
    # 9. Histogram features
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
    hist = hist.flatten() / hist.sum()
    # Take top 8 histogram bins as features
    sorted_idx = np.argsort(hist)[::-1][:8]
    for i in sorted_idx:
        features.append(hist[i])
        feature_names.append(f'hist_bin_{i}')
    
    return np.array(features)

def load_and_extract_features(data_dir, samples_per_class=400):
    """Load images and extract all features"""
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
        
        for f in files:
            try:
                img = Image.open(os.path.join(class_path, f)).convert('RGB')
                img = img.resize((128, 128))  # Smaller for speed
                img_array = np.array(img)
                
                feat = extract_all_features(img_array)
                features.append(feat)
                labels.append(idx)
                
            except Exception as e:
                continue
        
        print(f"  -> {len([l for l in labels if l == idx])} samples")
    
    return np.array(features), np.array(labels)

def main():
    print("="*50)
    print("Training Improved Model")
    print("="*50)
    
    # Load training data
    X, y = load_and_extract_features('tomato/train', samples_per_class=400)
    print(f"\nTotal: {len(X)} samples")
    print(f"Feature shape: {X.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)
    
    # Train with more trees
    print("\nTraining Random Forest (500 trees)...")
    clf = RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_split=2, 
                                random_state=42, n_jobs=-1, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    val_acc = accuracy_score(y_val, clf.predict(X_val))
    print(f"Train: {train_acc*100:.1f}%, Val: {val_acc*100:.1f}%")
    
    # Test predictions
    print("\nTest predictions:")
    test_preds = clf.predict(X_val[:10])
    for i, pred in enumerate(test_preds):
        print(f"  {i}: Predicted={DISEASE_CLASSES[pred]}, Actual={DISEASE_CLASSES[y_val[i]]}")
    
    # Save model
    model_data = {
        'classifier': clf,
        'scaler': scaler,
        'classes': DISEASE_CLASSES,
        'feature_type': 'improved'
    }
    
    with open('tomato_disease_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("\nModel saved!")

if __name__ == '__main__':
    main()
