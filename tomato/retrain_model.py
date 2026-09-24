"""
Improved retrain script with more training data and better parameters
"""
import os
import sys
import numpy as np
from PIL import Image
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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

def build_feature_extractor():
    print("Building MobileNetV2...")
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    return Model(inputs=base_model.input, outputs=x)

def extract_features(extractor, img_array):
    if img_array.max() <= 1.0:
        img_array = img_array * 255.0
    img_array = img_array.astype(np.uint8)
    
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array]*3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[:,:,:3]
    
    img_array = img_array[..., ::-1]
    features = extractor.predict(np.expand_dims(img_array, axis=0), verbose=0)
    return features[0]

def load_data(data_dir, samples=500):
    extractor = build_feature_extractor()
    features, labels = [], []
    
    for idx, class_name in enumerate(DISEASE_CLASSES):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.exists(class_path):
            continue
        
        print(f"Loading {class_name}...")
        files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        files = files[:samples]
        
        for f in files:
            try:
                img = Image.open(os.path.join(class_path, f)).resize((224, 224))
                feat = extract_features(extractor, np.array(img))
                features.append(feat)
                labels.append(idx)
            except:
                continue
        print(f"  -> {len([l for l in labels if l == idx])} samples")
    
    return np.array(features), np.array(labels), extractor

def main():
    print("="*50)
    print("Retraining with MORE data")
    print("="*50)
    
    features, labels, extractor = load_data('tomato/train', samples=500)
    print(f"\nTotal: {len(features)} samples, {len(np.unique(labels))} classes")
    
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.15, stratify=labels, random_state=42)
    
    print("\nTraining Random Forest (200 trees)...")
    clf = RandomForestClassifier(n_estimators=200, max_depth=40, random_state=42, n_jobs=-1, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    val_acc = accuracy_score(y_val, clf.predict(X_val))
    print(f"Train: {train_acc*100:.1f}%, Val: {val_acc*100:.1f}%")
    
    # Save
    model_data = {
        'feature_extractor_weights': extractor.get_weights(),
        'classifier': clf,
        'classes': DISEASE_CLASSES
    }
    with open('tomato_disease_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    print("\nModel saved!")
    
    # Verify
    with open('tomato_disease_model.pkl', 'rb') as f:
        data = pickle.load(f)
    print(f"Classes in saved model: {len(data['classifier'].classes_)}")

if __name__ == '__main__':
    main()
