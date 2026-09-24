import os
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
import joblib
import pickle

# Disease classes (10 diseases + healthy)
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

class TomatoDiseaseDetector:
    def __init__(self):
        self.feature_extractor = None
        self.classifier = None
        self.model_path = 'tomato_disease_model.pkl'
        self.classifier_path = 'random_forest_classifier.pkl'

    def build_feature_extractor(self):
        """Build MobileNetV2 feature extractor with frozen weights."""
        # Load MobileNetV2 without top layers
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        # Freeze the base model
        base_model.trainable = False

        # Add global average pooling
        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)

        # Create feature extractor model
        self.feature_extractor = Model(inputs=base_model.input, outputs=x)

    def extract_features(self, images):
        """Extract features from images using MobileNetV2."""
        if self.feature_extractor is None:
            self.build_feature_extractor()

        # Ensure images are in the right format
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=0)

        # Convert BGR to RGB for MobileNetV2
        images = images[..., ::-1]

        # Extract features
        features = self.feature_extractor.predict(images, verbose=0)
        return features

    def load_data(self, data_dir='tomato/train'):
        """Load and preprocess training data."""
        features = []
        labels = []

        for idx, class_name in enumerate(DISEASE_CLASSES):
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} not found")
                continue

            print(f"Loading {class_name}...")

            for img_file in os.listdir(class_dir)[:500]:  # Limit samples for faster training
                if img_file.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_dir, img_file)

                    try:
                        # Load and preprocess image
                        image = cv2.imread(img_path)
                        if image is None:
                            continue

                        image = cv2.resize(image, (224, 224))
                        image = image.astype(np.float32) / 255.0

                        # Extract features
                        feature = self.extract_features(image)
                        features.append(feature.flatten())
                        labels.append(idx)

                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
                        continue

        return np.array(features), np.array(labels)

    def train_classifier(self, features, labels):
        """Train Random Forest classifier."""
        print("Training Random Forest classifier...")

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # Train Random Forest
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )

        self.classifier.fit(X_train, y_train)

        # Evaluate
        train_pred = self.classifier.predict(X_train)
        val_pred = self.classifier.predict(X_val)

        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)

        print(".2f")
        print(".2f")

        return val_acc >= 0.93  # Check if accuracy meets requirement

    def train_model(self):
        """Train the complete model."""
        print("Building feature extractor...")
        self.build_feature_extractor()

        print("Loading training data...")
        features, labels = self.load_data()

        if len(features) == 0:
            raise ValueError("No training data found")

        print(f"Loaded {len(features)} samples with {len(DISEASE_CLASSES)} classes")

        # Train classifier
        success = self.train_classifier(features, labels)

        if not success:
            print("Warning: Model accuracy below 93%. Consider more training data or hyperparameter tuning.")

        # Save model
        self.save_model()

        return success

    def save_model(self):
        """Save the trained model."""
        model_data = {
            'feature_extractor_weights': self.feature_extractor.get_weights(),
            'classifier': self.classifier,
            'classes': DISEASE_CLASSES
        }

        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {self.model_path}")

    def load_model(self):
        """Load the trained model."""
        if not os.path.exists(self.model_path):
            print("Model not found. Training new model...")
            self.train_model()
            return

        print("Loading saved model...")
        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)

        # Check if it's a GLCM or improved model (no feature extractor)
        if 'feature_type' in model_data:
            print(f"{model_data['feature_type']} model loaded - method1 will handle predictions")
            self.classifier = model_data['classifier']
            self.feature_extractor = None
        else:
            # Old CNN-based model
            # Rebuild feature extractor
            self.build_feature_extractor()
            self.feature_extractor.set_weights(model_data['feature_extractor_weights'])

            # Load classifier
            self.classifier = model_data['classifier']

        print("Model loaded successfully")

    def predict(self, image):
        """
        Predict disease from image.

        Args:
            image: Preprocessed image array (224x224x3)

        Returns:
            dict: Prediction results with probabilities
        """
        if self.classifier is None:
            self.load_model()

        # Extract features
        features = self.extract_features(image)

        # Get prediction probabilities
        probabilities = self.classifier.predict_proba(features)[0]

        # Get top prediction
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = DISEASE_CLASSES[predicted_class_idx]
        confidence = probabilities[predicted_class_idx]

        # Prepare results
        results = {
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'probabilities': {class_name: float(prob) for class_name, prob in zip(DISEASE_CLASSES, probabilities)}
        }

        return results

# Global model instance
detector = TomatoDiseaseDetector()

def load_model():
    """Load the disease detection model."""
    detector.load_model()

def predict_disease(image):
    """
    Predict tomato leaf disease from image.

    Args:
        image: Preprocessed image array

    Returns:
        dict: Prediction results
    """
    return detector.predict(image)
