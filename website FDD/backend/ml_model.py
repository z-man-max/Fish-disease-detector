"""
ml_model.py - Fish Disease Detection using Random Forest
NOW WITH 3 CLASSES: Healthy, Ich (White Spot), Black Spot
"""

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import joblib
import os

class FishDiseaseDetector:
    def __init__(self):
        """
        Initialize the detector.
        We'll load trained model if it exists, otherwise start fresh.
        """
        self.classifier = None  # Will hold our Random Forest model
        self.scaler = None      # Will hold our feature scaler
        self.model_path = 'fish_model_3classes.pkl'  # NEW: Different name for 3-class model
        self.scaler_path = 'scaler_3classes.pkl'      # NEW: Different scaler name
        
        # NEW: Define class names for 3 classes
        self.class_names = ['Healthy', 'Ich (White Spot)', 'Black Spot']
        
    def extract_features(self, image_path):
        """
        Convert image to NUMBERS that ML can understand
        NOW detects BOTH white spots AND black spots!
        """
        print(f"📸 Extracting features from: {os.path.basename(image_path)}")
        
        # Read the image using OpenCV
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        # Resize to standard size
        img = cv2.resize(img, (224, 224))
        
        # === FEATURE 1: COLOR FEATURES ===
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        color_features = [
            np.mean(hsv[:,:,0]),  # Average hue
            np.std(hsv[:,:,0]),    # Hue variation
            np.mean(hsv[:,:,1]),  # Average saturation
            np.std(hsv[:,:,1]),    # Saturation variation
            np.mean(hsv[:,:,2]),  # Average brightness
            np.std(hsv[:,:,2]),    # Brightness variation
        ]
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # === FEATURE 2: WHITE SPOT DETECTION (for Ich) ===
        # Look for bright spots
        _, white_spots = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        white_spot_density = np.sum(white_spots > 0) / (224 * 224)
        
        # White blob detection
        white_params = cv2.SimpleBlobDetector_Params()
        white_params.filterByArea = True
        white_params.minArea = 20
        white_params.maxArea = 500
        white_params.filterByCircularity = True
        white_params.minCircularity = 0.5
        
        white_detector = cv2.SimpleBlobDetector_create(white_params)
        white_keypoints = white_detector.detect(gray)
        
        white_spot_features = [
            len(white_keypoints),  # Number of white spots
            np.mean([k.size for k in white_keypoints]) if white_keypoints else 0,
        ]
        
        # === FEATURE 3: BLACK SPOT DETECTION (NEW for Black Spot disease) ===
        # Look for dark spots
        _, black_spots = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        black_spot_density = np.sum(black_spots > 0) / (224 * 224)
        
        # Black blob detection
        black_params = cv2.SimpleBlobDetector_Params()
        black_params.filterByArea = True
        black_params.minArea = 20
        black_params.maxArea = 500
        black_params.filterByCircularity = True
        black_params.minCircularity = 0.4
        black_params.filterByColor = True
        black_params.blobColor = 0  # 0 = dark blobs
        
        black_detector = cv2.SimpleBlobDetector_create(black_params)
        black_keypoints = black_detector.detect(gray)
        
        black_spot_features = [
            len(black_keypoints),  # Number of black spots
            np.mean([k.size for k in black_keypoints]) if black_keypoints else 0,
        ]
        
        # === FEATURE 4: TEXTURE FEATURES ===
        lbp = local_binary_pattern(gray, 8, 1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
        
        # === FEATURE 5: EDGE FEATURES ===
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (224 * 224)
        
        # === COMBINE ALL FEATURES ===
        # Total: 6 color + 1 white_density + 2 white_spot + 1 black_density + 2 black_spot + 10 texture + 1 edge = 23 features
        all_features = np.concatenate([
            color_features,           # 6 features
            [white_spot_density],     # 1 feature
            white_spot_features,      # 2 features
            [black_spot_density],     # 1 feature
            black_spot_features,      # 2 features
            lbp_hist,                 # 10 features
            [edge_density]            # 1 feature
        ])
        
        print(f"✅ Extracted {len(all_features)} features")
        return all_features
    
    def prepare_training_data(self, healthy_folder, ich_folder, black_spot_folder):
        """
        Prepare data for training with 3 classes
        
        Args:
            healthy_folder: Folder with healthy fish images (label = 0)
            ich_folder: Folder with ich/white spot images (label = 1)
            black_spot_folder: Folder with black spot images (label = 2)
        
        Returns:
            X: All features
            y: All labels (0, 1, or 2)
        """
        X = []
        y = []
        
        print("\n📚 Loading training data for 3 classes...")
        
        # Load healthy fish (label = 0)
        print(f"\n🟢 Loading Healthy fish from: {healthy_folder}")
        healthy_files = os.listdir(healthy_folder)
        for filename in healthy_files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    img_path = os.path.join(healthy_folder, filename)
                    features = self.extract_features(img_path)
                    X.append(features)
                    y.append(0)  # 0 = healthy
                    print(f"  ✓ Healthy: {filename}")
                except Exception as e:
                    print(f"  ✗ Error with {filename}: {e}")
        
        # Load ich/white spot fish (label = 1)
        print(f"\n⚪ Loading Ich (White Spot) from: {ich_folder}")
        ich_files = os.listdir(ich_folder)
        for filename in ich_files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    img_path = os.path.join(ich_folder, filename)
                    features = self.extract_features(img_path)
                    X.append(features)
                    y.append(1)  # 1 = ich
                    print(f"  ✓ Ich: {filename}")
                except Exception as e:
                    print(f"  ✗ Error with {filename}: {e}")
        
        # Load black spot fish (label = 2)
        print(f"\n⚫ Loading Black Spot from: {black_spot_folder}")
        black_files = os.listdir(black_spot_folder)
        for filename in black_files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    img_path = os.path.join(black_spot_folder, filename)
                    features = self.extract_features(img_path)
                    X.append(features)
                    y.append(2)  # 2 = black spot
                    print(f"  ✓ Black Spot: {filename}")
                except Exception as e:
                    print(f"  ✗ Error with {filename}: {e}")
        
        return np.array(X), np.array(y)
    
    def train(self, healthy_folder, ich_folder, black_spot_folder):
        """
        Train the Random Forest model for 3 classes
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        
        print("\n🎯 Starting training process for 3 classes...")
        
        # Prepare the data
        X, y = self.prepare_training_data(healthy_folder, ich_folder, black_spot_folder)
        
        print(f"\n📊 Data summary:")
        print(f"   Total images: {len(X)}")
        print(f"   Healthy (0): {sum(y==0)}")
        print(f"   Ich (1): {sum(y==1)}")
        print(f"   Black Spot (2): {sum(y==2)}")
        print(f"   Features per image: {len(X[0])}")
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n🔄 Training set: {len(X_train)} images")
        print(f"🔄 Test set: {len(X_test)} images")
        
        # Scale the features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create and train Random Forest
        print("\n🌲 Training Random Forest...")
        self.classifier = RandomForestClassifier(
            n_estimators=100,      # Number of trees
            max_depth=12,          # How deep each tree can grow
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Train the model
        self.classifier.fit(X_train_scaled, y_train)
        
        # Evaluate on test set
        train_score = self.classifier.score(X_train_scaled, y_train)
        test_score = self.classifier.score(X_test_scaled, y_test)
        
        print(f"\n📈 Training accuracy: {train_score:.2%}")
        print(f"📊 Test accuracy: {test_score:.2%}")
        
        # Save the trained model
        self.save_model()
        
        return test_score
    
    def predict(self, image_path):
        """
        Make a prediction on a new image
        Returns: class name (Healthy, Ich, or Black Spot) and probabilities
        """
        if self.classifier is None or self.scaler is None:
            if not self.load_model():
                return {
                    'error': 'Model not trained',
                    'message': 'Please run train_model.py first'
                }
        
        # Extract features from the new image
        features = self.extract_features(image_path)
        
        # Scale features the same way we did during training
        features_scaled = self.scaler.transform([features])
        
        # Get prediction (0, 1, or 2)
        prediction = self.classifier.predict(features_scaled)[0]
        
        # Get probabilities for all 3 classes
        probabilities = self.classifier.predict_proba(features_scaled)[0]
        
        result = {
            'class_id': int(prediction),
            'disease': self.class_names[prediction],
            'confidence': float(max(probabilities)),
            'probabilities': {
                'healthy': float(probabilities[0]),
                'ich': float(probabilities[1]),
                'black_spot': float(probabilities[2])
            }
        }
        
        print(f"\n🤖 Prediction result:")
        print(f"   Disease: {result['disease']}")
        print(f"   Confidence: {result['confidence']:.2%}")
        print(f"   Probabilities: Healthy={result['probabilities']['healthy']:.1%}, "
              f"Ich={result['probabilities']['ich']:.1%}, "
              f"Black Spot={result['probabilities']['black_spot']:.1%}")
        
        return result
    
    def save_model(self):
        """Save trained model to disk"""
        joblib.dump(self.classifier, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        print(f"\n💾 Model saved to {self.model_path}")
        print(f"💾 Scaler saved to {self.scaler_path}")
    
    def load_model(self):
        """Load trained model from disk"""
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.classifier = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            print(f"\n📂 Model loaded from {self.model_path}")
            return True
        else:
            print(f"\n⚠️ No trained model found. Please run train_model.py first.")
            print(f"   Looking for: {self.model_path} and {self.scaler_path}")
            return False