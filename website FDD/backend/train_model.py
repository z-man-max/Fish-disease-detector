"""
train_model.py - Script to train the fish disease detector for 3 classes
Run this after collecting your training images!
"""

from ml_model import FishDiseaseDetector
import os

def main():
    print("🐟 Fish Disease Detector - Training Script (3 Classes)")
    print("=" * 60)
    print("Classes: Healthy | Ich (White Spot) | Black Spot")
    print("=" * 60)
    
    # Define paths to training data
    healthy_path = "../training_data/healthy"
    ich_path = "../training_data/ich"
    black_spot_path = "../training_data/black_spot"
    
    # Check if folders exist
    missing_folders = []
    
    if not os.path.exists(healthy_path):
        missing_folders.append(f"Healthy folder: {healthy_path}")
    if not os.path.exists(ich_path):
        missing_folders.append(f"Ich folder: {ich_path}")
    if not os.path.exists(black_spot_path):
        missing_folders.append(f"Black Spot folder: {black_spot_path}")
    
    if missing_folders:
        print("\n❌ ERROR: Missing folders:")
        for folder in missing_folders:
            print(f"   - {folder}")
        print("\n📁 Please create these folders and add images:")
        print("   1. training_data/healthy/   - Add healthy fish images")
        print("   2. training_data/ich/       - Add fish with white spots")
        print("   3. training_data/black_spot/ - Add fish with black spots")
        return
    
    # Count images in each folder
    healthy_count = len([f for f in os.listdir(healthy_path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    ich_count = len([f for f in os.listdir(ich_path) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    black_count = len([f for f in os.listdir(black_spot_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"\n📸 Found training images:")
    print(f"   Healthy: {healthy_count} images")
    print(f"   Ich (White Spot): {ich_count} images")
    print(f"   Black Spot: {black_count} images")
    
    if healthy_count == 0 or ich_count == 0 or black_count == 0:
        print("\n❌ Need at least one image in EACH folder!")
        print("   Please add images to all 3 folders before training.")
        return
    
    # Create detector and train
    detector = FishDiseaseDetector()
    
    print("\n🚀 Starting training...")
    accuracy = detector.train(healthy_path, ich_path, black_spot_path)
    
    print("\n" + "=" * 60)
    print(f"✅ Training complete!")
    print(f"🎯 Model accuracy: {accuracy:.2%}")
    print("=" * 60)
    
    # Quick test on sample images
    print("\n🔍 Quick test on training images:")
    
    # Test on first healthy image
    healthy_sample = os.path.join(healthy_path, os.listdir(healthy_path)[0])
    result = detector.predict(healthy_sample)
    print(f"   Healthy test: {result['disease']} ✓")
    
    # Test on first ich image
    ich_sample = os.path.join(ich_path, os.listdir(ich_path)[0])
    result = detector.predict(ich_sample)
    print(f"   Ich test: {result['disease']} ✓")
    
    # Test on first black spot image
    black_sample = os.path.join(black_spot_path, os.listdir(black_spot_path)[0])
    result = detector.predict(black_sample)
    print(f"   Black Spot test: {result['disease']} ✓")
    
    print("\n💡 Next steps:")
    print("   1. Run 'python app.py' to start the Flask server")
    print("   2. Open your frontend index.html")
    print("   3. Test with new fish photos!")

if __name__ == "__main__":
    main()