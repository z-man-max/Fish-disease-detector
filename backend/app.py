# backend/app.py - Updated with REAL ML for 3 Classes (Healthy, Ich, Black Spot)
from flask import Flask, request, jsonify
from flask_cors import CORS
from ml_model import FishDiseaseDetector
import os
import tempfile

app = Flask(__name__)
CORS(app)

# Create uploads folder
os.makedirs('uploads', exist_ok=True)

# Initialize the detector
detector = FishDiseaseDetector()

# Try to load trained model
if detector.load_model():
    print("✅ ML Model loaded successfully!")
    print(f"   Model supports: {detector.class_names}")
else:
    print("⚠️ No trained model found. Please run train_model.py first!")

@app.route('/')
def home():
    return "Flask backend is running! Go to /api/analyze to test."

@app.route('/api/health', methods=['GET'])
def health_check():
    """Test if backend is alive"""
    return jsonify({
        "status": "healthy",
        "message": "Backend is running!",
        "service": "Fish Detection API",
        "model_loaded": detector.classifier is not None,
        "supported_classes": detector.class_names if detector.classifier else None
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_fish():
    """Receive image, run ML model, return analysis for 3 classes"""
    print("🎣 Received fish image for analysis!")
    
    # 1. Check if image was sent
    if 'image' not in request.files:
        return jsonify({"error": "No image sent"}), 400
    
    file = request.files['image']
    
    # Check if model is loaded
    if detector.classifier is None:
        return jsonify({
            "error": "ML Model not trained yet",
            "message": "Please run train_model.py first with training images"
        }), 503
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        file.save(tmp.name)
        temp_path = tmp.name
    
    try:
        # Run prediction (now returns class_id, disease name, and probabilities)
        result = detector.predict(temp_path)
        
        # Check if there was an error
        if 'error' in result:
            return jsonify({"error": result['error'], "message": result['message']}), 500
        
        # Format response based on prediction (class_id)
        if result['class_id'] == 0:  # HEALTHY
            analysis = {
                "diagnosis": "Fish Appears Healthy",
                "confidence": round(result['confidence'] * 100, 1),
                "status": "healthy",
                "diseases_detected": [],
                "recommendations": [
                    "Maintain regular water changes",
                    "Monitor for any behavioral changes",
                    "Keep water temperature stable",
                    "Feed high-quality food"
                ],
                "probabilities": {
                    "healthy": round(result['probabilities']['healthy'] * 100, 1),
                    "ich": round(result['probabilities']['ich'] * 100, 1),
                    "black_spot": round(result['probabilities']['black_spot'] * 100, 1)
                }
            }
        
        elif result['class_id'] == 1:  # ICH (WHITE SPOT)
            analysis = {
                "diagnosis": "Ich (White Spot Disease)",
                "confidence": round(result['confidence'] * 100, 1),
                "status": "infected",
                "diseases_detected": [
                    {
                        "name": "Ichthyophthirius (White Spot)",
                        "confidence": round(result['probabilities']['ich'] * 100, 1),
                        "description": "White spots visible on body and fins"
                    }
                ],
                "recommendations": [
                    "⚠️ Isolate infected fish immediately",
                    "🌡️ Increase water temperature to 30°C (86°F)",
                    "🧂 Add aquarium salt (1 tbsp per 5 gallons)",
                    "💊 Use anti-parasitic medication (malachite green or formalin)",
                    "💧 Perform 25% water changes daily"
                ],
                "probabilities": {
                    "healthy": round(result['probabilities']['healthy'] * 100, 1),
                    "ich": round(result['probabilities']['ich'] * 100, 1),
                    "black_spot": round(result['probabilities']['black_spot'] * 100, 1)
                }
            }
        
        else:  # result['class_id'] == 2 - BLACK SPOT
            analysis = {
                "diagnosis": "Black Spot Disease",
                "confidence": round(result['confidence'] * 100, 1),
                "status": "infected",
                "diseases_detected": [
                    {
                        "name": "Black Spot (Diplopstomiasis)",
                        "confidence": round(result['probabilities']['black_spot'] * 100, 1),
                        "description": "Dark/black cysts visible on skin and fins"
                    }
                ],
                "recommendations": [
                    "🔍 Quarantine affected fish immediately",
                    "💊 Treat with anti-parasitic medication (Praziquantel)",
                    "🧹 Clean the aquarium thoroughly",
                    "🐌 Remove snails (they are intermediate hosts)",
                    "🔄 Perform 30% water change every other day"
                ],
                "probabilities": {
                    "healthy": round(result['probabilities']['healthy'] * 100, 1),
                    "ich": round(result['probabilities']['ich'] * 100, 1),
                    "black_spot": round(result['probabilities']['black_spot'] * 100, 1)
                }
            }
        
        # Save a copy to uploads for reference
        file.save(f"uploads/{file.filename}")
        
        return jsonify({
            "success": True,
            "analysis": analysis,
            "image_info": {
                "filename": file.filename,
                "size": request.content_length
            }
        })
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

if __name__ == '__main__':
    print("🚀 Starting Flask backend...")
    print("🔗 Backend URL: http://localhost:5000")
    print("📁 Uploads folder: backend/uploads/")
    print("🎯 Supported classes: Healthy, Ich (White Spot), Black Spot")
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)