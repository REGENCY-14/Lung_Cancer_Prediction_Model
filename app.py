from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

with open('best_rfmodel', 'rb') as file:
    best_model = pickle.load(file)

print("Model loaded successfully!")

required_features = [
    "YELLOW_FINGERS", "ANXIETY", "PEER_PRESSURE", "CHRONIC DISEASE", 
    "FATIGUE ", "ALLERGY ", "WHEEZING", "ALCOHOL CONSUMING", "COUGHING",
    "SWALLOWING DIFFICULTY", "CHEST PAIN", "ANXYELFIN"
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        
        missing_features = [feature for feature in required_features[:-1] if feature not in input_data]
        if missing_features:
            return jsonify({"error": f"Missing features: {', '.join(missing_features)}"}), 400

        input_data["ANXYELFIN"] = input_data["ANXIETY"] * input_data["YELLOW_FINGERS"]
        
        input_df = pd.DataFrame([input_data])[required_features]
        
        prediction = best_model.predict(input_df)
        
        return jsonify({"prediction": int(prediction[0])})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
     app.run(host='0.0.0.0', port=5000, debug=True)
