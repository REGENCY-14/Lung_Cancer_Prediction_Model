import pickle
import pandas as pd

with open('best_rfmodel', 'rb') as file:
    best_model = pickle.load(file)

print("Model loaded successfully!")

required_features = [
    "YELLOW_FINGERS", "ANXIETY", "PEER_PRESSURE", "CHRONIC DISEASE", 
    "FATIGUE ", "ALLERGY ", "WHEEZING", "ALCOHOL CONSUMING", "COUGHING",
    "SWALLOWING DIFFICULTY", "CHEST PAIN", "ANXYELFIN"
]

input_data = {
    "YELLOW_FINGERS": 1,
    "ANXIETY": 2,
    "PEER_PRESSURE": 2,
    "CHRONIC DISEASE": 2,
    "FATIGUE ": 1,
    "ALLERGY ": 1,
    "WHEEZING": 1,
    "ALCOHOL CONSUMING": 2,
    "COUGHING": 1,
    "SWALLOWING DIFFICULTY": 1,
    "CHEST PAIN": 1
}

input_data["ANXYELFIN"] = input_data["ANXIETY"] * input_data["YELLOW_FINGERS"]

input_df = pd.DataFrame([input_data])[required_features]

prediction = best_model.predict(input_df)

print(f"Prediction for the input: {prediction[0]}")
