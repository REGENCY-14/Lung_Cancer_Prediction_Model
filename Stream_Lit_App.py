import streamlit as st
import pickle
import pandas as pd

# Load the pre-trained Random Forest model
with open('best_rfmodel', 'rb') as file:
    best_model = pickle.load(file)

st.title("Lung Cancer Prediction using Random Forest Model")
st.write("Enter the details below to predict the health outcome:")

# Define user input fields
YELLOW_FINGERS = st.selectbox("Yellow Fingers (0: No, 1: Yes)", [0, 1], index=1)
ANXIETY = st.slider("Anxiety Level (0: None to 3: Severe)", 0, 3, value=2)
PEER_PRESSURE = st.slider("Peer Pressure Level (0: None to 3: High)", 0, 3, value=2)
CHRONIC_DISEASE = st.slider("Chronic Disease Presence (0: No, 1: Yes)", 0, 1, value=1)
FATIGUE = st.selectbox("Fatigue (0: No, 1: Yes)", [0, 1], index=1)
ALLERGY = st.selectbox("Allergy (0: No, 1: Yes)", [0, 1], index=1)
WHEEZING = st.selectbox("Wheezing (0: No, 1: Yes)", [0, 1], index=1)
ALCOHOL_CONSUMING = st.selectbox("Alcohol Consuming (0: No, 1: Yes)", [0, 1], index=1)
COUGHING = st.selectbox("Coughing (0: No, 1: Yes)", [0, 1], index=1)
SWALLOWING_DIFFICULTY = st.selectbox("Swallowing Difficulty (0: No, 1: Yes)", [0, 1], index=1)
CHEST_PAIN = st.selectbox("Chest Pain (0: No, 1: Yes)", [0, 1], index=1)

# Automatically compute "ANXYELFIN"
ANXYELFIN = ANXIETY * YELLOW_FINGERS

# Create input data dictionary
input_data = {
    "YELLOW_FINGERS": YELLOW_FINGERS,
    "ANXIETY": ANXIETY,
    "PEER_PRESSURE": PEER_PRESSURE,
    "CHRONIC DISEASE": CHRONIC_DISEASE,
    "FATIGUE ": FATIGUE,
    "ALLERGY ": ALLERGY,
    "WHEEZING": WHEEZING,
    "ALCOHOL CONSUMING": ALCOHOL_CONSUMING,
    "COUGHING": COUGHING,
    "SWALLOWING DIFFICULTY": SWALLOWING_DIFFICULTY,
    "CHEST PAIN": CHEST_PAIN,
    "ANXYELFIN": ANXYELFIN
}

# Organize data into a DataFrame for prediction
input_df = pd.DataFrame([input_data])

# Perform prediction when button is clicked
if st.button("Predict"):
    try:
        prediction = best_model.predict(input_df)
        st.success(f"Prediction: {'Positive Outcome' if prediction[0] == 1 else 'Negative Outcome'}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
