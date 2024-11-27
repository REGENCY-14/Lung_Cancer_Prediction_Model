#Lung Cancer Prediction Model
This project utilizes a Random Forest model to predict the likelihood of lung cancer based on various health-related features. The model was trained on a dataset containing symptoms and lifestyle factors that contribute to lung cancer, providing an accurate and reliable classification.

#Streamlit Cloud Link
https://regency-14-fail-stream-lit-app-k9apf6.streamlit.app/

#Model Overview
The model predicts the risk of lung cancer by classifying input data into two categories:

Positive Outcome: Higher likelihood of lung cancer.
Negative Outcome: Lower likelihood of lung cancer.

#Key Features
Below are the health-related features used by the model for prediction:

Yellow Fingers: Indicates whether the person has yellow-stained fingers (0: No, 1: Yes).
Anxiety: Level of anxiety on a scale of 0 (none) to 3 (severe).
Peer Pressure: Influence from peers on smoking habits (0: None to 3: High).
Chronic Disease: Presence of chronic diseases (0: No, 1: Yes).
Fatigue: Persistent tiredness or weakness (0: No, 1: Yes).
Allergy: Presence of any allergies (0: No, 1: Yes).
Wheezing: Presence of wheezing or breathing difficulty (0: No, 1: Yes).
Alcohol Consuming: Habit of alcohol consumption (0: No, 1: Yes).
Coughing: Persistent coughing (0: No, 1: Yes).
Swallowing Difficulty: Trouble swallowing food or liquid (0: No, 1: Yes).
Chest Pain: Experience of chest pain (0: No, 1: Yes).
Anxyelfin: A computed feature combining anxiety and yellow fingers to capture their interaction effect.
These features collectively provide important indicators of lung cancer risks, enabling the model to make precise predictions.

#How It Works
Model Training:
A Random Forest model is trained on a labeled dataset containing instances of lung cancer and non-cancer cases.
Data Scaling:
Features are preprocessed to ensure the model works efficiently and accurately.
Prediction:
The trained model predicts whether an individual is at high or low risk for lung cancer based on the provided input.
Additionally, the model can compute probabilities for the classification.
Using the Model
You can interact with the Lung Cancer Prediction model through the Streamlit web application. Input your health details via the intuitive interface, and the app will:

#Predict whether you are at high or low risk for lung cancer.
Display the classification result along with any additional insights.
Deployment
This model is deployed as a web application using Streamlit and is accessible via Streamlit Cloud or other hosting platforms. The app offers a user-friendly experience with clear instructions for input and results.

#Requirements
To run the application locally, ensure you have the following installed:

Python 3.11
Streamlit
Pandas
NumPy
scikit-learn version 1.5.2
Pickle
Run the Application Locally
To run the app on your local machine:

Clone the repository:

bash
Copy code
git clone https://github.com/your-repository/lung-cancer-prediction.git
cd lung-cancer-prediction
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Run the Streamlit app:

bash
Copy code
streamlit run app.py
Open your browser and navigate http://localhost:8501

#Deployment to the Cloud
The app is live and can be accessed at the following link:(https://regency-14-fail-stream-lit-app-k9apf6.streamlit.app/)
