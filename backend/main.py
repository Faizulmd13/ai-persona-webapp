# # backend/main.py
# from fastapi import FastAPI, Request
# from fastapi.middleware.cors import CORSMiddleware
# import joblib
# import numpy as np

# app = FastAPI()

# # Enable CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # for testing, replace with frontend URL
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load model components
# model = joblib.load("persona_model.pkl")
# scaler = joblib.load("scaler.pkl")
# education_encoder = joblib.load("education_encoder.pkl")
# occupation_encoder = joblib.load("occupation_encoder.pkl")
# label_encoder = joblib.load("label_encoder.pkl")

# @app.post("/predict")
# async def predict_persona(request: Request):
#     data = await request.json()
#     age = data["age"]
#     education = education_encoder.transform([data["education"]])[0]
#     occupation = occupation_encoder.transform([data["occupation"]])[0]

#     features = np.array([[age, education, occupation]])
#     scaled = scaler.transform(features)
#     prediction = model.predict(scaled)[0]
#     persona = label_encoder.inverse_transform([prediction])[0]
    
#     return {"persona": persona}


# pip install fastapi uvicorn scikit-learn numpy joblib

# cd backend
# uvicorn main:app --reload


from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load models and encoders
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('education_encoder.pkl', 'rb') as f:
    education_encoder = pickle.load(f)

with open('occupation_encoder.pkl', 'rb') as f:
    occupation_encoder = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('persona_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    age = data['age']
    education = data['education']
    employment = data['employment']

    try:
        # Encode inputs
        education_encoded = education_encoder.transform([education])[0]
        employment_encoded = occupation_encoder.transform([employment])[0]

        features = np.array([[age, education_encoded, employment_encoded]])
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)
        persona = label_encoder.inverse_transform(prediction)[0]

        return jsonify({'persona': persona})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
