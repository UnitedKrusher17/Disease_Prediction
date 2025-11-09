from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

from google import genai

# --- Setup API Key and Client ---
API_KEY = os.getenv("GENAI_API_KEY")
client = genai.Client(api_key=API_KEY)

# --- FastAPI App ---
app = FastAPI()

# Allow CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Model and Features ---
model = joblib.load("rf_model_local.joblib")
le = joblib.load("label_encoder_local.joblib")
features = model.feature_names_in_

# --- Pydantic Schemas ---
class Symptoms(BaseModel):
    symptoms: list

class DiseaseDetailRequest(BaseModel):
    disease: str
    symptoms: list = []

# --- Endpoints ---

@app.get("/")
def home():
    return FileResponse("index.html")

@app.get("/all_symptoms")
def get_all_symptoms():
    # Return all feature names as symptom options
    return JSONResponse(content={"symptoms": features.tolist()})

@app.post("/predict")
def predict(data: Symptoms):
    # Convert selected symptoms to feature vector
    x_input = np.zeros(len(features))
    for i, f in enumerate(features):
        if f in data.symptoms:
            x_input[i] = 1
    x_input = x_input.reshape(1, -1)

    # Make prediction
    pred_encoded = model.predict(x_input)[0]
    pred_proba = model.predict_proba(x_input)[0]

    # Get top 3 predictions
    top_idx = np.argsort(pred_proba)[::-1][:3]
    top_diseases = le.inverse_transform(top_idx).tolist()
    top_probs = pred_proba[top_idx].tolist()

    return {"top_diseases": top_diseases, "top_probs": top_probs}

@app.post("/disease_detail")
def disease_detail(data: DiseaseDetailRequest):
    prompt = f"""
Provide an easy-to-understand overview of the disease '{data.disease}'.
1. AI Overview: What it is in simple terms.
2. Patient Experience: How people feel when going through it.
3. Severity: How serious the disease can be.
4. Recommended Actions: Steps to take or precautions.
Keep it concise and informative.
"""

    # Generate disease detail using the correct GenAI method
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )

    # Return text output
    return {"detail": response.text}
