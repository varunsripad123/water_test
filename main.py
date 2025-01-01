from fastapi import FastAPI, HTTPException
import pickle
import pandas as pd
from pydantic import BaseModel
import logging

# Define the data model for incoming data using Pydantic
class Water(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float

# Set up basic logging
logging.basicConfig(level=logging.INFO)

# Creating an instance of FastAPI
app = FastAPI(
    title="Water Potability Prediction",
    description="Predicting water potability"
)

# Load the pre-trained model
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    # Optionally raise an HTTPException or continue depending on your error handling policy

@app.get('/')
def index():
    return "Welcome to our water potability prediction system"

@app.post('/predict')
def model_predict(water: Water):
    try:
        sample = pd.DataFrame([water.dict()])
        predicted_value = model.predict(sample)
        result = "Water is Consumable" if predicted_value[0] == 1 else "Water is not Consumable"
        return result
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

