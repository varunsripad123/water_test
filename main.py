from fastapi import FastAPI
import pandas as pd
from joblib import load  # Import load function from joblib
from data_model import Water

# Create an instance of FastAPI
app = FastAPI(
    title="Water Potability Prediction",
    description='Predicting water potability'
)

# Load the pretrained model
# Update the path if necessary to be relative or correctly configured for deployment
model = load('model.joblib')

@app.get('/')
def index():
    return "Welcome to our water potability prediction system"

@app.post('/predict')
def model_predict(water: Water):
    sample = pd.DataFrame([{
        'ph': water.ph,
        'Hardness': water.Hardness,
        'Solids': water.Solids,
        'Chloramines': water.Chloramines,
        'Sulfate': water.Sulfate,
        'Conductivity': water.Conductivity,
        'Organic_carbon': water.Organic_carbon,
        'Trihalomethanes': water.Trihalomethanes,  # Corrected spelling
        'Turbidity': water.Turbidity
    }])

    predicted_value = model.predict(sample)

    return "Water is Consumable" if predicted_value == 1 else "Water is not Consumable"
