from fastapi import FastAPI
import pickle 
import pandas as pd
from data_model import Water

#Creating an instance of fastapi
app=FastAPI(
    title="water potability prediction",
    description='predicting water potability'
)

#Loading our pretrained model
with open('model.pkl','rb') as f:
    model=pickle.load(f)

    #Here is our first checkpoint

@app.get('/')
def index():
    return "Welcome to our water potability system fast api"

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

    if predicted_value == 1:
        return "Water is Consumable"
    else:
        return "Water is not Consumable"

    
