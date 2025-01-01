from fastapi import FastAPI
import pickle 
import pandas as pd

#Creating an instance of fastapi
app=FastAPI(
    title="water potability prediction",
    description='predicting water potability'
)

#Loading our pretrained model
with open('C:/Users/kvaru/OneDrive - UT Arlington/MLops/water potability/model.pkl','rb') as f:
    model=pickle.load(f)

    #Here is our first checkpoint

@app.get('/')
def index():
    return "Welcome to our water potability system fast api"
