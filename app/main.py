## app/main.py this is where everything runs
from fastapi import FastAPI
import uvicorn
import joblib
import numpy as np
from pydantic import BaseModel

# initiate an instance of fastapi
app = FastAPI()

# Load the model 
model = joblib.load("model/model.pkl")

class Inputs(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def read_root():
    return {"message": "Hello, this is your FastAPI endpoint!"}

@app.post('/predict')
def predict_species(data:Inputs):
    data = data.dict()
    sepal_length = data['sepal_length']
    sepal_width = data['sepal_width']
    petal_length = data['petal_length']
    petal_width = data['petal_width']

    prediction = model.predict([[sepal_length,sepal_width, petal_length,petal_width]])

    return {'prediction': prediction}
    
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8080)