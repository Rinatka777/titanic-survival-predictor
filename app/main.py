from fastapi import FastAPI
from app.schema import Passenger
from joblib import load
import pandas as pd

app = FastAPI()

model = load("model/titanic_model.pkl")

@app.post("/predict")
def predict_passenger(passenger: Passenger):
    input_dict = passenger.model_dump()
    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)[0]
    return {"prediction": int(prediction)}




