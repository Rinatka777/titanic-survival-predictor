from fastapi import FastAPI
from fastapi import FastAPI
from app.schema import Passenger
from joblib import load
import pandas as pd

app = FastAPI()

# Load trained model
model = load("model/titanic_model.pkl")

@app.post("/predict")
def predict_passenger(passenger: Passenger):
    # 1. Convert raw input to model-ready features
    sex = 1 if passenger.Sex.lower() == "female" else 0
    embarked_q = 1 if passenger.Embarked.upper() == "Q" else 0
    embarked_s = 1 if passenger.Embarked.upper() == "S" else 0

    # 2. Build input dictionary for model
    input_dict = {
        "Pclass": passenger.Pclass,
        "Sex": sex,
        "Age": passenger.Age,
        "SibSp": passenger.SibSp,
        "Parch": passenger.Parch,
        "Fare": passenger.Fare,
        "Embarked_Q": embarked_q,
        "Embarked_S": embarked_s
    }

    # 3. Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # 4. Make prediction
    prediction = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)
    survival_prob = probs[0][1]

    # 5. Return result
    return {
        "prediction": int(prediction),
        "survival_probability": round(float(survival_prob), 3)
    }







