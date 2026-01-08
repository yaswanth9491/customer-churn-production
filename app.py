from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

# 1. Initialize the app and load the model
app = FastAPI(title="Customer Churn Prediction API")
model = joblib.load("models/churn_model_v1.joblib")

# 2. Define the data structure for input
class CustomerData(BaseModel):
    Age: int
    Gender: int  # 0 or 1 based on your LabelEncoder
    Tenure: int
    Usage_Frequency: int
    Support_Calls: int
    Payment_Delay: int
    Subscription_Type: int # 0, 1, 2
    Contract_Length: int # 0, 1, 2
    Total_Spend: float
    Last_Interaction: int

@app.get("/")
def home():
    return {"message": "Churn Prediction API is Running"}

@app.post("/predict")
def predict(data: CustomerData):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 2),
        "status": "Churn Risk" if prediction == 1 else "Loyal Customer"
    }
