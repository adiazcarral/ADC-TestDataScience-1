import torch
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from src.adc_testdatascience_1.models.logistic import LogisticRegression
from src.adc_testdatascience_1.utils.data_utils import get_dataloaders

app = FastAPI()

# Load the model from pickle file
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LogisticRegression()

# Assuming you've saved the model as a pickle file
with open("src/adc_testdatascience_1/models/logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

model.eval()  # Set model to evaluation mode

# Pydantic model for input validation
class PredictionInput(BaseModel):
    inputs: list

# Prediction endpoint
@app.post("/predict")
async def predict(data: PredictionInput):
    inputs = torch.tensor(data.inputs).to(device)
    with torch.no_grad():
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)  # Get predicted class
        return {"prediction": predicted.item()}
