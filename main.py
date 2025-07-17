from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="🌿 Garden Plant Recommender API")

# Load the model and encoders
model = joblib.load("model/garden_plant_recommender_model.pkl")
le_pot = joblib.load("model/label_encoder_pot_size.pkl")
le_plant = joblib.load("model/label_encoder_plant_name.pkl")

class PlantInput(BaseModel):
    light_intensity: float
    soil_moisture: float
    temperature: float
    humidity: float
    soil_pH: float
    pot_size: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Garden Plant Recommender API!"}

@app.post("/predict")
def predict_plant(data: PlantInput):
    try:
        pot_encoded = le_pot.transform([data.pot_size])[0]
        features = np.array([[data.light_intensity, data.soil_moisture,
                              data.temperature, data.humidity,
                              data.soil_pH, pot_encoded]])
        encoded_prediction = model.predict(features)[0]
        plant_name = le_plant.inverse_transform([encoded_prediction])[0]
        return {
            "prediction": plant_name,
            "input_received": data.dict()
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/pot_sizes")
def get_known_pot_sizes():
    return {"valid_pot_sizes": list(le_pot.classes_)}
