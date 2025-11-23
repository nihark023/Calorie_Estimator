# backend/app.py

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

from pathlib import Path
import sys

# Make sure Python can find the 'ml' and 'data' folders
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from ml.predict import predict_food  # type: ignore

app = FastAPI(
    title="Food Calorie Estimator API",
    description="Simple API to estimate calories from a food image + quantity",
    version="0.1.0"
)

# Allow frontend (later React/Flutter/web) to call this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in prod, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load nutrition data once at startup
NUTRITION_CSV_PATH = BASE_DIR / "data" / "food_nutrition.csv"
nutrition_df = pd.read_csv(NUTRITION_CSV_PATH)

class CalorieResponse(BaseModel):
    food: str
    confidence: float
    quantity_in_grams: float
    estimated_calories: float

def get_calories(food_name: str, grams: float) -> float:
    # Simple lookup in the CSV (case-insensitive)
    df = nutrition_df
    row = df[df["food_name"].str.lower() == food_name.lower()]
    if row.empty:
        # If food not found, return 0 (or you can raise an error)
        return 0.0
    cal_per_100g = float(row.iloc[0]["calories_per_100g"])
    return cal_per_100g * (grams / 100.0)


@app.get("/")
def root():
    return {"message": "Food Calorie Estimator API is running"}


@app.post("/predict-food", response_model=CalorieResponse)
async def predict_food_endpoint(
    file: UploadFile = File(...),
    quantity_in_grams: float = Form(100.0)
):
    """
    Accepts:
      - image file (jpg/png)
      - quantity in grams (default 100g)
    Returns:
      - predicted food label
      - confidence
      - calories
    """
    # Read image bytes
    image_bytes = await file.read()

    # Call dummy ML predictor
    result = predict_food(image_bytes)
    food_label = result["label"]
    confidence = result["confidence"]

    # Lookup calories
    calories = get_calories(food_label, quantity_in_grams)

    return CalorieResponse(
        food=food_label,
        confidence=confidence,
        quantity_in_grams=quantity_in_grams,
        estimated_calories=calories
    )
