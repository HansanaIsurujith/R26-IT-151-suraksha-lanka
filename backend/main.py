from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import pickle
import numpy as np
import os
from datetime import datetime

# ─────────────────────────────────────────────
# Load trained model
# ─────────────────────────────────────────────
MODEL_PATH = "./disaster_rf_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file '{MODEL_PATH}' not found. "
        "Run train_model.py first to generate it."
    )

with open(MODEL_PATH, "rb") as f:
    model_data = pickle.load(f)

model       = model_data["model"]
feature_cols = model_data["feature_cols"]
label_map   = model_data["label_map"]
accuracy    = model_data["accuracy"]

print(f"Model loaded. Training accuracy: {accuracy * 100:.2f}%")

# ─────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────
app = FastAPI(
    title="Disaster Prediction API",
    description="Predicts Flood, Landslide, Fog, Warning, No Risk based on weather data.",
    version="1.0.0"
)

# CORS — allow React Native app to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your app domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────
class WeatherInput(BaseModel):
    rainfall:    float = Field(..., description="Current rainfall in mm")
    rain_3d:     float = Field(..., description="Cumulative rainfall last 3 days (mm)")
    rain_7d:     float = Field(..., description="Cumulative rainfall last 7 days (mm)")
    temperature: float = Field(..., description="Temperature in Celsius")
    humidity:    float = Field(..., description="Humidity percentage (0-100)")
    latitude:    float = Field(..., description="Latitude of location")
    longitude:   float = Field(..., description="Longitude of location")
    elevation:   float = Field(..., description="Elevation in meters")
    date:        Optional[str] = Field(None, description="Date in YYYY-MM-DD format. Defaults to today.")

    class Config:
        json_schema_extra = {
            "example": {
                "rainfall":    25.0,
                "rain_3d":     80.0,
                "rain_7d":     130.0,
                "temperature": 27.5,
                "humidity":    85.0,
                "latitude":    6.9271,
                "longitude":   79.8612,
                "elevation":   8.0,
                "date":        "2026-05-10"
            }
        }

class BatchWeatherInput(BaseModel):
    locations: List[WeatherInput]

class PredictionResult(BaseModel):
    event:       str
    event_label: int
    confidence:  float
    risk_level:  str
    message:     str
    probabilities: dict

# ─────────────────────────────────────────────
# Helper: build feature vector from input
# ─────────────────────────────────────────────
MONSOON_MONTHS = {5, 6, 7, 8, 9, 12, 1, 2}

def build_features(data: WeatherInput) -> np.ndarray:
    # Parse date
    date_str = data.date or datetime.today().strftime("%Y-%m-%d")
    dt = datetime.strptime(date_str, "%Y-%m-%d")

    month       = dt.month
    day_of_year = dt.timetuple().tm_yday
    day_of_week = dt.weekday()
    is_monsoon  = 1 if month in MONSOON_MONTHS else 0

    # Engineered features
    rain_accel       = data.rain_3d - (data.rain_7d / 7) * 3
    humid_rain_index = (data.humidity * data.rain_3d) / 100
    heat_index       = (data.temperature * data.humidity) / 100
    elev_risk        = 2 if data.elevation <= 9 else (1 if data.elevation <= 14 else 0)

    # MinMax scale using approximate training ranges
    # These ranges match the original dataset_clean.csv
    RANGES = {
        "Rainfall":         (0.0,  85.0),
        "Rain_3d":          (0.0, 210.0),
        "Rain_7d":          (0.0, 420.0),
        "Temperature":      (24.5, 32.0),
        "Humidity":         (60.0, 95.0),
        "Latitude":         (6.82,  7.42),
        "Longitude":        (79.84, 80.10),
        "Elevation":        (2.0,  25.0),
        "Rain_Accel":       (-60.0, 90.0),
        "Humid_Rain_Index": (0.0, 180.0),
        "Heat_Index":       (15.0,  30.0),
    }

    def scale(val, col):
        lo, hi = RANGES.get(col, (0, 1))
        if hi == lo:
            return 0.0
        return max(0.0, min(1.0, (val - lo) / (hi - lo)))

    raw = {
        "Rainfall":         data.rainfall,
        "Rain_3d":          data.rain_3d,
        "Rain_7d":          data.rain_7d,
        "Temperature":      data.temperature,
        "Humidity":         data.humidity,
        "Latitude":         data.latitude,
        "Longitude":        data.longitude,
        "Elevation":        data.elevation,
        "Month":            month,
        "DayOfYear":        day_of_year,
        "DayOfWeek":        day_of_week,
        "Is_Monsoon":       is_monsoon,
        "Rain_Accel":       rain_accel,
        "Humid_Rain_Index": humid_rain_index,
        "Heat_Index":       heat_index,
        "Elev_Risk":        elev_risk,
    }

    # Scale only numeric columns (not binary/categorical flags)
    SCALE_KEYS = {
        "Rainfall", "Rain_3d", "Rain_7d", "Temperature", "Humidity",
        "Latitude", "Longitude", "Elevation",
        "Rain_Accel", "Humid_Rain_Index", "Heat_Index"
    }

    feature_vector = []
    for col in feature_cols:
        val = raw.get(col, 0.0)
        if col in SCALE_KEYS:
            val = scale(val, col)
        feature_vector.append(val)

    return np.array(feature_vector).reshape(1, -1)


def build_result(features: np.ndarray) -> PredictionResult:
    prediction   = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    prob_dict    = {
        label_map[i]: round(float(p), 4)
        for i, p in enumerate(probabilities)
        if i in label_map
    }
    event       = label_map[int(prediction)]
    confidence  = round(float(max(probabilities)) * 100, 2)

    # Risk level mapping
    risk_map = {
        "No Risk":   ("LOW",      "No disaster risk detected. Conditions are safe."),
        "Warning":   ("MEDIUM",   "Weather conditions are concerning. Stay alert."),
        "Flood":     ("HIGH",     "High flood risk detected! Avoid low-lying areas."),
        "Landslide": ("HIGH",     "Landslide risk detected! Avoid hilly terrain."),
        "Fog":       ("MODERATE", "Dense fog expected. Drive carefully, visibility low."),
    }
    risk_level, message = risk_map.get(event, ("UNKNOWN", "Unable to determine risk."))

    return PredictionResult(
        event=event,
        event_label=int(prediction),
        confidence=confidence,
        risk_level=risk_level,
        message=message,
        probabilities=prob_dict
    )

# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Check if API is running."""
    return {
        "status":   "ok",
        "model":    "Random Forest",
        "accuracy": f"{accuracy * 100:.2f}%",
        "events":   list(label_map.values()),
        "version":  "1.0.0"
    }


@app.post("/predict", response_model=PredictionResult)
def predict(data: WeatherInput):
    """
    Predict disaster risk for a single location.
    Send current weather data and get back event type + risk level.
    """
    try:
        features = build_features(data)
        return build_result(features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch(data: BatchWeatherInput):
    """
    Predict disaster risk for multiple locations at once.
    Useful for map-based predictions across a grid.
    """
    if len(data.locations) > 100:
        raise HTTPException(
            status_code=400,
            detail="Batch limit is 100 locations per request."
        )
    try:
        results = []
        for loc in data.locations:
            features = build_features(loc)
            result   = build_result(features)
            results.append({
                "latitude":   loc.latitude,
                "longitude":  loc.longitude,
                "event":      result.event,
                "risk_level": result.risk_level,
                "confidence": result.confidence,
                "message":    result.message,
            })
        return {"count": len(results), "predictions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
