
from pathlib import Path
from time import perf_counter
from datetime import datetime, timezone
import json

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "models" / "accident_severity_model.joblib"
TRAIN_PATH = PROJECT_DIR / "ml" / "artifacts" / "train.csv"
MONITOR_DIR = PROJECT_DIR / "monitoring"
MONITOR_LOG = MONITOR_DIR / "prediction_logs.jsonl"

MONITOR_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Traffic Accident Severity API",
    description="Predict accident severity using the trained ML model.",
    version="1.1.0",
)

model = joblib.load(MODEL_PATH)
FEATURES = [
    c for c in pd.read_csv(TRAIN_PATH, nrows=0).columns
    if c != "collision_severity"
]


def log_request(predicted_severity, latency_ms, success):
    """Append one request's monitoring data to a JSON Lines log."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "predicted_severity": predicted_severity,
        "latency_ms": round(latency_ms, 2),
        "success": success,
    }

    try:
        with MONITOR_LOG.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")
    except OSError as exc:
        # A logging problem should not hide the prediction result.
        print(f"Monitoring log write failed: {exc}")


class AccidentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    collision_year: int = 2023
    location_easting_osgr: float = 530000
    location_northing_osgr: float = 180000
    longitude: float = -0.12
    latitude: float = 51.50
    police_force: int = 1
    number_of_vehicles: int = 2
    day_of_week: int = 3
    local_authority_district: int = 1
    local_authority_ons_district: str = "E09000001"
    local_authority_highway: str = "E09000001"
    local_authority_highway_current: str = "E09000001"
    first_road_class: int = 3
    first_road_number: int = 0
    road_type: int = 6
    speed_limit: int = 30
    junction_detail_historic: int = 0
    junction_detail: int = 0
    junction_control: int = 0
    second_road_class: int = -1
    second_road_number: int = 0
    pedestrian_crossing_human_control_historic: int = 0
    pedestrian_crossing_physical_facilities_historic: int = 0
    pedestrian_crossing: int = 0
    light_conditions: int = 1
    weather_conditions: int = 1
    road_surface_conditions: int = 1
    special_conditions_at_site: int = 0
    carriageway_hazards_historic: int = 0
    carriageway_hazards: int = 0
    urban_or_rural_area: int = 1
    did_police_officer_attend_scene_of_accident: int = 1
    trunk_road_flag: int = 2
    lsoa_of_accident_location: str = "E01000001"
    year: int = 2023
    month: int = 6
    month_name: str = "June"
    day: int = 15
    weekday: str = "Thursday"
    hour: int = 14
    road_type_label: str = "Single carriageway"
    weather_label: str = "Fine no high winds"
    light_conditions_label: str = "Daylight"
    road_surface_label: str = "Dry"
    urban_rural_label: str = "Urban"


@app.get("/")
def home():
    return {
        "message": "Traffic Accident Severity API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
def predict(accident: AccidentInput):
    start_time = perf_counter()

    try:
        row = pd.DataFrame(
            [accident.model_dump()],
            columns=FEATURES,
        )
        prediction = int(model.predict(row)[0])
        latency_ms = (perf_counter() - start_time) * 1000

        log_request(
            predicted_severity=prediction,
            latency_ms=latency_ms,
            success=True,
        )

        labels = {1: "Fatal", 2: "Serious", 3: "Slight"}

        return {
            "predicted_severity": prediction,
            "severity_label": labels.get(prediction, "Unknown")
        }

    except Exception as exc:
        latency_ms = (perf_counter() - start_time) * 1000

        log_request(
            predicted_severity=None,
            latency_ms=latency_ms,
            success=False,
        )

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {type(exc).__name__}: {exc}",
        ) from exc
