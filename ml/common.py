"""Shared definitions for training and inference."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "agrismart_sensor_dataset.csv"
MODEL_DIR = ROOT / "ml" / "models"
REPORT_DIR = ROOT / "ml" / "reports"

SOIL_TYPES = ["Alluvial", "Black", "Red", "Laterite", "Sandy"]

# Two feature sets: pH is optional (manual entry / optional probe)
FEATURES_FULL = ["moisture_pct", "soil_temp_c", "air_temp_c", "humidity_pct", "ph", "soil_type"]
FEATURES_NO_PH = ["moisture_pct", "soil_temp_c", "air_temp_c", "humidity_pct", "soil_type"]

# Physically possible sensor ranges (used for validation + quality scoring)
SENSOR_RANGES = {
    "moisture_pct": (0.0, 60.0),
    "soil_temp_c": (-5.0, 60.0),   # DS18B20 is fine to 125, soil never is
    "air_temp_c": (0.0, 50.0),     # DHT11 spec range
    "humidity_pct": (5.0, 100.0),  # DHT11 spec 20-90 (we accept wider)
    "ph": (3.0, 10.0),
}
