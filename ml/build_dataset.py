"""
AgriSmart - Dataset builder
===========================

Builds the sensor-oriented training dataset used by the AgriSmart crop
recommender, starting from the public "Crop Recommendation Dataset"
(Atharva Ingle, Kaggle, 2020; 2,200 rows, 22 crops, 100 rows per crop).

WHY THIS STEP EXISTS
--------------------
Our hardware node has NO N-P-K sensor. It measures only:
    * soil moisture        - capacitive soil-moisture sensor  (analog A0)
    * soil temperature     - DS18B20 waterproof probe         (1-Wire D4)
    * air temperature      - DHT11                            (D2)
    * air humidity         - DHT11                            (D2)
    * soil pH              - PH-4502C analog probe on a 1:2.5 soil:water
                             slurry (or manual entry from a lab / Soil
                             Health Card report)
The public dataset has temperature, humidity, pH and rainfall but no soil
moisture, soil temperature or soil type. So we DERIVE those columns from the
public values with documented, simple soil-physics relations, and DROP N, P, K.

DERIVATION (all parameters are listed below so they can be cited / changed)
--------------------------------------------------------------------------
1. soil_type   : sampled per crop from standard Indian crop-soil associations
                 (alluvial / black / red / laterite / sandy-arid).  The mapping
                 is a team assumption based on NCERT Class X Geography,
                 "Resources and Development" and "Agriculture" chapters.
2. moisture_pct: volumetric water content (%) =
                     WP + (FC - WP) * (1 - exp(-rainfall / R0)) + noise
                 FC (field capacity) and WP (wilting point) are mid-points of
                 the ranges in FAO Irrigation & Drainage Paper 56, Table 19
                 (Allen et al., 1998) for the texture class of each soil type.
3. soil_temp_c : near-surface (5-10 cm) soil temperature =
                     air_temp + 1.5 - 0.08 * (moisture - 20) + noise
                 i.e. bare soil runs slightly warmer than air in the day, and
                 wetter soil is cooler because of evaporation & heat capacity.
4. Sensor realism: values are rounded to the resolution of the real sensors
                 (DHT11 = 1 degC / 1 %RH, DS18B20 = 0.0625 degC, pH = 0.01).

THE RESULT IS PARTLY SYNTHETIC. The README and report must say so.
Run:  python ml/build_dataset.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "crop_recommendation.csv"
OUT = ROOT / "data" / "processed" / "agrismart_sensor_dataset.csv"
SEED = 42

# FAO-56 Table 19 mid-points (m3/m3 * 100 -> %), mapped to Indian soil groups
SOIL_WATER = {
    #             texture used              FC     WP
    "Alluvial": ("loam / silt loam",       27.0,  12.0),
    "Black":    ("clay (vertisol)",        36.0,  22.0),
    "Red":      ("sandy loam",             18.0,   8.0),
    "Laterite": ("sandy clay loam",        24.0,  13.0),
    "Sandy":    ("sand / loamy sand",      12.0,   4.5),
}
R0 = 120.0  # mm, rainfall at which ~63 % of the FC-WP band is filled

# Probability of each soil type for each crop (rows sum to 1)
CROP_SOILS = {
    "rice":        {"Alluvial": .6, "Black": .2, "Laterite": .2},
    "maize":       {"Alluvial": .5, "Red": .3, "Black": .2},
    "chickpea":    {"Black": .5, "Alluvial": .4, "Red": .1},
    "kidneybeans": {"Alluvial": .5, "Red": .5},
    "pigeonpeas":  {"Black": .4, "Red": .4, "Alluvial": .2},
    "mothbeans":   {"Sandy": .8, "Red": .2},
    "mungbean":    {"Alluvial": .5, "Red": .3, "Sandy": .2},
    "blackgram":   {"Black": .4, "Alluvial": .4, "Red": .2},
    "lentil":      {"Alluvial": .7, "Black": .3},
    "pomegranate": {"Red": .4, "Black": .3, "Sandy": .3},
    "banana":      {"Alluvial": .6, "Laterite": .2, "Black": .2},
    "mango":       {"Alluvial": .4, "Laterite": .3, "Red": .3},
    "grapes":      {"Black": .5, "Red": .3, "Sandy": .2},
    "watermelon":  {"Sandy": .6, "Alluvial": .4},
    "muskmelon":   {"Sandy": .6, "Alluvial": .4},
    "apple":       {"Laterite": .4, "Alluvial": .3, "Red": .3},
    "orange":      {"Red": .4, "Black": .3, "Laterite": .3},
    "papaya":      {"Alluvial": .5, "Red": .3, "Laterite": .2},
    "coconut":     {"Laterite": .5, "Sandy": .3, "Alluvial": .2},
    "cotton":      {"Black": .7, "Alluvial": .2, "Red": .1},
    "jute":        {"Alluvial": 1.0},
    "coffee":      {"Laterite": .5, "Red": .5},
}

FEATURES_NUM = ["moisture_pct", "soil_temp_c", "air_temp_c", "humidity_pct", "ph"]
FEATURES_CAT = ["soil_type"]


def build(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    raw = pd.read_csv(RAW)
    assert set(raw.label.unique()) == set(CROP_SOILS), "crop list mismatch"

    rows = []
    for idx, r in raw.iterrows():
        soils = CROP_SOILS[r.label]
        soil = rng.choice(list(soils), p=list(soils.values()))
        _, fc, wp = SOIL_WATER[soil]

        fill = 1 - np.exp(-r.rainfall / R0)
        moisture = wp + (fc - wp) * fill + rng.normal(0, 1.2)
        moisture = float(np.clip(moisture, 1.0, 55.0))

        air_t = r.temperature
        soil_t = air_t + 1.5 - 0.08 * (moisture - 20) + rng.normal(0, 0.8)

        rows.append({
            "source_row": idx,                        # traceability to raw file
            "moisture_pct": round(moisture, 1),
            "soil_temp_c": round(soil_t / 0.0625) * 0.0625,   # DS18B20 resolution
            "air_temp_c": float(round(air_t)),                # DHT11 resolution
            "humidity_pct": float(round(r.humidity)),         # DHT11 resolution
            "ph": round(r.ph, 2),
            "soil_type": soil,
            "rainfall_mm_ref": round(r.rainfall, 1),  # kept for reference, NOT a model input
            "label": r.label,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {len(df)} rows -> {OUT.relative_to(ROOT)}")
    print(df.describe().round(2).to_string())
    print(df.soil_type.value_counts().to_string())
