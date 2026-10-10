# AgriSmart: Smart Farm & Crop Recommendation System

**BCSE302L / BCSE302P Database Management Systems, VIT**
An IoT-fed relational database (MySQL 8) that recommends the best crop for a plot from **low-cost soil sensors**, using no NPK sensor, plus its own cultivation history.

```
Arduino Uno (moisture + DHT11 + DS18B20)  --USB serial-->  serial_bridge.py  --INSERT-->  MySQL
                                                                                          |
          Flask website  <-- views / triggers / procedures / fn_virtual_n_index ----------+
                |
                +--> ML engine (calibrated classifier + QW-TC + abstention + context re-ranking)
```

### Dataset credit

The models are trained on the **Crop Recommendation Dataset** by **Atharva Ingle**, published on Kaggle (2020):
https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset

- `data/raw/crop_recommendation.csv` is the author's file, unchanged (2,200 rows, 22 crops).
- `data/processed/agrismart_sensor_dataset.csv` is derived from it by `ml/build_dataset.py`. N, P, K and rainfall are dropped, and soil moisture, soil temperature and soil type are calculated with documented formulas (see §4).

All credit for the original data goes to its author. Please cite the dataset (reference [1] in §9) if you reuse it, and check the Kaggle page for its licence terms.

---

## 1. Folder structure

```
AgriSmart/
├── README.md                     <- this file
├── requirements.txt              <- Python packages
├── config.py / .env.example      <- DB + serial settings (copy .env.example -> .env)
├── data/
│   ├── raw/crop_recommendation.csv          public dataset (unchanged)
│   └── processed/agrismart_sensor_dataset.csv   sensor-oriented dataset (derived, see §4)
├── ml/
│   ├── build_dataset.py          raw -> processed dataset (documented derivation)
│   ├── train_models.py           8 models x 2 feature sets, saves every evaluation
│   ├── experiments.py            novelty experiments (exp3-exp6)
│   ├── recommender.py            the recommendation engine used by the website
│   ├── crop_knowledge.py         season / family / N-demand / maturation / yield per crop
│   ├── common.py                 feature lists, sensor ranges
│   └── models/                   model_meta.json (+ *.joblib created by train_models.py; not shipped, ~50 MB)
├── experiments/                  ALL evaluation output for the report ("Experimentation")
│   ├── EXPERIMENTS_SUMMARY.md    every result table in one file (auto-generated)
│   ├── exp0_dataset/             statistics, correlation, box plots
│   ├── exp1_npk_ablation/        cost of not having an NPK sensor
│   ├── exp2_model_comparison/    summary + per-fold CV + calibration + feature importance
│   │   └── per_model/<with_pH|without_pH>/<model>/  metrics.json, classification_report.csv,
│   │                                                confusion_matrix.csv/.png  (16 folders)
│   ├── exp3_temporal_consensus/  novelty N1
│   ├── exp4_adaptive_model_selection/  novelty N2
│   ├── exp5_abstention/          novelty N3
│   └── exp6_context_reranking/   novelty N4
├── database/
│   ├── 01_schema.sql             9 tables, 3NF, PK/FK/CHECK constraints
│   ├── 02_views.sql              8 views incl. v_recommended_crops (SQL rule-based recommender)
│   ├── 03_routines.sql           3 triggers, 4 procedures, 1 function
│   ├── 04_seed_data.sql          demo data (generated)
│   └── generate_seed.py          generator for 04_seed_data.sql
├── webapp/                       Flask website
│   ├── app.py  services.py  db.py
│   ├── templates/                11 pages + 3 partials
│   └── static/css/style.css
├── hardware/
│   ├── arduino/agrismart_node/agrismart_node.ino
│   ├── bridge/serial_bridge.py   serial -> MySQL (also --simulate without hardware)
│   └── CALIBRATION.md            wiring + sensor calibration
├── scripts/
│   ├── setup_windows.ps1         installs MySQL if missing, venv, packages, database
│   ├── setup_database.py         creates DB + app user, loads all SQL files
│   └── run_pipeline.py           rebuilds dataset, models, experiments, seed data
└── docs/NOVELTY.md               the novel contributions with measured results
```

## 2. Quick start (Windows)

1. Install **Python 3.10+** (Anaconda is fine).
2. Open PowerShell in this folder and run
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
   ```
   The script:
   - checks for MySQL and, if it is missing, installs MySQL Server with `winget` (you set the root password in the MySQL configurator);
   - creates `.venv` and installs `requirements.txt`;
   - writes `.env`;
   - runs `scripts\setup_database.py`.
3. Start the website:
   ```powershell
   .\.venv\Scripts\python.exe -m webapp.app
   ```
   Then open **http://127.0.0.1:5000**.

**Manual alternative.** Install MySQL 8 from https://dev.mysql.com/downloads/installer/, then run:

```bash
pip install -r requirements.txt
cd ml && python train_models.py && cd ..   # creates ml/models/*.joblib (~3 min, seeded -> same results)
copy .env.example .env          # put your MySQL root password in DB_ADMIN_PASSWORD
python scripts/setup_database.py
python -m webapp.app
```

**Without the Arduino**, run `python hardware/bridge/serial_bridge.py --simulate --interval 10`. NODE-01 then sends fake readings every 10 s, so the live data path can still be demonstrated.

**With the Arduino**, upload `agrismart_node.ino`, find the port with `python hardware/bridge/serial_bridge.py --list-ports`, then run `python hardware/bridge/serial_bridge.py --port COM3`.

**Rebuilding everything from the raw dataset** takes about 10–15 min:

```bash
python scripts/run_pipeline.py
python scripts/setup_database.py
```

## 3. Hardware (simple, breadboard only)

| Sensor | Measures | Pin |
|---|---|---|
| Capacitive soil-moisture sensor | soil volumetric water content % | A0 |
| DHT11 | air temperature, relative humidity | D2 |
| DS18B20 waterproof probe (+4.7 kΩ) | soil temperature | D4 |
| *optional* PH-4502C | soil pH (1:2.5 soil-water slurry) | A1 |

There is **no NPK sensor**. pH is optional: type it in from a Soil Health Card or test kit, or skip it, in which case the sensor-only model is used automatically. See `hardware/CALIBRATION.md`.

## 4. Dataset: where every number comes from

| Item | Source | Real or derived |
|---|---|---|
| `data/raw/crop_recommendation.csv` | *Crop Recommendation Dataset*, A. Ingle, Kaggle (2020), 2,200 rows, 22 crops. Downloaded from the author's GitHub mirror `Gladiator07/Harvestify` | public dataset, unchanged |
| air_temp_c, humidity_pct, ph, label | taken from the public dataset (rounded to sensor resolution) | public |
| soil_type | sampled per crop from standard Indian crop–soil associations (NCERT Class X Geography) | **team assumption** |
| moisture_pct | `WP + (FC − WP)·(1 − e^(−rainfall/120))` + noise, FC/WP from FAO-56 Table 19 for the soil texture | **derived** |
| soil_temp_c | `air_temp + 1.5 − 0.08·(moisture − 20)` + noise | **derived** |
| N, P, K, rainfall | dropped (not sensed); rainfall kept only as a reference column | – |
| Database demo data (farmers, plots, cycles, logs) | `database/generate_seed.py` | **synthetic demo data** |

Neither the processed dataset nor the demo database is field-measured data, and the report should say so. Readings from your real NODE-01 add genuine measurements alongside the demo data.

## 5. Models (Experiment 2)

Nine classifiers (the eight baselines plus a Soft Voting ensemble of RF, HGB and SVM) were compared with stratified 5-fold CV on two feature sets: moisture, soil temperature, air temperature, humidity, soil type, with or without pH.

| Feature set | Selected model | CV accuracy | CV macro-F1 | Hold-out top-3 accuracy |
|---|---|---|---|---|
| with pH | Soft Voting RF+HGB+SVM (+ isotonic calibration) | 0.895 | 0.894 | 0.993 |
| without pH | Random Forest (+ isotonic calibration) | 0.831 | 0.829 | 0.986 |

The other models compared were Logistic Regression, Gaussian Naive Bayes, KNN, Decision Tree, SVM and Extra Trees. Every model's metrics, classification report and confusion matrix are saved in `experiments/exp2_model_comparison/per_model/`.

Because top-1 accuracy without NPK is about 83–89% but top-3 accuracy is about 99%, the website always shows a ranked **top-3**.

## 6. Novelty (details and numbers in `docs/NOVELTY.md`)

| # | Contribution | Evidence |
|---|---|---|
| N1 | **Quality-Weighted Temporal Consensus**: classify a 7-day, quality-weighted consensus of readings instead of a single snapshot | Exp 3: 0.45–0.51 → 0.81–0.87 top-1 accuracy under 0–30% sensor faults |
| N2 | **Adaptive model selection** when pH is missing | Exp 4: 0.814 vs 0.707–0.759 for imputing pH |
| N3 | **Calibrated, abstention-aware output** (Isolation Forest + confidence threshold) | Exp 5: flags 70% of implausible inputs vs 0.45% of real ones |
| N4 | **Virtual Nitrogen Balance Index**: estimate soil N from DB history (no NPK sensor) and re-rank by season × rotation × nitrogen | Exp 6 case study; same logic in SQL (`fn_virtual_n_index`) and Python |
| N5 | **Explainable fit card**: each measurement vs the crop's learned P5–P95 envelope, also stored in `crops` | shown on the Recommend page |

## 7. Database objects (for the DBMS rubric)

- **Tables (9):** farmers, plots, crops, sensor_nodes, soil_health_logs, cultivation_cycles, treatments, recommendation_runs, recommendation_items.
- **Views (8):** v_latest_soil_log, v_latest_ph, v_plot_recent_conditions, v_last_crop, **v_recommended_crops**, v_cycle_performance, v_plot_overview, v_node_health.
- **Triggers:**
  - `trg_cycle_rotation_check` blocks an annual crop following itself, two GROWING cycles on one plot, and overlapping dates.
  - `trg_soil_log_before_insert` nulls DS18B20 fault codes and forces plot_id from the node.
  - `trg_soil_log_after_insert` updates the node's last-seen time.
- **Procedures:** **sp_yield_report(crop, year)**, sp_plot_history, sp_start_cycle, sp_record_harvest (transaction + rollback).
- **Function:** fn_virtual_n_index(plot).
- **Constraints:** PK, FK with ON DELETE rules, UNIQUE, and CHECK constraints (ranges, phone format, status/date consistency).

Things to try in the demo:
- **Cultivation cycles:** start the same crop twice on a plot to see the trigger message.
- **Recommend a crop:** pick plot #4 (no pH) to see the sensor-only model; pick a month outside the crop's season to see the season penalty.
- **Yield reports:** this page calls the stored procedure.

## 8. API

| Method | URL | Purpose |
|---|---|---|
| POST | `/api/readings` | JSON reading from a node, e.g. `{"node_id":"NODE-01","moisture_pct":21.4,"soil_temp_c":27.3,"air_temp_c":29,"humidity_pct":71}` |
| GET | `/api/plots/<id>/readings?days=7` | time series |
| GET | `/api/plots/<id>/recommend?month=11` | recommendation as JSON (not saved) |

## 9. References used in the code

1. A. Ingle, "Crop Recommendation Dataset," Kaggle, 2020. https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
2. R. G. Allen, L. S. Pereira, D. Raes, M. Smith, *Crop Evapotranspiration*, FAO Irrigation and Drainage Paper 56, FAO, Rome, 1998 (Table 19, soil water characteristics).
3. NCERT, *Contemporary India II*, Class X Geography, Ch. 1 "Resources and Development" and Ch. 4 "Agriculture."
4. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *JMLR*, vol. 12, pp. 2825–2830, 2011.
5. F. T. Liu, K. M. Ting, Z.-H. Zhou, "Isolation Forest," in *Proc. IEEE ICDM*, 2008, pp. 413–422.
6. B. Zadrozny, C. Elkan, "Transforming classifier scores into accurate multiclass probability estimates," in *Proc. ACM SIGKDD*, 2002, pp. 694–699.
7. Maxim Integrated, *DS18B20 Programmable Resolution 1-Wire Digital Thermometer* datasheet (85 °C power-on reset value); DallasTemperature library (`DEVICE_DISCONNECTED_C = -127`).
8. Aosong, *DHT11 Humidity & Temperature Sensor* datasheet (±2 °C, ±5 %RH).
9. Oracle, *MySQL 8.0 Reference Manual* (CHECK constraints, triggers, stored programs, window functions).

Before the final report, verify the agronomic constants in `ml/crop_knowledge.py` (seasons, maturation days, typical yields) and the fertiliser prices in `database/generate_seed.py` against ICAR / Government of India sources.
