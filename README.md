# AgriSmart: Smart Farm & Crop Recommendation System

**BCSE302L / BCSE302P Database Management Systems, VIT**
Team: Aarav Gupta (25BAI0131), Chirag Saxena (25BCE0223), Suyash Agarwal (25BBS0088)

AgriSmart is an IoT-fed relational database (MySQL 8) that recommends the best crop for a farmer's plot. It uses **low-cost soil sensors with no N-P-K sensor**, plus the plot's own history of readings, crops and fertiliser stored in the database. A Flask website gives farmers plain-language crop advice and lets them record what they grow and harvest.

```
Arduino Uno (soil moisture + DHT11 + DS18B20)
        |  USB serial, one CSV line per reading
        v
hardware/bridge/serial_bridge.py  -- validates, then INSERTs -->  MySQL 8 (9 tables, 8 views,
                                                                   3 triggers, 4 procedures,
                                                                   fn_virtual_n_index)
                                                                        |
Farmer's browser  <-->  Flask website (webapp/)  <-->  ML engine (ml/recommender.py):
                                                       7-day quality-weighted consensus,
                                                       calibrated classifier, abstention,
                                                       season / rotation / nitrogen re-ranking
```

**Contents**
1. [What the system does](#1-what-the-system-does)
2. [Repository structure](#2-repository-structure)
3. [Quick start (Windows)](#3-quick-start-windows)
4. [Configuration (`.env`)](#4-configuration-env)
5. [Hardware and the serial bridge](#5-hardware-and-the-serial-bridge)
6. [Using the website](#6-using-the-website)
7. [Database design](#7-database-design)
8. [Dataset: where every number comes from](#8-dataset-where-every-number-comes-from)
9. [Models](#9-models)
10. [Novelty](#10-novelty)
11. [Experiments and how to reproduce them](#11-experiments-and-how-to-reproduce-them)
12. [API](#12-api)
13. [Limitations](#13-limitations)
14. [Credits and references](#14-credits-and-references)

---

## 1. What the system does

1. A sensor node in the field measures soil moisture, air temperature and humidity, and (when fitted) soil temperature. Soil pH is optional and can be typed in from the farmer's Soil Health Card.
2. The serial bridge checks each reading and stores it in `soil_health_logs`. Database triggers clean sensor fault codes and keep track of when each node was last seen.
3. When the farmer asks for advice, the engine:
   - combines the last 7 days of readings into one quality-weighted consensus, so that one faulty or noisy reading does not change the answer;
   - uses the with-pH model if a valid pH exists, otherwise the sensor-only model;
   - re-ranks the crops by season, crop rotation and an estimate of soil nitrogen taken from the plot's recorded crops and fertiliser;
   - flags low-confidence or unusual inputs and suggests a lab test instead of guessing.
4. The farmer picks a crop ("I'll grow this"). This saves the recommendation and starts a cultivation cycle, and the rotation trigger blocks growing the same annual crop twice in a row. Treatments and the harvest are recorded later, and the yield report is a stored procedure.

## 2. Repository structure

```
AgriSmart/
├── README.md                   this file
├── requirements.txt            Python packages (scikit-learn pinned: saved models depend on it)
├── config.py                   reads settings from .env
├── .env.example                copy to .env and fill in (see section 4)
├── .vscode/tasks.json          VS Code tasks: check system, setup, start website, simulate sensor
├── run/                        Windows helpers used by the VS Code tasks
│   ├── 0_check_system.bat  1_setup.bat  2_start_website.bat  3_simulate_sensor.bat
│   ├── setup_all.py            one-click setup: .venv, packages, model training, database
│   └── *_log.txt               logs of the Exp 7 / Exp 8 / retraining / figure runs
├── data/
│   ├── raw/crop_recommendation.csv              public dataset, unchanged (2,200 rows, 22 crops)
│   └── processed/agrismart_sensor_dataset.csv   sensor-oriented dataset derived from it (section 8)
├── ml/
│   ├── build_dataset.py        raw -> processed dataset (documented derivation)
│   ├── train_models.py         Exp 0-2: compares 9 models on 2 feature sets, calibrates and saves the deployed ones
│   ├── experiments.py          Exp 3-6 (novelty) + writes a local summary, experiments/EXPERIMENTS_SUMMARY.md
│   ├── model_zoo.py            extra models for Exp 7 (add one with one line in EXTRA_MODELS)
│   ├── compare_models.py       Exp 7: extended comparison of 16 models
│   ├── regularization_experiment.py   Exp 8: nested-CV regularisation study
│   ├── make_figures.py         26 numbered report figures + confusion matrices
│   ├── recommender.py          the recommendation engine used by the website
│   ├── crop_knowledge.py       season / family / N demand / maturation / typical yield per crop
│   ├── common.py               feature lists and physical sensor ranges
│   └── models/                 model_meta.json (the *.joblib files are created by train_models.py, not shipped)
├── experiments/                every result used in the report
│   ├── exp0_dataset/ ... exp8_regularization/   result CSVs and charts of each experiment
│   └── figures/                fig01-fig26, confusion_matrices/
├── database/
│   ├── 01_schema.sql           9 tables in 3NF with PK / FK / UNIQUE / CHECK constraints
│   ├── 02_views.sql            8 views, including v_recommended_crops (rule-based SQL recommender)
│   ├── 03_routines.sql         3 triggers, 4 stored procedures, fn_virtual_n_index
│   ├── 04_seed_data.sql        generated demo data
│   └── generate_seed.py        generator for 04_seed_data.sql
├── webapp/
│   ├── app.py                  Flask routes (farmer pages, admin pages, JSON API)
│   ├── farmer.py               turns the engine's output into plain-language advice
│   ├── services.py  db.py      recommendation service and database helpers
│   ├── templates/              f_*.html = farmer pages; the rest = admin pages
│   └── static/css/style.css
├── hardware/
│   ├── arduino/agrismart_node/agrismart_node.ino   reference sketch for the sensor node
│   └── bridge/serial_bridge.py serial -> MySQL (also --simulate without hardware)
└── scripts/
    ├── setup_windows.ps1       PowerShell setup (installs MySQL with winget if missing)
    ├── setup_database.py       creates the database and app user and loads all SQL files
    └── run_pipeline.py         rebuilds dataset, models, Exp 0-6 and seed data from the raw data
```

Not in the repository: `.env` (your passwords), `.venv/`, `ml/models/*.joblib` (about 75 MB, rebuilt by setup), machine-specific setup logs, the project reports, and generated Markdown summaries such as `experiments/EXPERIMENTS_SUMMARY.md` (written locally by `ml/experiments.py`).

## 3. Quick start (Windows)

You need **Python 3.10+** (Anaconda is fine; tested on 3.13) and **MySQL 8**.

### Option A: VS Code tasks (easiest)
Open the folder in VS Code, press `Ctrl+Shift+P` and choose **Tasks: Run Task**:

| Task | What it does |
|---|---|
| `AgriSmart 0: Check system` | lists the installed Python, MySQL (client, service, install folder), winget and VS Code in `run/system_check.txt` |
| `AgriSmart 1: Setup (packages + models + database)` | runs `run/setup_all.py`: creates `.venv`, installs packages, trains the models, builds the database. It asks you to put your MySQL root password in `.env`. The log goes to `run/setup_log.txt` |
| `AgriSmart 2: Start website` | starts the website at **http://127.0.0.1:5000** |
| `AgriSmart 3: Simulate sensor node` | sends simulated NODE-01 readings every 10 s (no hardware needed) |

### Option B: PowerShell
```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
.\.venv\Scripts\python.exe -m webapp.app
```
The script installs MySQL with `winget` if it is missing, creates `.venv`, installs the packages, trains the models if they are absent, writes `.env` (asking for your MySQL root password) and builds the database.

### Option C: manual
```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env                       # then set DB_ADMIN_PASSWORD to your MySQL root password
cd ml && ..\.venv\Scripts\python.exe train_models.py && cd ..
.venv\Scripts\python.exe scripts\setup_database.py
.venv\Scripts\python.exe -m webapp.app       # open http://127.0.0.1:5000
```

The first crop advice after the website starts is slower because the models are loaded then. Chart.js and the web fonts load from a CDN, so charts need an internet connection.

## 4. Configuration (`.env`)

| Key | Meaning |
|---|---|
| `DB_HOST`, `DB_PORT`, `DB_NAME` | MySQL server and database (default `localhost`, `3306`, `agrismart`) |
| `DB_USER`, `DB_PASSWORD` | application user, created by `setup_database.py` |
| `DB_ADMIN_USER`, `DB_ADMIN_PASSWORD` | MySQL root account, used only by `setup_database.py` |
| `FLASK_SECRET_KEY` | session key for the website; set your own random string |
| `SERIAL_PORT`, `SERIAL_BAUD` | Arduino port and speed (Windows `COM3`, Linux `/dev/ttyACM0`; 9600) |
| `SERIAL_NODE_ID` | node id to use when the sketch sends a bare number such as `1` (empty: `1` -> `NODE-01`) |
| `MOISTURE_SCALE` | `vwc` if the sketch sends volumetric water content (the reference sketch); `relative` if it sends a 0-100 % dry-to-wet scale |
| `MOISTURE_SAT_VWC` | with `relative`, the water content that 100 % maps to (default 45 %) |
| `RECOMMEND_WINDOW_DAYS` | length of the consensus window (default 7 days) |

## 5. Hardware and the serial bridge

| Sensor | Measures | Pin |
|---|---|---|
| Capacitive soil-moisture sensor | soil volumetric water content (%) | A0 |
| DHT11 | air temperature, relative humidity | D2 |
| DS18B20 waterproof probe (+4.7 kΩ pull-up to 5 V) | soil temperature | D4 |
| *optional* PH-4502C | soil pH (1:2.5 soil-water slurry) | A1 |

There is **no N-P-K sensor**. Use the capacitive moisture sensor, not the resistive fork, which corrodes.

**Moisture calibration (two-point):** with `PRINT_RAW 1`, note the raw reading in sun- or oven-dried field soil (`RAW_DRY`) and in the same soil saturated with water (`RAW_SAT`). Set `SAT_VWC` to the soil's saturated water content (about 45 for loam/alluvial, 50 for black clay, 40 for red/sandy loam, 38 for sand). The sketch maps linearly between the two points to volumetric water content, the unit of the training data.

**pH (optional PH-4502C):** mix 1 part soil with 2.5 parts distilled water, wait 30 min, and measure the liquid. Calibrate with pH 7 and pH 4 buffers, set `PH7_VOLT` and `PH_SLOPE`, and set `USE_PH 1`.

**Reference sketch** (`agrismart_node.ino`). It prints a header line, `# moisture=vwc`, and then one CSV line per sample (`node_id,moisture_pct,soil_temp_c,air_temp_c,humidity_pct,ph`, with `nan` for a missing value). Settings at the top of the file:
- `SAMPLE_MS` (2 min);
- `USE_DS18B20` (0 until the soil probe is wired; soil temperature is then `nan`);
- `USE_PH`;
- `PRINT_RAW`, which prints the raw moisture reading for calibration;
- the calibration constants `RAW_DRY`, `RAW_SAT` and `SAT_VWC`.

**Serial bridge** (`hardware/bridge/serial_bridge.py`):
```bash
python hardware/bridge/serial_bridge.py --list-ports        # find the Arduino
python hardware/bridge/serial_bridge.py --port COM3         # real node
python hardware/bridge/serial_bridge.py --simulate --interval 10   # no hardware
```
- **Frame format:** the bridge reads the column order from the sketch's `# node_id,...` header. Without a header it accepts 6-field frames (full node) or 4-field frames (`node_id,moisture,air_temp,humidity`).
- **Node id:** a bare node number from the sketch is stored as `SERIAL_NODE_ID` (or `NODE-01` if that is empty). The node must exist and be active in `sensor_nodes`.
- **Missing and out-of-range values:** `nan` is stored as NULL. Values outside the physical range in `ml/common.py` are also stored as NULL: moisture 0-60 %, soil temperature -5-60 °C, air temperature 0-50 °C, humidity 5-100 %, pH 3-10.
- **DS18B20 fault codes:** -127 °C and 85 °C are passed through, and the trigger `trg_soil_log_before_insert` nulls them.
- **Moisture scale:** with `MOISTURE_SCALE=relative`, a 0-100 % moisture value is rescaled to water content (100 % -> `MOISTURE_SAT_VWC`). A `# moisture=vwc` line from the sketch, or `--simulate`, turns the rescaling off.
- **Log output:** each stored line also shows the converted moisture and any field that was stored as NULL.
- **Reconnecting:** if the USB cable is unplugged, restart the bridge.

**Tested on 13 Oct 2026:** an Arduino Uno on COM3 running the team's own 4-field sketch, with a moisture sensor and a DHT air sensor; the DS18B20 was not fitted yet. Readings arrived about every 10 s and were stored and shown on the website. With the probe in air, moisture read 0 %. In freshly watered soil it read 96-100 % on the module's relative scale, which was stored as 43-45 %.

## 6. Using the website

**Farmer pages** (top of the menu):
1. **My field:** live readings, sensor status and the last 7 days.
2. **Add soil data** (optional): type in a reading or a pH value from the Soil Health Card.
3. **Crop advice:** the top three crops in plain language. For each crop it shows how well your soil fits, the season, time to harvest, typical yield and fertiliser need. "I'll grow this" saves the run and starts a cultivation cycle (`sp_start_cycle`).
4. **My crops & harvest:** treatments and the harvest of each cycle.

**+ New account / field** (`/register`) creates the farmer, the field (including pH from the Soil Health Card), the sensor node and the sensors fitted to it (or manual entry only), an optional last crop and an optional first reading, all in **one transaction**.

**Admin & project pages:**
- Overview, Farmers, Fields & sensors, All soil readings, All cultivation cycles;
- Yield reports (calls the stored procedure `sp_yield_report`);
- Technical recommendation (the full fit card and scores);
- Crop catalogue, and Model & experiments.

Things to try in a demo:
- start the same crop twice on a plot to see the rotation trigger refuse it;
- recommend for plot #4, which has no pH, to see the sensor-only model;
- pick a sowing month outside the crop's season to see the season penalty;
- open Yield reports to run the stored procedure.

## 7. Database design

- **Tables (9, 3NF):** farmers, plots, crops, sensor_nodes, soil_health_logs, cultivation_cycles, treatments, recommendation_runs, recommendation_items.
- **Views (8):** v_latest_soil_log, v_latest_ph, v_plot_recent_conditions, v_last_crop, **v_recommended_crops** (a rule-based SQL recommender that uses the crops' learned ranges), v_cycle_performance, v_plot_overview, v_node_health.
- **Triggers (3):**
  - `trg_cycle_rotation_check` blocks an annual crop following itself, two GROWING cycles on one plot, and overlapping dates;
  - `trg_soil_log_before_insert` nulls DS18B20 fault codes and forces plot_id from the node;
  - `trg_soil_log_after_insert` updates the node's last-seen time.
- **Stored procedures (4):** **sp_yield_report(crop, year)**, sp_plot_history, sp_start_cycle, sp_record_harvest (transaction with rollback).
- **Function:** `fn_virtual_n_index(plot)`, the nitrogen estimate described in section 10, N4.
- **Constraints:** primary keys, foreign keys with ON DELETE rules, UNIQUE, and CHECK constraints (value ranges, phone format, status/date consistency).
- **Demo data** (`database/generate_seed.py`): 10 farmers, 16 plots, 22 crops, 6 sensor nodes, 2,942 soil logs (10 days, one reading per 30 min per node), 62 cultivation cycles and 151 treatments. This is **synthetic** demo data.

## 8. Dataset: where every number comes from

| Item | Source | Real or derived |
|---|---|---|
| `data/raw/crop_recommendation.csv` | *Crop Recommendation Dataset*, A. Ingle, Kaggle (2020): 2,200 rows, 22 crops, 100 per crop. Downloaded from the author's GitHub mirror `Gladiator07/Harvestify` | public dataset, unchanged |
| air_temp_c, humidity_pct, ph, label | taken from the public dataset (rounded to sensor resolution) | public |
| soil_type | sampled per crop from Indian crop-soil associations (NCERT Class X Geography) | **team assumption** |
| moisture_pct | `WP + (FC − WP)·(1 − e^(−rainfall/120))` + noise, with FC/WP from FAO-56 Table 19 for the soil texture | **derived** |
| soil_temp_c | `air_temp + 1.5 − 0.08·(moisture − 20)` + noise | **derived** |
| N, P, K, rainfall | dropped (not sensed) | – |
| database demo data | `database/generate_seed.py` | **synthetic** |

All experiments use a stratified 80/20 split with seed 42 (440 test rows) and stratified 5-fold cross-validation. Neither the processed dataset nor the demo database is field-measured.

## 9. Models

Features: moisture, soil temperature, air temperature, humidity and soil type, with or without pH.

**Exp 2:** nine classifiers, 5-fold CV accuracy.

| Model | With pH | Without pH |
|---|---|---|
| **Soft Voting (RF + HGB + SVM)** | **0.895** | 0.833 |
| Hist Gradient Boosting | 0.888 | 0.818 |
| **Random Forest** | 0.878 | **0.831** |
| SVM (RBF) | 0.872 | 0.819 |
| Extra Trees | 0.870 | 0.817 |
| Decision Tree | 0.820 | 0.772 |
| K-Nearest Neighbours | 0.811 | 0.795 |
| Logistic Regression | 0.732 | 0.671 |
| Gaussian Naive Bayes | 0.595 | 0.546 |

**Deployed models.** These are fixed in `DEPLOY` in `ml/train_models.py` after Exp 7, and both are isotonic-calibrated:

| Feature set | Model | CV accuracy | Hold-out accuracy | Hold-out top-3 | ECE before → after |
|---|---|---|---|---|---|
| with pH | Soft Voting (RF+HGB+SVM) | 0.895 | 0.902 | 0.993 | 0.086 → 0.022 |
| without pH | Random Forest | 0.831 | 0.814 | 0.986 | 0.074 → 0.051 |

- **Exp 7 (16 models, `ml/compare_models.py`):**
  - With pH, Soft Voting had the highest CV accuracy and beat Hist Gradient Boosting in 4 of 5 folds; Stacking reached 0.892.
  - Without pH, Stacking (0.834) and Soft Voting (0.833) beat the Random Forest in only 2 of 5 folds, so the simpler Random Forest was kept.
- **Exp 8 (nested-CV regularisation, `ml/regularization_experiment.py`):**
  - Logistic Regression improved by +0.064 with pH and +0.054 without (5 of 5 folds).
  - Without pH, Hist Gradient Boosting improved by +0.011 and SVM by +0.010 (5 of 5 folds).
  - The Random Forest changed by at most 0.003 and Hist Gradient Boosting with pH dropped by 0.005, so the deployment did not change.
- **Permutation importance (with pH, hold-out):** humidity 0.639, moisture 0.247, pH 0.200, soil type 0.148, air temperature 0.113, soil temperature 0.060.

Top-1 accuracy without N-P-K is 0.81-0.90, but top-3 accuracy is about 0.99, so the website always shows a ranked **top three**.

## 10. Novelty

All numbers come from the result files in `experiments/` (seed 42).

| # | Contribution | Evidence |
|---|---|---|
| N1 | **Quality-Weighted Temporal Consensus (QW-TC):** classify a 7-day, quality-weighted, day-blocked consensus of readings instead of the latest reading | Exp 3: top-1 accuracy 0.857-0.902 vs 0.539-0.614 for the latest reading, under 0-30 % simulated sensor faults |
| N2 | **Adaptive model selection** when pH is missing | Exp 4: 0.814, vs 0.791 (pH = training median) and 0.677 (pH = 7.0) |
| N3 | **Calibrated, abstention-aware output** (isotonic calibration, Isolation Forest, confidence threshold 0.50) | Exp 5: flags 70.1 % of random in-range inputs vs 0.45 % of real ones; the threshold keeps 97.0 % of answers at 91.8 % accuracy |
| N4 | **Virtual Nitrogen Balance Index (VNBI):** estimate soil N from the recorded crops and fertiliser (no N-P-K sensor) and re-rank by season × rotation × nitrogen | Exp 6 case study; same formula in SQL (`fn_virtual_n_index`) and Python |
| N5 | **Explainable fit card:** each measurement compared with the crop's learned P5-P95 range, stored in `crops` | shown on the advice pages |

**How QW-TC works** (`ml/recommender.py`):
- Each reading gets a quality score:
  - 0 for a missing, out-of-range or fault-code value;
  - reduced for a spike against the window median;
  - ×0.1 for moisture under 1 % (probe out of soil);
  - ×0.5 when soil and air temperature differ by more than 15 °C.
- Each reading also gets a day-blocked recency weight (τ = 7 days), and the engine takes the weighted mean of the features.
- If no reading in the window has a soil temperature (no DS18B20 fitted), it is estimated with the same formula used to build the training data, and the advice says so.

## 11. Experiments and how to reproduce them

| Exp | Question | Output folder |
|---|---|---|
| 0 | dataset statistics | `experiments/exp0_dataset/` |
| 1 | cost of removing the N-P-K sensor: Random Forest accuracy falls from 0.996 to 0.742 without N, P, K and rainfall | `exp1_npk_ablation/` |
| 2 | model comparison, calibration, feature importance | `exp2_model_comparison/` |
| 3 | QW-TC vs snapshot under sensor faults (N1) | `exp3_temporal_consensus/` |
| 4 | missing pH (N2) | `exp4_adaptive_model_selection/` |
| 5 | abstention and out-of-distribution test (N3) | `exp5_abstention/` |
| 6 | context-aware re-ranking case study (N4) | `exp6_context_reranking/` |
| 7 | extended comparison of 16 models | `exp7_extended_model_comparison/` |
| 8 | regularisation (nested CV) | `exp8_regularization/` |
| – | 26 report figures + confusion matrices | `experiments/figures/` |

All runs use seed 42. To rebuild everything from the raw dataset:
```bash
python scripts/run_pipeline.py              # dataset, models, Exp 0-6, seed data
python ml/compare_models.py                 # Exp 7
python ml/regularization_experiment.py      # Exp 8
python ml/make_figures.py                   # figures
python scripts/setup_database.py            # reload the database
```

## 12. API

| Method | URL | Purpose |
|---|---|---|
| POST | `/api/readings` | JSON reading from a node, e.g. `{"node_id":"NODE-01","moisture_pct":21.4,"soil_temp_c":27.3,"air_temp_c":29,"humidity_pct":71}` |
| GET | `/api/plots/<id>/readings?days=7` | time series of a plot |
| GET | `/api/plots/<id>/recommend?month=11` | recommendation as JSON (not saved) |

## 13. Limitations

- Moisture, soil temperature and soil type in the training data are **derived** with documented formulas, not measured in the field. The collection method of the public dataset is not documented in detail by its author.
- The Exp 3 sensor windows are simulated around dataset rows.
- The VNBI weights and re-ranking factors are heuristic and were evaluated by a case study, not learned from yield data.
- The agronomic constants in `ml/crop_knowledge.py` (seasons, maturation days, typical yields) and the fertiliser prices in `database/generate_seed.py` still need to be checked against ICAR / Government of India sources.
- The hardware test so far used one node without the DS18B20 soil-temperature probe, so soil temperature was estimated.

## 14. Credits and references

The models are trained on the **Crop Recommendation Dataset** by **Atharva Ingle** (Kaggle, 2020). All credit for the original data goes to its author; please cite it if you reuse it and check the Kaggle page for its licence terms.

References used in the code:
1. A. Ingle, "Crop Recommendation Dataset," Kaggle, 2020. https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
2. R. G. Allen, L. S. Pereira, D. Raes, M. Smith, *Crop Evapotranspiration*, FAO Irrigation and Drainage Paper 56, FAO, Rome, 1998 (Table 19, soil water characteristics).
3. NCERT, *Contemporary India II*, Class X Geography, Ch. 1 "Resources and Development" and Ch. 4 "Agriculture."
4. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *JMLR*, vol. 12, pp. 2825–2830, 2011.
5. F. T. Liu, K. M. Ting, Z.-H. Zhou, "Isolation Forest," in *Proc. IEEE ICDM*, 2008, pp. 413–422.
6. B. Zadrozny, C. Elkan, "Transforming classifier scores into accurate multiclass probability estimates," in *Proc. ACM SIGKDD*, 2002, pp. 694–699.
7. Maxim Integrated, *DS18B20 Programmable Resolution 1-Wire Digital Thermometer* datasheet (85 °C power-on reset value); DallasTemperature library (`DEVICE_DISCONNECTED_C = -127`).
8. Aosong, *DHT11 Humidity & Temperature Sensor* datasheet (±2 °C, ±5 %RH).
9. Oracle, *MySQL 8.0 Reference Manual* (CHECK constraints, triggers, stored programs, window functions).
