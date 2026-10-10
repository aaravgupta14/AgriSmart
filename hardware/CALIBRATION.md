# Hardware: wiring and calibration

## Parts (all breadboard-friendly)

| Part | Connects to | Approx. price |
|---|---|---|
| Arduino Uno R3 + USB cable | laptop USB | ₹700 |
| Capacitive soil moisture sensor v1.2 / v2.0 | AOUT → A0, VCC → 5V, GND | ₹100–150 |
| DHT11 temperature-humidity module (3-pin) | DATA → D2, VCC → 5V, GND | ₹100–120 |
| DS18B20 waterproof probe + 4.7 kΩ resistor | DQ → D4, 4.7 kΩ from D4 to 5V | ₹150–160 |
| Breadboard + jumper wires | | ₹150 |
| *Optional:* PH-4502C pH module + probe | PO → A1, VCC → 5V, GND | ₹900–1,200 |

No NPK sensor is needed. pH is optional: you can type it in from a Soil Health Card or a pH test kit on the **Soil readings** page instead.
Prices are rough market estimates; check them before putting them in the report.

Use the **capacitive** moisture sensor, not the resistive FC-28 fork, because the resistive probe corrodes within weeks.

## Soil-moisture calibration (two-point)
1. Take soil from the field and dry it in the sun (or an oven at about 105 °C). Put the probe in it and note the number from the Serial Monitor. Calibration sketch tip: temporarily print `analogRead(A0)`. Write this into `RAW_DRY`.
2. Saturate the same soil with water until it glistens, insert the probe to the same depth, and note the reading in `RAW_SAT`.
3. `SAT_VWC` is the soil's water content when saturated (≈ its porosity). Use 45 for loam/alluvial, 50 for black clay, 40 for red/sandy loam, and 38 for sand.
4. The sketch maps linearly between the two points, giving an approximate **volumetric water content %**, which is the same unit as the training data.

## pH (optional PH-4502C)
Measure soil pH the lab way: mix 1 part soil with 2.5 parts distilled water, stir, wait 30 min, and dip the probe in the liquid. This is the standard 1 : 2.5 soil-water method. Calibrate with pH 7 and pH 4 buffer sachets, then set `PH7_VOLT` and `PH_SLOPE` in the sketch. Set `USE_PH 1`.

## Data path
`agrismart_node.ino` prints `NODE-01,21.4,27.31,29.0,71.0,nan` every 2 min, and `hardware/bridge/serial_bridge.py` validates the line and INSERTs it into `soil_health_logs`. The trigger `trg_soil_log_before_insert` nulls DS18B20 fault codes (-127 / 85) and `trg_soil_log_after_insert` updates `sensor_nodes.last_seen_ts`.

Find the COM port with `python hardware/bridge/serial_bridge.py --list-ports`. Close the Arduino IDE Serial Monitor first, because only one program can hold the port.
