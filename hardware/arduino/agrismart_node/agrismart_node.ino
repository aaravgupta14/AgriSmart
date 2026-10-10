/*
  AgriSmart soil node  -  Arduino Uno R3 (breadboard build, no soldering)
  ----------------------------------------------------------------------
  Sensors (all common hobby modules):
    * Capacitive soil moisture sensor v1.2/v2.0  -> A0   (VCC 5V, GND)
    * DHT11 temperature + humidity module        -> D2   (VCC 5V, GND)
    * DS18B20 waterproof temperature probe       -> D4   (4.7k pull-up D4 -> 5V)
    * OPTIONAL PH-4502C pH module (PO pin)       -> A1   (set USE_PH to 1)

  Output: one CSV line every SAMPLE_MS on USB serial @ 9600 baud
      node_id,moisture_pct,soil_temp_c,air_temp_c,humidity_pct,ph
  e.g. NODE-01,21.4,27.31,29.0,71.0,nan
  Lines starting with '#' are comments and are ignored by the bridge.

  Libraries (Arduino IDE -> Library Manager):
    "DHT sensor library" by Adafruit (+ "Adafruit Unified Sensor")
    "OneWire" by Paul Stoffregen
    "DallasTemperature" by Miles Burton
*/
#include <DHT.h>
#include <OneWire.h>
#include <DallasTemperature.h>

// ------------------------------------------------------------- settings
const char* NODE_ID      = "NODE-01";     // must exist in sensor_nodes table
const unsigned long SAMPLE_MS = 120000UL; // 2 minutes (use 10000 for testing)
#define USE_PH 0                          // 1 if a PH-4502C module is connected

// Two-point calibration of the capacitive sensor (see hardware/CALIBRATION.md)
//   RAW_DRY : reading in oven/sun-dried soil from your field  -> 0 % water
//   RAW_SAT : reading in the same soil fully saturated        -> SAT_VWC %
const int   RAW_DRY = 520;
const int   RAW_SAT = 260;
const float SAT_VWC = 45.0;               // typical porosity of loam (vol. %)

// pH calibration (PH-4502C): voltage at pH 7 buffer and slope (V per pH unit)
const float PH7_VOLT  = 2.50;
const float PH_SLOPE  = -0.18;

// ------------------------------------------------------------- pins
#define PIN_MOIST A0
#define PIN_PH    A1
#define PIN_DHT   2
#define PIN_DS    4

DHT dht(PIN_DHT, DHT11);
OneWire oneWire(PIN_DS);
DallasTemperature ds(&oneWire);

int readAvg(int pin, int n) {             // simple averaging to reduce ADC noise
  long s = 0;
  for (int i = 0; i < n; i++) { s += analogRead(pin); delay(20); }
  return (int)(s / n);
}

float readMoisturePct() {
  int raw = readAvg(PIN_MOIST, 10);
  float vwc = (float)(RAW_DRY - raw) / (RAW_DRY - RAW_SAT) * SAT_VWC;
  if (vwc < 0) vwc = 0;
  if (vwc > 60) vwc = 60;
  return vwc;
}

float readPH() {
#if USE_PH
  float v = readAvg(PIN_PH, 20) * 5.0 / 1023.0;
  return 7.0 + (v - PH7_VOLT) / PH_SLOPE;
#else
  return NAN;
#endif
}

void printVal(float v, int digits) {
  if (isnan(v)) Serial.print("nan"); else Serial.print(v, digits);
}

void setup() {
  Serial.begin(9600);
  dht.begin();
  ds.begin();
  ds.setResolution(12);                   // 0.0625 degC
  Serial.println(F("# AgriSmart node started"));
  Serial.println(F("# node_id,moisture_pct,soil_temp_c,air_temp_c,humidity_pct,ph"));
}

void loop() {
  ds.requestTemperatures();
  float soilT = ds.getTempCByIndex(0);    // -127 if probe disconnected
  float airT  = dht.readTemperature();    // NAN on read failure
  float hum   = dht.readHumidity();
  float moist = readMoisturePct();
  float ph    = readPH();

  Serial.print(NODE_ID);  Serial.print(',');
  printVal(moist, 1);     Serial.print(',');
  printVal(soilT, 2);     Serial.print(',');
  printVal(airT, 1);      Serial.print(',');
  printVal(hum, 1);       Serial.print(',');
  printVal(ph, 2);        Serial.println();

  delay(SAMPLE_MS);
}
