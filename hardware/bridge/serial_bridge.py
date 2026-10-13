"""
AgriSmart serial bridge  -  Arduino (USB serial) -> MySQL soil_health_logs
=========================================================================
Reads the CSV lines printed by agrismart_node.ino, validates them
(edge validation) and INSERTs them with a parameterised query.

    python hardware/bridge/serial_bridge.py                 # port from .env (SERIAL_PORT)
    python hardware/bridge/serial_bridge.py --port COM4
    python hardware/bridge/serial_bridge.py --simulate      # no hardware: fake node readings
    python hardware/bridge/serial_bridge.py --list-ports

Rules applied before insert:
  * column order comes from the sketch's '# node_id,...' header line; without one, a
    6-field (full node) or 4-field (no DS18B20 / pH) frame is accepted
  * node_id must be known and active; a bare number from the sketch ('1') means NODE-01
  * 'nan' -> NULL (DHT11 read failure / pH module absent)
  * values outside physical range -> NULL (the row is still kept if any value is valid)
  * DS18B20 -127 / 85 are passed through: the DB trigger nulls and annotates them,
    so the fault stays visible in the data for the quality score
"""
import argparse
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import pymysql

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "ml"))
import config                               # noqa: E402
from common import SENSOR_RANGES            # noqa: E402

FIELDS = ["node_id", "moisture_pct", "soil_temp_c", "air_temp_c", "humidity_pct", "ph"]
# frame layouts by field count, used until the sketch's "# node_id,..." header line is seen
LAYOUTS = {6: FIELDS,
           4: ["node_id", "moisture_pct", "air_temp_c", "humidity_pct"]}   # node without DS18B20 / pH


def header_columns(line: str):
    """Column order from a sketch header such as '# node_id,moisture_pct,air_temp_c,humidity_pct'."""
    cols = [c.strip() for c in line.lstrip("#").split(",")]
    if cols and cols[0] == "node_id" and len(cols) > 1 and all(c in FIELDS for c in cols):
        return cols
    return None


def normalise_node_id(raw: str) -> str:
    """A bare number from the sketch (e.g. '1') means NODE-01."""
    return f"NODE-{int(raw):02d}" if raw.isdigit() else raw


def parse(line: str, columns=None):
    line = line.strip()
    if not line or line.startswith("#"):
        return None, "comment"
    parts = [p.strip() for p in line.split(",")]
    cols = columns or LAYOUTS.get(len(parts))
    if cols is None or len(parts) != len(cols):
        return None, f"malformed frame ({len(parts)} fields)"
    rec = dict.fromkeys(FIELDS[1:])
    rec["node_id"] = normalise_node_id(parts[0])
    for k, raw in zip(cols[1:], parts[1:]):
        try:
            v = float(raw)
        except ValueError:
            return None, f"non-numeric {k}={raw!r}"
        if math.isnan(v):
            rec[k] = None
        elif k == "soil_temp_c" and v in (-127.0, 85.0):
            rec[k] = v                                   # trigger handles it
        else:
            lo, hi = SENSOR_RANGES[k]
            rec[k] = v if lo <= v <= hi else None
    if all(rec[k] is None for k in FIELDS[1:]):
        return None, "no valid values"
    return rec, "ok"


def insert(con, rec):
    with con.cursor() as cur:
        cur.execute("SELECT plot_id FROM sensor_nodes WHERE node_id=%s AND is_active", (rec["node_id"],))
        row = cur.fetchone()
        if not row:
            return f"unknown node {rec['node_id']}"
        cur.execute("""INSERT INTO soil_health_logs
                       (plot_id, node_id, log_ts, source, moisture_pct, soil_temp_c, air_temp_c, humidity_pct, ph_value)
                       VALUES (%s,%s,NOW(),'SENSOR',%s,%s,%s,%s,%s)""",
                    (row[0], rec["node_id"], rec["moisture_pct"], rec["soil_temp_c"],
                     rec["air_temp_c"], rec["humidity_pct"], rec["ph"]))
    con.commit()
    return "stored"


def simulated_lines(node_id, interval):
    """Fake Arduino output for demos without hardware."""
    base = {"m": random.uniform(14, 28), "s": random.uniform(22, 30), "a": random.uniform(22, 32), "h": random.uniform(55, 85)}
    while True:
        hr = datetime.now().hour + datetime.now().minute / 60
        d = math.sin(2 * math.pi * (hr - 9) / 24)
        st = base["s"] + 1.5 * d + random.gauss(0, .25)
        if random.random() < 0.02:
            st = -127.0
        yield (f"{node_id},{base['m'] + random.gauss(0, 1):.1f},{st:.2f},"
               f"{round(base['a'] + 4 * d + random.gauss(0, 1)):.1f},{round(base['h'] - 8 * d + random.gauss(0, 2.5)):.1f},nan")
        time.sleep(interval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=config.SERIAL_PORT)
    ap.add_argument("--baud", type=int, default=config.SERIAL_BAUD)
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--node", default="NODE-01", help="node id for --simulate")
    ap.add_argument("--interval", type=float, default=10, help="seconds between simulated readings")
    ap.add_argument("--list-ports", action="store_true")
    a = ap.parse_args()

    if a.list_ports:
        from serial.tools import list_ports
        for p in list_ports.comports():
            print(p.device, "-", p.description)
        return

    con = pymysql.connect(**config.DB, charset="utf8mb4")
    if a.simulate:
        src = simulated_lines(a.node, a.interval)
        print(f"Simulating {a.node} every {a.interval}s - Ctrl+C to stop")
    else:
        import serial
        ser = serial.Serial(a.port, a.baud, timeout=5)
        time.sleep(2)                                    # Uno resets when the port opens
        print(f"Listening on {a.port} @ {a.baud} - Ctrl+C to stop")
        src = (ser.readline().decode("ascii", "ignore") for _ in iter(int, 1))
    ok = bad = 0
    columns = None
    last_reject, repeats = None, 0                       # print a burst of identical rejects once
    try:
        for line in src:
            if not line:
                continue
            if line.lstrip().startswith("#") and header_columns(line):
                columns = header_columns(line)
                print(f"frame layout from sketch header: {','.join(columns)}")
                continue
            rec, why = parse(line, columns)
            if rec is None:
                if why != "comment":
                    bad += 1
                    if line.strip() == last_reject:
                        repeats += 1
                    else:
                        if repeats:
                            print(f"   ... same line rejected {repeats} more times")
                        last_reject, repeats = line.strip(), 0
                        print(f"[{datetime.now():%H:%M:%S}] rejected: {why}: {line.strip()[:120]}")
                continue
            if repeats:
                print(f"   ... same line rejected {repeats} more times")
            last_reject, repeats = None, 0
            res = insert(con, rec)
            ok += res == "stored"
            print(f"[{datetime.now():%H:%M:%S}] {res}: {line.strip()}   (stored={ok}, rejected={bad})")
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        con.close()


if __name__ == "__main__":
    main()
