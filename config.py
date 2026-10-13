"""Central configuration. Values come from environment variables or a .env file
in the project root (copy .env.example to .env and edit)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_env = ROOT / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

DB = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", "agrismart"),
    "password": os.getenv("DB_PASSWORD", "agrismart123"),
    "database": os.getenv("DB_NAME", "agrismart"),
}
DB_ADMIN_USER = os.getenv("DB_ADMIN_USER", "root")
DB_ADMIN_PASSWORD = os.getenv("DB_ADMIN_PASSWORD", "")

SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "change-me-in-.env")
SERIAL_PORT = os.getenv("SERIAL_PORT", "COM3")        # Windows e.g. COM3, Linux /dev/ttyACM0
SERIAL_BAUD = int(os.getenv("SERIAL_BAUD", "9600"))
# "vwc": the sketch already sends volumetric water content (agrismart_node.ino).
# "relative": the sketch sends 0-100 % (dry air -> wet); the bridge rescales it so 100 % = MOISTURE_SAT_VWC.
# node id used when the sketch sends a bare number such as "1" (empty: "1" -> NODE-01)
SERIAL_NODE_ID = os.getenv("SERIAL_NODE_ID", "")
MOISTURE_SCALE = os.getenv("MOISTURE_SCALE", "vwc")
MOISTURE_SAT_VWC = float(os.getenv("MOISTURE_SAT_VWC", "45"))   # same as SAT_VWC in agrismart_node.ino
RECOMMEND_WINDOW_DAYS = int(os.getenv("RECOMMEND_WINDOW_DAYS", "7"))
