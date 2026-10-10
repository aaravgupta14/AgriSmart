"""
Creates the AgriSmart database, the application user and loads every SQL file
in database/ in order (01_schema, 02_views, 03_routines, 04_seed_data).
Works without the mysql command-line client (handles DELIMITER blocks).

Usage:
    python scripts/setup_database.py            # schema + demo data
    python scripts/setup_database.py --no-seed  # schema only (empty database)
"""
import argparse
import re
import sys
from pathlib import Path

import pymysql

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402


def split_sql(text: str):
    """Split a .sql file into statements, honouring DELIMITER changes."""
    delim, buf, out = ";", [], []
    for line in text.splitlines():
        s = line.strip()
        m = re.match(r"(?i)^DELIMITER\s+(\S+)", s)
        if m:
            delim = m.group(1); continue
        if not buf and (not s or s.startswith("--")):
            continue
        buf.append(line)
        if s.endswith(delim):
            stmt = "\n".join(buf).rstrip()
            stmt = stmt[: -len(delim)].strip()
            if stmt:
                out.append(stmt)
            buf = []
    if "".join(buf).strip():
        out.append("\n".join(buf))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-seed", action="store_true")
    a = ap.parse_args()

    print(f"Connecting to MySQL at {config.DB['host']}:{config.DB['port']} as {config.DB_ADMIN_USER} ...")
    try:
        con = pymysql.connect(host=config.DB["host"], port=config.DB["port"], user=config.DB_ADMIN_USER,
                              password=config.DB_ADMIN_PASSWORD, autocommit=True, charset="utf8mb4")
    except pymysql.err.OperationalError as e:
        sys.exit(f"Could not connect: {e}\nCheck DB_ADMIN_USER / DB_ADMIN_PASSWORD in .env and that MySQL is running.")
    cur = con.cursor()
    cur.execute("SELECT VERSION()"); ver = cur.fetchone()[0]; print("MySQL version:", ver)

    files = sorted((ROOT / "database").glob("0*.sql"))
    if a.no_seed:
        files = [f for f in files if "seed" not in f.name]
    for f in files:
        stmts = split_sql(f.read_text(encoding="utf-8"))
        for st in stmts:
            cur.execute(st)
        print(f"  loaded {f.name:22s} ({len(stmts)} statements)")

    u, p, db = config.DB["user"], config.DB["password"], config.DB["database"]
    for host in ("localhost", "127.0.0.1"):
        cur.execute(f"CREATE USER IF NOT EXISTS '{u}'@'{host}' IDENTIFIED BY %s", (p,))
        cur.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE, EXECUTE, SHOW VIEW ON `{db}`.* TO '{u}'@'{host}'")
    cur.execute("FLUSH PRIVILEGES")
    print(f"Application user '{u}' ready.")

    cur.execute(f"USE `{db}`")
    for t in ["farmers", "plots", "crops", "sensor_nodes", "soil_health_logs", "cultivation_cycles", "treatments"]:
        cur.execute(f"SELECT COUNT(*) FROM {t}"); print(f"  {t:20s} {cur.fetchone()[0]:>6} rows")
    print("Done.")


if __name__ == "__main__":
    main()
