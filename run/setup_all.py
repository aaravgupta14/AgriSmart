"""One-click AgriSmart setup (run via VS Code task 'AgriSmart 1' or run\\1_setup.bat).
Creates .venv, installs packages, trains the models, then builds the MySQL database.
Everything is also written to run\\setup_log.txt."""
import os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "run" / "setup_log.txt"
VPY = ROOT / ".venv" / "Scripts" / "python.exe"
PLACEHOLDER = "PUT_YOUR_MYSQL_ROOT_PASSWORD_HERE"
log = open(LOG, "w", encoding="utf-8", errors="replace")

def say(msg):
    print(msg, flush=True); log.write(msg + "\n"); log.flush()

def run(cmd, cwd=ROOT):
    say("> " + " ".join(str(c) for c in cmd))
    p = subprocess.Popen([str(c) for c in cmd], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace")
    for line in p.stdout:
        print(line, end="", flush=True); log.write(line); log.flush()
    p.wait()
    if p.returncode:
        say(f"!! command failed with exit code {p.returncode}"); sys.exit(p.returncode)

say(f"==== AgriSmart setup started {time.ctime()} (python {sys.version.split()[0]})")
say("\n== 1. Virtual environment")
if not VPY.exists():
    run([sys.executable, "-m", "venv", ROOT / ".venv"])
else:
    say(".venv already exists")
say("\n== 2. Python packages")
run([VPY, "-m", "pip", "install", "--disable-pip-version-check", "-r", ROOT / "requirements.txt"])

say("\n== 3. Train the models (about 3 minutes)")
if not (ROOT / "ml" / "models" / "crop_model_with_pH.joblib").exists():
    run([VPY, "-W", "ignore", "train_models.py"], cwd=ROOT / "ml")
else:
    say("models already present")

say("\n== 4. MySQL service")
q = subprocess.run(["sc", "query", "MySQL80"], capture_output=True, text=True)
say("MySQL80 is RUNNING" if "RUNNING" in q.stdout else "MySQL80 is NOT running - start it from services.msc (MySQL80 -> Start)")

say("\n== 5. Database")
env = ROOT / ".env"
if not env.exists():
    env.write_text((ROOT / ".env.example").read_text().replace("your_mysql_root_password", PLACEHOLDER))
def pw():
    for line in env.read_text().splitlines():
        if line.startswith("DB_ADMIN_PASSWORD="):
            return line.split("=", 1)[1].strip()
    return ""
waited = 0
while pw() in ("", PLACEHOLDER) and waited < 1800:
    if waited == 0:
        say("Waiting for you to put your MySQL root password in .env (DB_ADMIN_PASSWORD=...) and save (Ctrl+S)...")
    time.sleep(5); waited += 5
if pw() in ("", PLACEHOLDER):
    say("No password entered - re-run this task after editing .env"); sys.exit(1)
run([VPY, ROOT / "scripts" / "setup_database.py"])
say("\n==== SETUP COMPLETE. Now run task 'AgriSmart 2: Start website' and open http://127.0.0.1:5000")
