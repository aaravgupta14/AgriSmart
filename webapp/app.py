"""
AgriSmart web application (Flask + MySQL)
Run from the project root:   python -m webapp.app      ->  http://127.0.0.1:5000
"""
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path

import pymysql
from flask import (Flask, abort, flash, jsonify, redirect, render_template, request, send_from_directory,
                   session, url_for)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from webapp import farmer as fm  # noqa: E402
from webapp import services as svc  # noqa: E402
from webapp.db import callproc, close_db, execute, get_db, query  # noqa: E402

app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY
app.teardown_appcontext(close_db)

SOIL_TYPES = ["Alluvial", "Black", "Red", "Laterite", "Sandy"]
FEATURE_LABEL = {"moisture_pct": "Soil moisture (%)", "soil_temp_c": "Soil temp (°C)",
                 "air_temp_c": "Air temp (°C)", "humidity_pct": "Humidity (%)", "ph": "pH",
                 "soil_type": "Soil type"}


def db_error_message(e):
    return e.args[1] if isinstance(e, pymysql.err.MySQLError) and len(e.args) > 1 else str(e)


@app.context_processor
def inject():
    try:
        fields = query("SELECT plot_id, plot_name, farmer FROM v_plot_overview ORDER BY farmer, plot_name")
    except pymysql.err.MySQLError:
        fields = []
    return {"FEATURE_LABEL": FEATURE_LABEL, "now": datetime.now(), "SOIL_TYPES": SOIL_TYPES,
            "all_fields": fields, "current_field": session.get("plot_id"), "MONTHS": fm.MONTHS,
            "crop_name": fm.crop_name, "ago": fm.ago}


def current_field_id():
    """The farmer's selected field (kept in the session); defaults to the first field."""
    pid = request.values.get("plot_id", type=int) or session.get("plot_id")
    if pid is None:
        first = query("SELECT plot_id FROM plots ORDER BY plot_id LIMIT 1", one=True)
        pid = first["plot_id"] if first else None
    if pid:
        session["plot_id"] = pid
    return pid


@app.template_filter("pct")
def pct(v, digits=0):
    return "-" if v is None else f"{100 * float(v):.{digits}f}%"


@app.template_filter("num")
def num(v, digits=1):
    return "-" if v is None else f"{float(v):,.{digits}f}"


@app.errorhandler(pymysql.err.OperationalError)
def db_down(e):
    return render_template("db_error.html", error=str(e)), 500


# ------------------------------------------------------- farmer pages
# The farmer flow: My field (live data) -> Add soil data (manual, optional)
# -> Crop advice -> My crops (fertiliser, harvest, results).
def field_or_404(pid):
    if not pid:
        abort(404)
    return query("SELECT * FROM v_plot_overview WHERE plot_id = %s", (pid,), one=True) or abort(404)


SENSOR_CHOICES = [("MOISTURE", "Soil moisture sensor (capacitive)"),
                  ("AIR_TEMP_HUMIDITY", "Air temperature + humidity sensor (DHT11 / DHT22)"),
                  ("SOIL_TEMP", "Soil temperature probe (DS18B20)")]
# no pH probe option: pH is typed in from the Soil Health Card (sketch has USE_PH 0)


def next_node_id():
    r = query("SELECT MAX(CAST(SUBSTRING(node_id, 6) AS UNSIGNED)) n FROM sensor_nodes WHERE node_id REGEXP '^NODE-[0-9]+$'",
              one=True)
    return f"NODE-{(r['n'] or 0) + 1:02d}"


@app.route("/register", methods=["GET", "POST"])
def register():
    """New account: farmer -> field -> how data is collected -> (optional) last crop + first reading.
    Everything is inserted in ONE transaction so a failure leaves nothing half-created."""
    farmers_ = query("SELECT farmer_id, full_name, village FROM farmers ORDER BY full_name")
    crops_ = query("SELECT crop_name, maturation_days FROM crops ORDER BY crop_name")
    f = request.form
    if request.method == "POST":
        errs = []
        existing = f.get("account") == "existing"
        if existing and not f.get("farmer_id", type=int):
            errs.append("choose your name from the list")
        if not existing:
            if not f.get("full_name", "").strip():
                errs.append("enter your name")
            if not re.fullmatch(r"[6-9]\d{9}", f.get("phone_no", "").strip()):
                errs.append("phone number must be 10 digits starting with 6-9")
            if not f.get("village", "").strip():
                errs.append("enter your village")
        try:
            area = float(f.get("area_acres", ""))
            if not 0 < area <= 500:
                raise ValueError
        except ValueError:
            errs.append("field size must be between 0 and 500 acres")
        if f.get("soil_type") not in SOIL_TYPES:
            errs.append("choose a soil type")
        if not f.get("plot_name", "").strip():
            errs.append("give the field a name")
        has_node = f.get("collection") == "sensor"
        sensors = [s for s, _ in SENSOR_CHOICES if f.get(f"s_{s}")]
        node_id = f.get("node_id", "").strip().upper()
        if has_node:
            if not re.fullmatch(r"NODE-\d{2,3}", node_id):
                errs.append("sensor node ID must look like NODE-07")
            if not sensors:
                errs.append("tick at least one sensor on the node")
        vals, rerrs = svc.validate_reading(f)
        first_reading = any(vals.get(k) is not None for k in ("moisture_pct", "soil_temp_c", "air_temp_c", "humidity_pct", "ph"))
        if first_reading:
            errs += [e for e in rerrs if e != "no values supplied"]
        if errs:
            flash("Please fix: " + "; ".join(errs) + ".", "error")
            return render_template("f_register.html", farmers=farmers_, crops=crops_, SENSOR_CHOICES=SENSOR_CHOICES,
                                   suggested_node=next_node_id(), form=f)

        db = get_db()
        try:
            db.begin()
            with db.cursor() as cur:
                if existing:
                    farmer_id = f.get("farmer_id", type=int)
                else:
                    cur.execute("INSERT INTO farmers (full_name, phone_no, village, coop_id) VALUES (%s,%s,%s,%s)",
                                (f["full_name"].strip(), f["phone_no"].strip(), f["village"].strip(),
                                 f.get("coop_id", "").strip() or "NONE"))
                    farmer_id = cur.lastrowid
                cur.execute("""INSERT INTO plots (farmer_id, plot_name, area_acres, soil_type, latitude, longitude)
                               VALUES (%s,%s,%s,%s,NULLIF(%s,''),NULLIF(%s,''))""",
                            (farmer_id, f["plot_name"].strip(), area, f["soil_type"],
                             f.get("latitude", ""), f.get("longitude", "")))
                plot_id = cur.lastrowid
                if has_node:
                    cur.execute("""INSERT INTO sensor_nodes (node_id, plot_id, firmware_ver, installed_on, sensors)
                                   VALUES (%s,%s,'1.0.0',CURDATE(),%s)""", (node_id, plot_id, ",".join(sensors)))
                if f.get("last_crop") and f.get("last_harvest"):
                    days = next(c["maturation_days"] for c in crops_ if c["crop_name"] == f["last_crop"])
                    cur.execute("""INSERT INTO cultivation_cycles (plot_id, crop_id, sow_date, harvest_date, status)
                                   SELECT %s, crop_id, DATE_SUB(%s, INTERVAL %s DAY), %s, 'HARVESTED'
                                   FROM crops WHERE crop_name = %s""",
                                (plot_id, f["last_harvest"] + "-15", days, f["last_harvest"] + "-15", f["last_crop"]))
                if first_reading:
                    cur.execute("""INSERT INTO soil_health_logs (plot_id, node_id, log_ts, source, moisture_pct,
                                   soil_temp_c, air_temp_c, humidity_pct, ph_value, note)
                                   VALUES (%s, NULL, NOW(), 'MANUAL', %s,%s,%s,%s,%s, 'entered at registration')""",
                                (plot_id, vals["moisture_pct"], vals["soil_temp_c"], vals["air_temp_c"],
                                 vals["humidity_pct"], vals["ph"]))
            db.commit()
        except pymysql.err.MySQLError as e:
            db.rollback()
            msg = db_error_message(e)
            if "Duplicate" in msg and "phone" in msg:
                msg = "this phone number already has an account - choose 'I already have an account'"
            elif "Duplicate" in msg and "PRIMARY" in msg:
                msg = f"sensor node {node_id} is already registered to another field"
            flash(f"Account not created: {msg}", "error")
            return render_template("f_register.html", farmers=farmers_, crops=crops_, SENSOR_CHOICES=SENSOR_CHOICES,
                                   suggested_node=next_node_id(), form=f)
        session["plot_id"] = plot_id
        flash(f"Welcome! Field '{f['plot_name'].strip()}' is set up.", "ok")
        return redirect(url_for("home"))
    return render_template("f_register.html", farmers=farmers_, crops=crops_, SENSOR_CHOICES=SENSOR_CHOICES,
                           suggested_node=next_node_id(), form={})


@app.route("/field/<int:plot_id>")
def choose_field(plot_id):
    session["plot_id"] = plot_id
    nxt = request.args.get("next", "")
    return redirect(nxt if nxt.startswith("/") and not nxt.startswith("//") else url_for("home"))


@app.route("/")
def home():
    p = field_or_404(current_field_id())
    nodes = query("""SELECT h.*, n.sensors FROM v_node_health h JOIN sensor_nodes n USING (node_id)
                     WHERE h.plot_id = %s ORDER BY h.node_id""", (p["plot_id"],))
    for n in nodes:
        n["sensor_list"] = [lbl for k, lbl in SENSOR_CHOICES if k in (n["sensors"] or "").split(",")]
        n["no_soil_temp"] = "SOIL_TEMP" not in (n["sensors"] or "")
    online = any(n["is_active"] and n["minutes_since_seen"] is not None and n["minutes_since_seen"] < 60 for n in nodes)
    latest_src = query("""SELECT source, log_ts FROM soil_health_logs WHERE plot_id = %s
                          ORDER BY log_ts DESC LIMIT 1""", (p["plot_id"],), one=True)
    ph = query("SELECT * FROM v_latest_ph WHERE plot_id = %s", (p["plot_id"],), one=True)
    growing = query("""SELECT * FROM v_cycle_performance WHERE plot_id = %s AND status = 'GROWING'
                       ORDER BY sow_date DESC LIMIT 1""", (p["plot_id"],), one=True)
    n24 = query("""SELECT COUNT(*) n FROM soil_health_logs WHERE plot_id = %s
                   AND log_ts >= NOW() - INTERVAL 1 DAY""", (p["plot_id"],), one=True)["n"]
    return render_template("f_home.html", p=p, nodes=nodes, online=online, latest_src=latest_src, ph=ph,
                           growing=growing, n24=n24, today=date.today())


@app.route("/add-data", methods=["GET", "POST"])
def add_data():
    pid = current_field_id()
    p = field_or_404(pid)
    quick = None
    if request.method == "POST":
        vals, errs = svc.validate_reading(request.form)
        if request.form.get("mode") == "quick":
            if errs or any(vals[k] is None for k in ("moisture_pct", "soil_temp_c", "air_temp_c", "humidity_pct")):
                flash("For a quick check, enter all four readings within valid ranges. " + "; ".join(errs), "error")
            else:
                month = request.form.get("month", type=int) or date.today().month
                res = svc.nan_to_none(svc.recommend_manual(vals, request.form["soil_type"], vals["ph"],
                                                           request.form.get("last_crop") or None, month))
                quick = {"res": res, "advice": fm.advice(res, month, None, request.form["soil_type"]),
                         "month": month}
        elif errs:
            flash("Reading not saved: " + "; ".join(errs), "error")
        else:
            try:
                execute("""INSERT INTO soil_health_logs (plot_id, node_id, log_ts, source, moisture_pct, soil_temp_c,
                           air_temp_c, humidity_pct, ph_value, note) VALUES (%s, NULL, COALESCE(NULLIF(%s,''), NOW()),
                           'MANUAL', %s,%s,%s,%s,%s, NULLIF(%s,''))""",
                        (pid, request.form.get("log_ts", "").replace("T", " "), vals["moisture_pct"],
                         vals["soil_temp_c"], vals["air_temp_c"], vals["humidity_pct"], vals["ph"],
                         request.form.get("note", "")))
                flash("Reading saved. Your crop advice now includes it.", "ok")
                return redirect(url_for("home"))
            except pymysql.err.MySQLError as e:
                flash(f"Reading not saved: {db_error_message(e)}", "error")
    return render_template("f_add_data.html", p=p, quick=quick,
                           crops=query("SELECT crop_name FROM crops ORDER BY crop_name"))


@app.route("/advice")
def advice():
    p = field_or_404(current_field_id())
    month = request.args.get("month", type=int) or date.today().month
    try:
        res = svc.nan_to_none(svc.recommend_for_plot(p["plot_id"], month, save=False))
    except ValueError as e:
        flash(str(e), "error")
        return redirect(url_for("home"))
    adv = fm.advice(res, month, p["area_acres"], p["soil_type"])
    growing = query("SELECT crop_name FROM v_cycle_performance WHERE plot_id = %s AND status = 'GROWING' LIMIT 1",
                    (p["plot_id"],), one=True)
    stale = res.get("window_end") and (datetime.now() - datetime.fromisoformat(res["window_end"])).days >= 2
    return render_template("f_advice.html", p=p, res=res, adv=adv, month=month, growing=growing, stale=stale)


@app.post("/advice/grow")
def grow():
    """Farmer picks a crop: save the recommendation run, then start the cycle via sp_start_cycle."""
    pid, crop, month = request.form.get("plot_id", type=int), request.form["crop_name"], request.form.get("month", type=int)
    sow = request.form.get("sow_date") or date.today().isoformat()
    try:
        run = svc.recommend_for_plot(pid, month, save=True)
        callproc("sp_start_cycle", (pid, crop, sow, run.get("run_id")))
        flash(f"Started growing {fm.crop_name(crop)} (sown {sow}). Track it here.", "ok")
        return redirect(url_for("my_crops"))
    except (pymysql.err.MySQLError, ValueError) as e:
        flash(f"Could not start this crop: {db_error_message(e)}", "error")
        return redirect(url_for("advice", month=month))


@app.route("/my-crops")
def my_crops():
    p = field_or_404(current_field_id())
    rows = query("SELECT * FROM v_cycle_performance WHERE plot_id = %s ORDER BY sow_date DESC", (p["plot_id"],))
    growing = next((r for r in rows if r["status"] == "GROWING"), None)
    treatments = query("""SELECT * FROM treatments WHERE cycle_id = %s ORDER BY applied_on DESC""",
                       (growing["cycle_id"],)) if growing else []
    past = [r for r in rows if r["status"] != "GROWING"]
    return render_template("f_my_crops.html", p=p, growing=growing, treatments=treatments, past=past,
                           today=date.today())


# -------------------------------------------------- admin / project pages
@app.route("/admin")
def dashboard():
    stats = query("""SELECT (SELECT COUNT(*) FROM farmers) farmers, (SELECT COUNT(*) FROM plots) plots,
                            (SELECT COUNT(*) FROM sensor_nodes WHERE is_active) nodes,
                            (SELECT COUNT(*) FROM soil_health_logs) logs,
                            (SELECT COUNT(*) FROM soil_health_logs WHERE log_ts >= NOW() - INTERVAL 1 DAY) logs_24h,
                            (SELECT COUNT(*) FROM cultivation_cycles WHERE status='GROWING') growing,
                            (SELECT COUNT(*) FROM recommendation_runs) runs""", one=True)
    plots = query("SELECT * FROM v_plot_overview ORDER BY plot_id")
    nodes = query("SELECT * FROM v_node_health ORDER BY node_id")
    recent = query("""SELECT r.run_id, r.created_at, r.status, r.model_used, p.plot_name, p.plot_id, c.crop_name, i.final_score
                      FROM recommendation_runs r JOIN plots p ON p.plot_id = r.plot_id
                      JOIN recommendation_items i ON i.run_id = r.run_id AND i.rank_no = 1
                      JOIN crops c ON c.crop_id = i.crop_id ORDER BY r.run_id DESC LIMIT 6""")
    return render_template("dashboard.html", stats=stats, plots=plots, nodes=nodes, recent=recent)


@app.route("/plots")
def plots():
    return render_template("plots.html", plots=query("SELECT * FROM v_plot_overview ORDER BY plot_id"),
                           farmers=query("SELECT farmer_id, full_name, village FROM farmers ORDER BY full_name"))


@app.post("/plots/add")
def add_plot():
    f = request.form
    try:
        execute("""INSERT INTO plots (farmer_id, plot_name, area_acres, soil_type, latitude, longitude)
                   VALUES (%s,%s,%s,%s,NULLIF(%s,''),NULLIF(%s,''))""",
                (f["farmer_id"], f["plot_name"], f["area_acres"], f["soil_type"], f.get("latitude", ""), f.get("longitude", "")))
        flash("Plot added.", "ok")
    except pymysql.err.MySQLError as e:
        flash(f"Could not add plot: {db_error_message(e)}", "error")
    return redirect(url_for("plots"))


@app.route("/plots/<int:plot_id>")
def plot_detail(plot_id):
    p = query("SELECT * FROM v_plot_overview WHERE plot_id = %s", (plot_id,), one=True) or abort(404)
    cycles = query("SELECT * FROM v_cycle_performance WHERE plot_id = %s ORDER BY sow_date DESC", (plot_id,))
    vnbi = query("SELECT fn_virtual_n_index(%s) v", (plot_id,), one=True)["v"]
    rule = query("SELECT * FROM v_recommended_crops WHERE plot_id = %s LIMIT 5", (plot_id,))
    runs = query("""SELECT r.*, c.crop_name, i.final_score FROM recommendation_runs r
                    JOIN recommendation_items i ON i.run_id = r.run_id AND i.rank_no = 1
                    JOIN crops c ON c.crop_id = i.crop_id WHERE r.plot_id = %s ORDER BY r.run_id DESC LIMIT 10""", (plot_id,))
    return render_template("plot_detail.html", p=p, cycles=cycles, vnbi=vnbi, rule=rule, runs=runs,
                           crops=query("SELECT crop_name FROM crops ORDER BY crop_name"))


@app.route("/recommend", methods=["GET", "POST"])
def recommend():
    plots_ = query("SELECT plot_id, plot_name, farmer, soil_type FROM v_plot_overview ORDER BY plot_id")
    result, manual = None, None
    plot_id = request.values.get("plot_id", type=int)
    month = request.values.get("month", type=int) or datetime.now().month
    if request.method == "POST" and request.form.get("mode") == "manual":
        vals, errs = svc.validate_reading(request.form)
        if errs or any(vals[k] is None for k in ("moisture_pct", "soil_temp_c", "air_temp_c", "humidity_pct")):
            flash("Please enter all four sensor values within valid ranges. " + "; ".join(errs), "error")
        else:
            manual = svc.nan_to_none(svc.recommend_manual(vals, request.form["soil_type"], vals["ph"],
                                                          request.form.get("last_crop") or None, month))
    elif plot_id:
        try:
            result = svc.nan_to_none(svc.recommend_for_plot(plot_id, month, save=request.method == "POST"))
        except ValueError as e:
            flash(str(e), "error")
    return render_template("recommend.html", plots=plots_, result=result, manual=manual, plot_id=plot_id,
                           month=month, crops=query("SELECT crop_name FROM crops ORDER BY crop_name"))


@app.route("/logs")
def logs():
    plot_id = request.args.get("plot_id", type=int)
    where, args = ("WHERE l.plot_id = %s", (plot_id,)) if plot_id else ("", ())
    rows = query(f"""SELECT l.*, p.plot_name FROM soil_health_logs l JOIN plots p ON p.plot_id = l.plot_id
                     {where} ORDER BY l.log_ts DESC LIMIT 300""", args)
    return render_template("logs.html", rows=rows, plot_id=plot_id,
                           plots=query("SELECT plot_id, plot_name FROM plots ORDER BY plot_id"))


@app.post("/logs/add")
def add_log():
    vals, errs = svc.validate_reading(request.form)
    if errs:
        flash("Reading rejected: " + "; ".join(errs), "error")
        return redirect(url_for("logs"))
    try:
        execute("""INSERT INTO soil_health_logs (plot_id, node_id, log_ts, source, moisture_pct, soil_temp_c,
                   air_temp_c, humidity_pct, ph_value, note) VALUES (%s, NULL, COALESCE(NULLIF(%s,''), NOW()),
                   'MANUAL', %s,%s,%s,%s,%s, NULLIF(%s,''))""",
                (request.form["plot_id"], request.form.get("log_ts", "").replace("T", " "), vals["moisture_pct"],
                 vals["soil_temp_c"], vals["air_temp_c"], vals["humidity_pct"], vals["ph"], request.form.get("note", "")))
        flash("Manual reading saved.", "ok")
    except pymysql.err.MySQLError as e:
        flash(f"Database rejected the reading: {db_error_message(e)}", "error")
    return redirect(url_for("logs", plot_id=request.form["plot_id"]))


@app.route("/cycles")
def cycles():
    rows = query("SELECT * FROM v_cycle_performance ORDER BY sow_date DESC")
    return render_template("cycles.html", rows=rows,
                           plots=query("SELECT plot_id, plot_name FROM plots ORDER BY plot_id"),
                           crops=query("SELECT crop_name, season FROM crops ORDER BY crop_name"))


@app.post("/cycles/start")
def start_cycle():
    f = request.form
    try:
        callproc("sp_start_cycle", (f["plot_id"], f["crop_name"], f["sow_date"], f.get("run_id") or None))
        flash(f"Cycle started: {f['crop_name']} on plot {f['plot_id']}.", "ok")
    except pymysql.err.MySQLError as e:
        flash(f"Blocked by database rule: {db_error_message(e)}", "error")
    return redirect(request.referrer or url_for("cycles"))


@app.post("/cycles/<int:cycle_id>/harvest")
def harvest(cycle_id):
    try:
        callproc("sp_record_harvest", (cycle_id, request.form["harvest_date"], request.form["yield_kg"]))
        flash("Harvest recorded.", "ok")
    except pymysql.err.MySQLError as e:
        flash(f"Could not record harvest: {db_error_message(e)}", "error")
    return redirect(request.referrer or url_for("cycles"))


@app.post("/cycles/<int:cycle_id>/treatment")
def add_treatment(cycle_id):
    f = request.form
    try:
        execute("""INSERT INTO treatments (cycle_id, applied_on, category, product_name, qty_kg, n_kg, p_kg, k_kg, cost_inr)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (cycle_id, f["applied_on"], f["category"], f["product_name"], f["qty_kg"],
                 f.get("n_kg") or 0, f.get("p_kg") or 0, f.get("k_kg") or 0, f.get("cost_inr") or 0))
        flash("Treatment recorded.", "ok")
    except pymysql.err.MySQLError as e:
        flash(f"Could not save treatment: {db_error_message(e)}", "error")
    return redirect(request.referrer or url_for("cycles"))


@app.route("/reports")
def reports():
    crops = query("SELECT DISTINCT crop_name FROM v_cycle_performance WHERE status='HARVESTED' ORDER BY crop_name")
    years = [r["y"] for r in query("SELECT DISTINCT YEAR(harvest_date) y FROM cultivation_cycles "
                                   "WHERE harvest_date IS NOT NULL ORDER BY y DESC")]
    crop = request.args.get("crop") or (crops[0]["crop_name"] if crops else None)
    year = request.args.get("year", type=int) or (years[0] if years else date.today().year)
    detail, total = [], None
    if crop:
        sets = callproc("sp_yield_report", (crop, year))
        detail = sets[0] if sets else []
        total = sets[1][0] if len(sets) > 1 and sets[1] else None
    by_crop = query("""SELECT crop_name, COUNT(*) cycles, ROUND(AVG(yield_efficiency_pct),1) eff,
                              ROUND(SUM(yield_kg)/SUM(area_acres),1) kg_acre
                       FROM v_cycle_performance WHERE status='HARVESTED' GROUP BY crop_name ORDER BY eff DESC""")
    return render_template("reports.html", crops=crops, years=years, crop=crop, year=year,
                           detail=detail, total=total, by_crop=by_crop)


@app.route("/farmers", methods=["GET", "POST"])
def farmers():
    if request.method == "POST":
        f = request.form
        try:
            execute("INSERT INTO farmers (full_name, phone_no, village, coop_id) VALUES (%s,%s,%s,%s)",
                    (f["full_name"], f["phone_no"], f["village"], f["coop_id"]))
            flash("Farmer added.", "ok")
        except pymysql.err.MySQLError as e:
            flash(f"Could not add farmer: {db_error_message(e)}", "error")
        return redirect(url_for("farmers"))
    rows = query("""SELECT f.*, COUNT(p.plot_id) n_plots, COALESCE(SUM(p.area_acres),0) acres
                    FROM farmers f LEFT JOIN plots p ON p.farmer_id = f.farmer_id
                    GROUP BY f.farmer_id ORDER BY f.full_name""")
    return render_template("farmers.html", rows=rows)


@app.route("/crops")
def crops():
    return render_template("crops.html", rows=query("SELECT * FROM crops ORDER BY season, crop_name"))


@app.route("/model")
def model_page():
    meta = json.loads((config.ROOT / "ml" / "models" / "model_meta.json").read_text())
    exp = config.ROOT / "experiments"
    tables = {}
    import pandas as pd
    for key, path in {"cmp": "exp2_model_comparison/summary_all_models.csv", "npk": "exp1_npk_ablation/results.csv",
                      "qwtc": "exp3_temporal_consensus/results_pivot_top1.csv",
                      "adaptive": "exp4_adaptive_model_selection/results.csv",
                      "ood": "exp5_abstention/ood_detection.csv"}.items():
        f = exp / path
        tables[key] = pd.read_csv(f).to_dict("records") if f.exists() else []
    return render_template("model.html", meta=meta, t=tables)


@app.route("/experiments/<path:fname>")
def experiment_file(fname):
    return send_from_directory(config.ROOT / "experiments", fname)


# ------------------------------------------------------------------- API
@app.post("/api/readings")
def api_readings():
    """Used by hardware/bridge/serial_bridge.py (or any HTTP client).
    JSON: {"node_id": "NODE-01", "moisture_pct": 21.4, "soil_temp_c": 27.1,
           "air_temp_c": 29, "humidity_pct": 71, "ph": null, "log_ts": optional}"""
    d = request.get_json(force=True, silent=True) or {}
    node = query("SELECT node_id, plot_id FROM sensor_nodes WHERE node_id = %s AND is_active", (d.get("node_id"),), one=True)
    if not node:
        return jsonify(ok=False, error="unknown or inactive node_id"), 400
    vals, errs = svc.validate_reading(d)
    if errs:
        return jsonify(ok=False, error="; ".join(errs)), 422
    log_id = execute("""INSERT INTO soil_health_logs (plot_id, node_id, log_ts, moisture_pct, soil_temp_c, air_temp_c,
                        humidity_pct, ph_value) VALUES (%s,%s,COALESCE(%s, NOW()),%s,%s,%s,%s,%s)""",
                     (node["plot_id"], node["node_id"], d.get("log_ts"), vals["moisture_pct"], vals["soil_temp_c"],
                      vals["air_temp_c"], vals["humidity_pct"], vals["ph"]))
    return jsonify(ok=True, log_id=log_id), 201


@app.get("/api/plots/<int:plot_id>/readings")
def api_plot_readings(plot_id):
    days = request.args.get("days", default=7, type=int)
    rows = query("""SELECT log_ts, moisture_pct, soil_temp_c, air_temp_c, humidity_pct, quality_score
                    FROM soil_health_logs WHERE plot_id = %s AND (moisture_pct IS NOT NULL OR air_temp_c IS NOT NULL)
                      AND log_ts >= (SELECT MAX(log_ts) FROM soil_health_logs WHERE plot_id=%s) - INTERVAL %s DAY
                    ORDER BY log_ts""", (plot_id, plot_id, days))
    return jsonify(svc.nan_to_none([{k: (float(v) if v is not None and k != "log_ts" else v) for k, v in r.items()}
                                    for r in rows]))


@app.get("/api/plots/<int:plot_id>/recommend")
def api_recommend(plot_id):
    res = svc.recommend_for_plot(plot_id, request.args.get("month", type=int), save=False)
    res.pop("quality_table", None)
    return jsonify(svc.nan_to_none(res))


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
