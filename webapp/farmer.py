"""Turns the recommender's technical output into plain-language advice for farmers.
Every statement here is derived from fields the recommender already returns
(fit card, season / rotation / nitrogen factors, crop catalogue) - no new numbers."""
from datetime import date, datetime, timedelta

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SEASON_MONTHS = {"Kharif": "Jun–Sep", "Rabi": "Oct–Jan", "Zaid": "Feb–May"}

N_NEED = {"HIGH": "Needs a lot of nitrogen fertiliser",
          "MEDIUM": "Needs a moderate amount of nitrogen fertiliser",
          "LOW": "Needs little nitrogen fertiliser",
          "FIXER": "Legume – makes much of its own nitrogen"}

# (too low, too high) wording per measurement
FIT_WORDS = {
    "moisture_pct": ("Soil is drier than {c} usually needs", "Soil is wetter than {c} usually likes"),
    "soil_temp_c": ("Soil is cooler than ideal for {c}", "Soil is warmer than ideal for {c}"),
    "air_temp_c": ("Weather is cooler than ideal for {c}", "Weather is hotter than ideal for {c}"),
    "humidity_pct": ("Air is drier than {c} usually likes", "Air is more humid than {c} usually likes"),
    "ph": ("Soil is more acidic than ideal for {c}", "Soil is more alkaline than ideal for {c}"),
}
FIT_GOOD = {"moisture_pct": "soil moisture", "soil_temp_c": "soil temperature",
            "air_temp_c": "air temperature", "humidity_pct": "humidity", "ph": "soil pH"}
UNIT = {"moisture_pct": "%", "soil_temp_c": " °C", "air_temp_c": " °C", "humidity_pct": "%", "ph": ""}

STATUS = {
    "CONFIDENT": ("ok", "Your soil readings clearly match the crops below."),
    "LOW_CONFIDENCE": ("warn", "Your readings suit several crops about equally – compare the top choices before deciding."),
    "OUT_OF_DISTRIBUTION": ("bad", "These readings look unusual. Check the sensor or take a fresh reading before deciding."),
    "NO_VALID_DATA": ("bad", "No usable soil readings yet. Add readings first."),
}


def crop_name(c: str) -> str:
    return {"kidneybeans": "Kidney beans", "pigeonpeas": "Pigeon peas", "mothbeans": "Moth beans",
            "mungbean": "Mung bean", "blackgram": "Black gram", "muskmelon": "Muskmelon"}.get(c, c.capitalize())


def sow_date_for(month: int) -> date:
    today = date.today()
    if month == today.month:
        return today
    year = today.year + (1 if month < today.month else 0)
    return date(year, month, 1)


def crop_card(x: dict, month: int, area_acres: float | None, soil_type: str) -> dict:
    """One farmer-facing card for one ranked crop."""
    info, name = x["info"], crop_name(x["crop"])
    good, warn, fits = [], [], []
    for f in x.get("fit_card", []):
        if f["feature"] == "soil_type":
            (good if f["state"] == "ok" else warn).append(
                f"Commonly grown on {soil_type} soil" if f["state"] == "ok" else f"Not often grown on {soil_type} soil")
        elif f["state"] == "ok":
            fits.append(FIT_GOOD[f["feature"]])
        else:
            lo_txt, hi_txt = FIT_WORDS[f["feature"]]
            rng = f"yours {f['value']:.1f}, usual {f['lo']:.1f}–{f['hi']:.1f}{UNIT[f['feature']]}"
            warn.append(f"{(lo_txt if f['state'] == 'low' else hi_txt).format(c=name.lower())} ({rng})")
    if fits:
        good.insert(0, "Your " + (", ".join(fits[:-1]) + " and " + fits[-1] if len(fits) > 1 else fits[0]) + " suit it")

    if x["season_f"] < 1:
        warn.append(f"Best sown in {info['season']} season ({SEASON_MONTHS.get(info['season'], '')}), not {MONTHS[month - 1]}")
    elif not info.get("is_perennial"):
        good.append(f"Right season to sow ({info['season']})")
    if x["rotation_f"] == 0:
        warn.append("You grew this last time – change crop to protect the soil")
    elif x["rotation_f"] < 1:
        warn.append(f"Same crop family ({info['family']}) as your last crop – rotating is better")
    elif x["rotation_f"] > 1:
        good.append("Good rotation – a legume puts nitrogen back into the soil")
    if x["nutrient_f"] < 1:
        warn.append("Soil nitrogen is probably low after your last crops – plan extra fertiliser")
    elif x["nutrient_f"] > 1:
        good.append("Suits your soil's estimated nitrogen level")

    sow = sow_date_for(month)
    days = int(info["maturation_days"])
    harvest = sow + timedelta(days=days)
    yield_total = float(info["expected_yield_kg_acre"]) * float(area_acres) if area_acres else None
    score = float(x["score"])
    label = "Best choice" if score >= 0.4 else ("Good option" if score >= 0.15 else "Possible")
    return {"crop": x["crop"], "name": name, "score": score, "label": label, "good": good, "warn": warn,
            "season": info["season"], "days": days, "sow": sow, "harvest": harvest,
            "n_need": N_NEED.get(info["n_demand"], ""), "yield_per_acre": float(info["expected_yield_kg_acre"]),
            "yield_total": yield_total, "blocked": x["rotation_f"] == 0}


def advice(res: dict, month: int, area_acres: float | None, soil_type: str, top: int = 3) -> dict:
    tone, msg = STATUS.get(res.get("status"), ("mute", ""))
    cards = [crop_card(x, month, area_acres, soil_type) for x in (res.get("ranking") or [])
             if x["score"] >= 0.01][:top]
    if not cards and res.get("ranking"):
        cards = [crop_card(res["ranking"][0], month, area_acres, soil_type)]
    if cards:
        cards[0]["label"] = "Best match"
    return {"tone": tone, "message": msg, "cards": cards, "with_ph": res.get("model_used") == "with_pH"}


def ago(ts) -> str:
    if not ts:
        return "never"
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts)
    mins = int((datetime.now() - ts).total_seconds() // 60)
    if mins < 1:
        return "just now"
    if mins < 60:
        return f"{mins} min ago"
    if mins < 48 * 60:
        return f"{mins // 60} h ago"
    return f"{mins // 1440} days ago"
