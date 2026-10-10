"""
Agronomic knowledge used by the context-aware re-ranker and the MySQL crop
catalogue seed.  These are general, textbook-level values for Indian
conditions and are NOT model outputs.  Verify / adjust before the final report
(e.g. against ICAR crop production guides and the Directorate of Economics &
Statistics "Agricultural Statistics at a Glance").

season       : Kharif (Jun-Oct sowing), Rabi (Oct-Dec sowing), Zaid (Feb-May
               sowing), Perennial (plantation / orchard)
family       : used for rotation rules (same family back-to-back is discouraged)
n_demand     : nitrogen demand class -> HIGH / MEDIUM / LOW / FIXER (legume)
days         : typical sowing-to-harvest days (perennials: one harvest cycle)
yield_kg_acre: approximate typical yield, used ONLY to seed demo data and for
               the actual-vs-expected efficiency metric
"""

CROPS = {
    #  name          season       family        n_demand  perennial days  yield_kg_acre
    "rice":        ("Kharif",    "Cereal",     "HIGH",   False, 120, 1100),
    "maize":       ("Kharif",    "Cereal",     "HIGH",   False, 100, 1250),
    "chickpea":    ("Rabi",      "Legume",     "FIXER",  False, 100, 420),
    "kidneybeans": ("Rabi",      "Legume",     "FIXER",  False, 90, 600),
    "pigeonpeas":  ("Kharif",    "Legume",     "FIXER",  False, 160, 330),
    "mothbeans":   ("Kharif",    "Legume",     "FIXER",  False, 75, 150),
    "mungbean":    ("Zaid",      "Legume",     "FIXER",  False, 65, 250),
    "blackgram":   ("Kharif",    "Legume",     "FIXER",  False, 80, 280),
    "lentil":      ("Rabi",      "Legume",     "FIXER",  False, 110, 370),
    "watermelon":  ("Zaid",      "Cucurbit",   "MEDIUM", False, 85, 10000),
    "muskmelon":   ("Zaid",      "Cucurbit",   "MEDIUM", False, 80, 7000),
    "cotton":      ("Kharif",    "Fibre",      "HIGH",   False, 165, 600),
    "jute":        ("Kharif",    "Fibre",      "MEDIUM", False, 120, 1000),
    "pomegranate": ("Perennial", "Fruit",      "MEDIUM", True, 180, 4000),
    "banana":      ("Perennial", "Fruit",      "HIGH",   True, 330, 14000),
    "mango":       ("Perennial", "Fruit",      "MEDIUM", True, 120, 3000),
    "grapes":      ("Perennial", "Fruit",      "MEDIUM", True, 150, 8500),
    "apple":       ("Perennial", "Fruit",      "MEDIUM", True, 150, 3500),
    "orange":      ("Perennial", "Fruit",      "MEDIUM", True, 240, 4000),
    "papaya":      ("Perennial", "Fruit",      "HIGH",   True, 270, 16000),
    "coconut":     ("Perennial", "Plantation", "MEDIUM", True, 365, 4000),
    "coffee":      ("Perennial", "Plantation", "MEDIUM", True, 240, 350),
}

FIELDS = ["season", "family", "n_demand", "is_perennial", "maturation_days", "expected_yield_kg_acre"]


def info(crop: str) -> dict:
    return dict(zip(FIELDS, CROPS[crop]))


def season_for_month(month: int) -> str:
    """Indian cropping season in which sowing normally happens in `month`."""
    if month in (6, 7, 8, 9):
        return "Kharif"
    if month in (10, 11, 12, 1):
        return "Rabi"
    return "Zaid"  # Feb - May
