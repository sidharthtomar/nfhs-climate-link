"""Central configuration: paths, variable maps, published benchmarks.

Everything that is survey-round-specific lives here so that the rest of the
codebase never hardcodes a variable name. This matters more than it looks --
see the note on suffix drift below.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"

for _p in (DATA_RAW, DATA_INTERIM, DATA_PROCESSED, FIGURES, TABLES):
    _p.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Menstrual protection items
# --------------------------------------------------------------------------
# NFHS-5 asks women aged 15-24 what they use for protection during
# menstruation. It is a MULTIPLE RESPONSE item: one binary variable per
# material, so a respondent can be 1 on several at once.
#
# !! DO NOT MAP THESE BY LETTER SUFFIX ACROSS ROUNDS !!
#
# NFHS-5 inserted "menstrual cup" at position (e), which shifted every
# response after it down one letter relative to NFHS-4:
#
#     material                  NFHS-4     NFHS-5
#     cloth                     s257a      s260a
#     locally prepared napkins  s257b      s260b
#     sanitary napkins          s257c      s260c
#     tampons                   s257d      s260d
#     menstrual cup             --         s260e
#     nothing                   s257e      s260f     <-- collides
#     other                     s257x      s260x
#
# A positional map sends NFHS-4 "nothing" onto NFHS-5 "menstrual cup",
# inverting the outcome for the most deprived respondents in the sample.
# The code therefore always resolves through the canonical label below.

PROTECTION_ITEMS = {
    "nfhs5": {
        "s260a": "cloth",
        "s260b": "locally_prepared_napkins",
        "s260c": "sanitary_napkins",
        "s260d": "tampons",
        "s260e": "menstrual_cup",
        "s260f": "nothing",
        "s260x": "other",
    },
    # Retained for reference / future extension. This project analyses
    # NFHS-5 only; pooling the rounds requires district harmonisation
    # (707 -> 640) which is deliberately out of scope.
    "nfhs4": {
        "s257a": "cloth",
        "s257b": "locally_prepared_napkins",
        "s257c": "sanitary_napkins",
        "s257d": "tampons",
        "s257e": "nothing",
        "s257x": "other",
    },
}

# Two competing conventions exist in the published literature. Which one you
# pick moves headline prevalence by ~28 percentage points, so it is a
# modelling decision, not a detail. We implement both and report both.
HYGIENIC_NARROW = {"sanitary_napkins", "tampons", "menstrual_cup"}
HYGIENIC_BROAD = HYGIENIC_NARROW | {"locally_prepared_napkins"}


# --------------------------------------------------------------------------
# Other variables
# --------------------------------------------------------------------------

SURVEY_DESIGN = {
    "weight": "v005",        # divide by 1e6
    "psu": "v021",
    "strata": "v023",
    "cluster": "v001",       # merge key to DHS GPS + geospatial covariates (DHSCLUST)
    "interview_month": "v006",
    "interview_year": "v007",
    "interview_cmc": "v008",
}

# --------------------------------------------------------------------------
# DHS geospatial covariates (IAGC7AFL.csv)
# --------------------------------------------------------------------------
# Cluster-level climate/environment already joined by DHS. Keyed on DHSCLUST,
# which equals the survey cluster id v001. These are long-run normals and
# 5-yearly snapshots (2000-2020), NOT interview-windowed daily weather -- good
# predictive features, but they describe a district's typical climate, not the
# acute heat a woman experienced before interview. The GEE layer (02b) adds the
# windowed metric; this file is the guaranteed no-GEE base (02a).

GEOCOV_FILE = "IAGC7AFL.csv"
GEOCOV_KEY = "DHSCLUST"          # == survey v001

# Columns to carry from the covariates file. 2015/2020 chosen as closest to
# NFHS-5 fieldwork (2019-21). Suffix pattern is <name>_<year>.
GEOCOV_FEATURES = [
    "Land_Surface_Temperature_2015",
    "Day_Land_Surface_Temp_2015",
    "Night_Land_Surface_Temp_2015",
    "Diurnal_Temperature_Range_2015",
    "Maximum_Temperature_2015",
    "Mean_Temperature_2015",
    "Minimum_Temperature_2015",
    "Frost_Days_2015",
    "Wet_Days_2015",
    "Aridity_2015",
    "PET_2015",
    "Precipitation_2015",
    "Enhanced_Vegetation_Index_2015",
    "Nightlights_Composite",
    "Elevation",
    "Travel_Times",
]

# The 12 monthly temperature normals -- lets us derive a hot-season feature
# without any external data.
GEOCOV_MONTHLY_TEMP = [
    "Temperature_January", "Temperature_February", "Temperature_March",
    "Temperature_April", "Temperature_May", "Temperature_June",
    "Temperature_July", "Temperature_August", "Temperature_September",
    "Temperature_October", "Temperature_November", "Temperature_December",
]

# --------------------------------------------------------------------------
# Earth Engine (02b -- additive interview-windowed heat exposure)
# --------------------------------------------------------------------------

GEE_PROJECT = "cogent-octane-504408-e4"
ERA5_LAND = "ECMWF/ERA5_LAND/DAILY_AGGR"
# Heat-index threshold is defined per district relative to its own 1991-2020
# climatology, so no absolute cutoff is hardcoded. See climate.py.
HEAT_PERCENTILE = 90
CLIMATOLOGY_YEARS = (1991, 2020)

COVARIATES = {
    "age": "v012",
    "education": "v106",
    "wealth_quintile": "v190",
    "residence": "v025",
    "state": "v024",
    "district": "sdist",
    "religion": "v130",
    "caste": "s116",
    "marital_status": "v501",
    "age_at_menarche": "s259",
    "media_newspaper": "v157",
    "media_radio": "v158",
    "media_tv": "v159",
    "water_source": "v113",
    "toilet_type": "v116",
    "chw_menstrual_talk": "s365s",
}

# Negative control: no plausible causal pathway from short-run heat, but
# shares the socioeconomic and geographic determinants of the real outcome.
# If the model "predicts" this as well as it predicts hygiene, the signal is
# confounding rather than substance.
NEGATIVE_CONTROL = "s931"   # owns a bank account she herself uses

STATE_MODULE_FLAG = "ssmod"


# --------------------------------------------------------------------------
# Published benchmarks
# --------------------------------------------------------------------------
# Reproduce at least one exclusive and one non-exclusive figure BEFORE
# modelling. If the recode is wrong (see suffix note above) it will fail
# loudly here rather than silently producing a plausible-looking model.

BENCHMARKS = {
    "n_women_15_24": {
        "expected": 241_180,
        "tolerance": 0.01,
        "source": "Singh & Singh 2025, Front Reprod Health",
    },
    "any_hygienic_broad": {
        "expected": 0.776,
        "tolerance": 0.02,
        "source": "Singh & Singh 2025, Table 3 (India total)",
    },
    "sanitary_napkin_use": {
        "expected": 0.644,
        "tolerance": 0.02,
        "source": "Singh & Singh 2025, Table 3 (India total)",
    },
    "exclusive_hygienic_broad": {
        "expected": 0.498,
        "tolerance": 0.03,
        "source": "Meher & Sahoo 2023, BMC Women's Health -- 'hygienic' category of their 3-way split; locally-prepared napkins counted hygienic, mixed users excluded",
    },
}

AGE_MIN, AGE_MAX = 15, 24
RANDOM_SEED = 20260819
