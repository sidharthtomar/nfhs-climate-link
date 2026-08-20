# nfhs-climate-link

A reproducible pipeline linking India's National Family Health Survey (NFHS-5, n≈241k women aged 15–24) to gridded climate reanalysis, with a predictive modelling layer and an honest evaluation protocol.

**Status:** Week 1 of 4 complete — survey ingestion and recode validation.

---

## What this is

Two things that are usually done badly, done carefully:

**1. Joining survey microdata to climate rasters.** NFHS is one of the largest health surveys in the world and is widely used, but every team that links it to environmental data rebuilds the join from scratch. This is an open, tested implementation.

**2. Evaluating a geospatial model without fooling yourself.** Random k-fold cross-validation on spatially clustered data leaks: test-set women live in the same districts as training-set women, so the model partly memorises district effects and reports accuracy it does not have. This repo measures the same model under random CV and under district-grouped CV and reports both. The gap is the point.

The substantive question — does ambient heat relate to menstrual hygiene practice, and for whom — is real and under-studied. But the engineering claim is what this repository is for.

---

## The bug this codebase is built around

NFHS records menstrual protection as multiple-response binaries, one per material. Between survey rounds, NFHS-5 inserted *menstrual cup* at position `(e)`, shifting every later response down one letter:

| material | NFHS-4 | NFHS-5 |
|---|---|---|
| cloth | `s257a` | `s260a` |
| locally prepared napkins | `s257b` | `s260b` |
| sanitary napkins | `s257c` | `s260c` |
| tampons | `s257d` | `s260d` |
| menstrual cup | — | `s260e` |
| **nothing** | **`s257e`** | **`s260f`** |
| other | `s257x` | `s260x` |

Mapping these by suffix — the obvious loop — sends NFHS-4 *"used nothing"* onto NFHS-5 *"used a menstrual cup"*. It inverts the outcome precisely for the most deprived respondents in the sample. It does not raise. It produces a believable prevalence and a believable model.

Two defences, both in the code:

- Every mapping resolves through a canonical label (`config.PROTECTION_ITEMS`); no downstream code touches a suffix.
- `tests/test_outcomes.py::test_suffix_trap` encodes the failure so a future refactor fails CI.

## The second defence: external validation

Before any modelling, the pipeline reproduces published figures from the raw file:

```
python scripts/01_build_survey.py --recode data/raw/IAIR7EFL.DTA
```

```
========================================================================
RECODE VALIDATION -- published NFHS-5 benchmarks
========================================================================
[    ] n_women_15_24            observed ...  expected 241,180
[    ] any_hygienic_broad       observed ...  expected  77.6%
[    ] exclusive_hygienic       observed ...  expected  49.8%
[    ] sanitary_napkin_use      observed ...  expected  64.4%
========================================================================
```

Failure raises and blocks the pipeline. A recode that cannot reproduce known numbers is not a recode you should model with.

## Definitional sensitivity

"Menstrual hygiene" has no single operationalisation. Two independent choices — whether locally prepared napkins count as hygienic, and whether *exclusive* use is required — produce four definitions whose published prevalence spans **49.8% to 77.6%**.

That 28-point range is larger than any effect this or any similar study could detect. The pipeline computes all four and reports them together rather than picking one silently.

---

## Layout

```
src/nfhs_climate/
  config.py            paths, variable maps, published benchmarks
  data/nfhs.py         recode loading, outcome construction
  data/validate.py     benchmark checks (fail-loud)
  data/climate.py      [week 2] heat index, climatology, district join
  models/              [week 3] baselines, spatial CV, interpretation
scripts/
  01_build_survey.py   ingest + validate  ✅
  02_build_climate.py  [week 2]
  03_train.py          [week 3]
tests/
```

## Setup

```bash
pip install -r requirements.txt
python scripts/01_build_survey.py --recode data/raw/IAIR7EFL.DTA
```

NFHS microdata is not redistributable. Obtain it from [The DHS Program](https://dhsprogram.com/data/); registration is free. Place the individual recode in `data/raw/`.

## Roadmap

- [x] **Week 1** — survey ingestion, outcome construction, benchmark validation, tests
- [ ] **Week 2** — climate pipeline: heat index from temperature and dewpoint, district-specific 1991–2020 percentile thresholds, exposure windows, spatial join
- [ ] **Week 3** — logistic baseline, gradient boosting, random vs. district-grouped CV, SHAP, calibration
- [ ] **Week 4** — figures, documentation, write-up

## Scope

NFHS-5 only. Pooling NFHS-4 requires harmonising 707 districts onto 640, which no published source in this literature provides — every existing NFHS-4/NFHS-5 comparison works at state level or maps each round separately. Deliberately out of scope.

This is predictive modelling, not causal inference. Associations here are not effects, and the repository does not claim otherwise.

## Data sources

- NFHS-5 (2019–21), IIPS and ICF, via The DHS Program
- ERA5-Land reanalysis, Copernicus Climate Data Store
- Surface PM2.5 estimates, Atmospheric Composition Analysis Group, Washington University

## Licence

Code MIT. No survey microdata is included or redistributable.
