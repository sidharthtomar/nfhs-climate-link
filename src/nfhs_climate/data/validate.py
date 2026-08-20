"""Validate the recode against published NFHS-5 figures.

Why this module exists
----------------------
The menstrual protection items are multiple-response binaries with letter
suffixes whose meaning shifted between survey rounds. A mis-coded outcome
does not crash; it produces a plausible prevalence, a plausible model, and a
completely wrong paper.

The defence is external validation: reproduce numbers other people have
published, from the raw file, before any modelling. If the recode is wrong
the check fails here -- loudly, at minute five, instead of never.

Run this before anything else.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from ..config import BENCHMARKS
from .nfhs import weighted_mean

log = logging.getLogger(__name__)


@dataclass
class CheckResult:
    name: str
    observed: float
    expected: float
    tolerance: float
    source: str

    @property
    def passed(self) -> bool:
        if pd.isna(self.observed):
            return False
        return abs(self.observed - self.expected) <= self.tolerance

    def __str__(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        if self.expected < 1:  # proportion
            body = (
                f"observed {self.observed:6.1%}  expected {self.expected:6.1%}  "
                f"(+/- {self.tolerance:.1%})"
            )
        else:  # count
            body = (
                f"observed {self.observed:,.0f}  expected {self.expected:,.0f}  "
                f"(+/- {self.tolerance:.0%})"
            )
        return f"[{mark}] {self.name:24s} {body}\n         source: {self.source}"


def run_checks(df: pd.DataFrame) -> list[CheckResult]:
    """Compare the built frame against published benchmarks."""
    w = df["survey_weight"]
    results: list[CheckResult] = []

    def add(key: str, observed: float) -> None:
        spec = BENCHMARKS[key]
        tol = spec["tolerance"]
        # count-type benchmarks express tolerance as a fraction of expected
        tol_abs = tol * spec["expected"] if spec["expected"] >= 1 else tol
        results.append(
            CheckResult(key, observed, spec["expected"], tol_abs, spec["source"])
        )

    add("n_women_15_24", float(len(df)))
    add("any_hygienic_broad", weighted_mean(df["any_hygienic_broad"], w))
    add("exclusive_hygienic_broad", weighted_mean(df["exclusive_hygienic_broad"], w))

    if "uses_sanitary_napkins" in df.columns:
        add("sanitary_napkin_use", weighted_mean(df["uses_sanitary_napkins"], w))

    return results


def report(df: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    """Print the benchmark table. Raises on failure when strict=True."""
    results = run_checks(df)

    print("\n" + "=" * 72)
    print("RECODE VALIDATION -- published NFHS-5 benchmarks")
    print("=" * 72)
    for r in results:
        print(r)
    print("=" * 72)

    failed = [r for r in results if not r.passed]
    if failed:
        msg = (
            f"{len(failed)} of {len(results)} benchmark checks failed: "
            f"{[r.name for r in failed]}.\n"
            "Do NOT proceed to modelling. Most likely causes, in order:\n"
            "  1. Protection items mapped by letter suffix rather than label\n"
            "     (s260e is MENSTRUAL CUP; s260f is NOTHING)\n"
            "  2. Survey weights not divided by 1e6\n"
            "  3. Age restriction not applied, or applied to the wrong variable\n"
            "  4. Wrong recode file (household instead of individual)\n"
            "  5. 'Hygienic' definition mismatched to the benchmark's convention"
        )
        if strict:
            raise ValueError(msg)
        log.warning(msg)
    else:
        print("All benchmarks reproduced. Recode is trustworthy.\n")

    return pd.DataFrame(
        [
            {
                "check": r.name,
                "observed": r.observed,
                "expected": r.expected,
                "passed": r.passed,
                "source": r.source,
            }
            for r in results
        ]
    )


def prevalence_table(df: pd.DataFrame) -> pd.DataFrame:
    """All four outcome definitions side by side.

    Include this in the README. It makes the definitional sensitivity of the
    outcome visible instead of burying it in a methods footnote.
    """
    w = df["survey_weight"]
    rows = []
    for scope in ("any", "exclusive"):
        for width in ("narrow", "broad"):
            col = f"{scope}_hygienic_{width}"
            if col in df.columns:
                rows.append(
                    {
                        "definition": f"{scope} / {width}",
                        "locally_prepared_counted": width == "broad",
                        "exclusive_use_required": scope == "exclusive",
                        "weighted_prevalence": weighted_mean(df[col], w),
                        "n_nonmissing": int(df[col].notna().sum()),
                    }
                )
    return pd.DataFrame(rows)
