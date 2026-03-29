"""
Calendar environment — multi-scale time navigation.

Generates training observations of the form:
"The current date is [SS:MM:HH] [MM/DD/YYYY] . In [N] [unit] , the date will be [SS:MM:HH] [MM/DD/YYYY] ."

Each word/token is a separate string in the observation list.
The system sees them one at a time through observe().

Training data covers 1980-2050 at multiple time scales:
- Second-level: consecutive seconds (reveals 60-period)
- Minute-level: consecutive minutes (reveals 60-period, hour carry)
- Hour-level: consecutive hours (reveals 24-period, day carry)
- Day-level: consecutive days across month boundaries (reveals 28/29/30/31)
- Month-level: consecutive months across year boundaries (reveals 12-period)
- Year-level: year transitions (reveals leap year pattern)

Test: generalize to unseen dates, including day clamping (Jan 31 + 1 month
= Feb 28) and leap year transitions.
"""
from __future__ import annotations

import calendar
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Any


# ---------------------------------------------------------------------------
# Date arithmetic
# ---------------------------------------------------------------------------

def _add_time(dt: datetime, n: int, unit: str) -> datetime:
    """Add n units of time to a datetime. Handles all calendar rules."""
    if unit == "seconds":
        return dt + timedelta(seconds=n)
    elif unit == "minutes":
        return dt + timedelta(minutes=n)
    elif unit == "hours":
        return dt + timedelta(hours=n)
    elif unit == "days":
        return dt + timedelta(days=n)
    elif unit == "months":
        return dt + relativedelta(months=n)
    elif unit == "years":
        return dt + relativedelta(years=n)
    else:
        raise ValueError(f"Unknown unit: {unit}")


def _format_time(dt: datetime) -> list[str]:
    """Format datetime as individual character tokens.

    SS:MM:HH MM/DD/YYYY → each character is one token.
    The system discovers which characters group into meaningful
    units through co-occurrence, not through our tokenization.
    """
    raw = (
        f"{dt.second:02d}:{dt.minute:02d}:{dt.hour:02d}"
        f" "
        f"{dt.month:02d}/{dt.day:02d}/{dt.year:04d}"
    )
    return list(raw)


def _make_observation(dt: datetime, n: int, unit: str) -> list[str]:
    """Build the full sentence as individual character tokens."""
    result_dt = _add_time(dt, n, unit)
    raw = (
        "The current date is "
        + "".join(_format_time(dt))
        + ". In " + str(n) + " " + unit
        + ", the date will be "
        + "".join(_format_time(result_dt))
        + "."
    )
    return list(raw)


# ---------------------------------------------------------------------------
# Training data generation
# ---------------------------------------------------------------------------

def generate_training_data(
    seed_year_start: int = 1980,
    seed_year_end: int = 2050,
) -> list[list[str]]:
    """Generate multi-scale calendar training observations.

    Returns a list of token lists, one per observation.
    """
    observations: list[list[str]] = []

    # --- Scale 1: Seconds (reveals 60-period) ---
    # 200 consecutive seconds starting at an arbitrary time.
    base = datetime(2000, 6, 15, 14, 58, 0)  # near minute boundary
    for i in range(200):
        dt = base + timedelta(seconds=i)
        observations.append(_make_observation(dt, 1, "seconds"))

    # --- Scale 2: Minutes (reveals 60-period, hour carry) ---
    # 200 consecutive minutes starting near hour boundary.
    base = datetime(2000, 6, 15, 22, 0, 0)  # near midnight
    for i in range(200):
        dt = base + timedelta(minutes=i)
        observations.append(_make_observation(dt, 1, "minutes"))

    # --- Scale 3: Hours (reveals 24-period, day carry) ---
    # 100 consecutive hours.
    base = datetime(2000, 6, 28, 0, 0, 0)  # near month boundary
    for i in range(100):
        dt = base + timedelta(hours=i)
        observations.append(_make_observation(dt, 1, "hours"))

    # --- Scale 4: Days (reveals month-dependent period) ---
    # Walk through every day of 4 different years to see all month lengths.
    # Include a leap year and a non-leap year.
    for year in [1999, 2000, 2023, 2024]:  # 2000 and 2024 are leap years
        dt = datetime(year, 1, 1, 12, 0, 0)
        while dt.year == year:
            observations.append(_make_observation(dt, 1, "days"))
            dt += timedelta(days=1)

    # --- Scale 5: Months (reveals 12-period, year carry) ---
    # Walk through every month from 1980 to 2050 on the 1st.
    for year in range(seed_year_start, seed_year_end + 1):
        for month in range(1, 13):
            dt = datetime(year, month, 1, 0, 0, 0)
            observations.append(_make_observation(dt, 1, "months"))

    # --- Scale 6: Years (reveals leap year pattern) ---
    # Every year transition from 1980 to 2050.
    for year in range(seed_year_start, seed_year_end + 1):
        dt = datetime(year, 1, 1, 0, 0, 0)
        observations.append(_make_observation(dt, 1, "years"))
        # Also show Feb 28/29 transitions for leap year detection.
        dt_feb = datetime(year, 2, 28, 0, 0, 0)
        observations.append(_make_observation(dt_feb, 1, "days"))

    # --- Multi-step examples (N > 1) ---
    # Show "In 2 hours", "In 3 days", "In 2 months", "In 5 years"
    # to demonstrate that the operation scales.
    for n in [2, 3, 5, 7, 10]:
        for unit in ["seconds", "minutes", "hours", "days", "months", "years"]:
            # Pick a few dates.
            for year in [1990, 2000, 2010, 2024]:
                for month, day in [(1, 15), (2, 28), (6, 30), (12, 31)]:
                    try:
                        dt = datetime(year, month, day, 13, 30, 45)
                        observations.append(_make_observation(dt, n, unit))
                    except ValueError:
                        pass  # invalid date (e.g., Feb 30)

    return observations


# ---------------------------------------------------------------------------
# Test data generation
# ---------------------------------------------------------------------------

def generate_test_data() -> list[tuple[list[str], list[str]]]:
    """Generate test cases with expected answers.

    Returns list of (input_tokens, expected_output_time_and_date).
    The expected output is just the [SS:MM:HH] [MM/DD/YYYY] part.
    """
    tests: list[tuple[list[str], list[str]]] = []

    def _add_test(dt: datetime, n: int, unit: str):
        tokens = _make_observation(dt, n, unit)
        result_dt = _add_time(dt, n, unit)
        expected = _format_time(result_dt)
        tests.append((tokens, expected))

    # --- Basic wraps ---
    # Second wrap: 59 -> 00
    _add_test(datetime(2025, 3, 15, 10, 30, 59), 1, "seconds")
    # Minute wrap: 59 -> 00, hour increments
    _add_test(datetime(2025, 3, 15, 10, 59, 0), 1, "minutes")
    # Hour wrap: 23 -> 00, day increments
    _add_test(datetime(2025, 3, 15, 23, 0, 0), 1, "hours")

    # --- Month-dependent day wrap ---
    # Jan 31 + 1 day = Feb 1
    _add_test(datetime(2025, 1, 31, 12, 0, 0), 1, "days")
    # Feb 28 + 1 day = Mar 1 (non-leap)
    _add_test(datetime(2025, 2, 28, 12, 0, 0), 1, "days")
    # Feb 28 + 1 day = Feb 29 (leap year 2024)
    _add_test(datetime(2024, 2, 28, 12, 0, 0), 1, "days")
    # Feb 29 + 1 day = Mar 1 (leap year)
    _add_test(datetime(2024, 2, 29, 12, 0, 0), 1, "days")

    # --- Month addition with day clamping ---
    # Jan 31 + 1 month = Feb 28 (non-leap)
    _add_test(datetime(2025, 1, 31, 12, 0, 0), 1, "months")
    # Jan 31 + 1 month = Feb 29 (leap year 2024)
    _add_test(datetime(2024, 1, 31, 12, 0, 0), 1, "months")
    # Mar 31 + 1 month = Apr 30
    _add_test(datetime(2025, 3, 31, 12, 0, 0), 1, "months")

    # --- Year addition with leap year ---
    # Feb 29 2024 + 1 year = Feb 28 2025
    _add_test(datetime(2024, 2, 29, 12, 0, 0), 1, "years")
    # Feb 28 2023 + 1 year = Feb 28 2024
    _add_test(datetime(2023, 2, 28, 12, 0, 0), 1, "years")

    # --- Multi-step ---
    # In 2 hours from 23:00 = 01:00 next day
    _add_test(datetime(2025, 3, 15, 23, 0, 0), 2, "hours")
    # In 2 months from Nov 30 = Jan 30
    _add_test(datetime(2025, 11, 30, 12, 0, 0), 2, "months")
    # In 5 years from 2024 = 2029
    _add_test(datetime(2024, 6, 15, 12, 0, 0), 5, "years")

    # --- Generalization: dates outside training range ---
    # Year 2055 (beyond training range 1980-2050)
    _add_test(datetime(2050, 12, 31, 23, 59, 59), 1, "seconds")
    # Large step: In 10 months
    _add_test(datetime(2025, 6, 15, 12, 0, 0), 10, "months")

    return tests
