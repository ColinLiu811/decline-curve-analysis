"""Loader for North Dakota DMR (Oil & Gas Division) monthly production reports.

The DMR publishes one Excel workbook per month containing every producing
well in the state:

    https://www.dmr.nd.gov/oilgas/mpr/YYYY_MM.xlsx

Each row is one well-month with oil (bbl), water (bbl), gas (mcf) and the
number of producing days. This module downloads a range of those workbooks,
caches them locally, and stitches them into a long-format production table
for the wells you care about.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

log = logging.getLogger(__name__)

BASE_URL = "https://www.dmr.nd.gov/oilgas/mpr/{year:04d}_{month:02d}.xlsx"
DEFAULT_CACHE = Path.home() / ".cache" / "dca" / "nd_dmr"

# Columns we keep from the raw workbook, renamed to the package's schema.
COLUMN_MAP = {
    "ReportDate": "date",
    "API_WELLNO": "api",
    "FileNo": "file_no",
    "WellName": "well_name",
    "Company": "operator",
    "FieldName": "field",
    "Pool": "pool",
    "County": "county",
    "Oil": "oil_bbl",
    "Gas": "gas_mcf",
    "Wtr": "water_bbl",
    "Days": "days_on",
}


def month_range(start: str, end: str) -> list[pd.Period]:
    """Inclusive list of monthly periods between two 'YYYY-MM' strings."""
    return list(pd.period_range(start=start, end=end, freq="M"))


def download_month(period: pd.Period, cache_dir: Path = DEFAULT_CACHE,
                   session: requests.Session | None = None,
                   timeout: float = 60.0) -> Path:
    """Download a single monthly workbook (skipped if already cached)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"{period.year:04d}_{period.month:02d}.xlsx"
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    url = BASE_URL.format(year=period.year, month=period.month)
    log.info("downloading %s", url)
    http = session or requests
    resp = http.get(url, timeout=timeout)
    resp.raise_for_status()

    tmp = dest.with_suffix(".part")
    tmp.write_bytes(resp.content)
    tmp.replace(dest)
    return dest


def read_month(path: Path) -> pd.DataFrame:
    """Read one cached workbook into the normalized schema."""
    raw = pd.read_excel(path, sheet_name="Oil")
    df = raw[list(COLUMN_MAP)].rename(columns=COLUMN_MAP)
    df["date"] = pd.to_datetime(df["date"])
    df["api"] = df["api"].astype(str).str.zfill(14)
    df["well_name"] = df["well_name"].astype(str).str.split().str.join(" ")
    return df


def load_production(start: str, end: str, *,
                    apis: Iterable[str] | None = None,
                    file_nos: Iterable[int] | None = None,
                    pool: str | None = None,
                    cache_dir: Path = DEFAULT_CACHE) -> pd.DataFrame:
    """Build a monthly production history for a set of ND wells.

    Parameters
    ----------
    start, end:
        Inclusive month range as 'YYYY-MM'.
    apis, file_nos:
        Optional well filters (14-digit API numbers or NDIC file numbers).
    pool:
        Optional pool filter, e.g. "BAKKEN".
    """
    api_set = {str(a).zfill(14) for a in apis} if apis else None
    file_set = {int(f) for f in file_nos} if file_nos else None

    frames = []
    with requests.Session() as session:
        for period in month_range(start, end):
            try:
                path = download_month(period, cache_dir, session=session)
            except requests.HTTPError as exc:
                log.warning("skipping %s: %s", period, exc)
                continue
            df = read_month(path)
            if api_set is not None:
                df = df[df["api"].isin(api_set)]
            if file_set is not None:
                df = df[df["file_no"].isin(file_set)]
            if pool is not None:
                df = df[df["pool"].str.upper() == pool.upper()]
            frames.append(df)

    if not frames:
        return pd.DataFrame(columns=list(COLUMN_MAP.values()))
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["api", "date"]).reset_index(drop=True)


def first_production_month(df: pd.DataFrame) -> pd.Series:
    """Month of first non-zero oil production for each well."""
    producing = df[df["oil_bbl"] > 0]
    return producing.groupby("api")["date"].min()
