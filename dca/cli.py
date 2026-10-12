"""Command line interface.

    python -m dca fetch-nd --start 2020-01 --end 2026-07 --file-no 37170 -o wells.csv
    python -m dca analyze wells.csv --well 33053090560000 --plot out.png
    python -m dca batch wells.csv -o summary.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from .analysis import analyze_many, analyze_well
from .cleaning import CleaningConfig
from .forecast import ForecastSettings

UNITS = {"oil": ("bbl", "bbl/d"), "gas": ("mcf", "mcf/d"), "water": ("bbl", "bbl/d")}


def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"], dtype={"api": str})
    if "api" in df:
        df["api"] = df["api"].str.zfill(14)
    return df


def _settings(args) -> ForecastSettings:
    return ForecastSettings(
        econ_limit=args.econ_limit,
        horizon_years=args.horizon,
        terminal_decline=None if args.dmin <= 0 else args.dmin,
    )


def _common(p: argparse.ArgumentParser):
    p.add_argument("--phase", choices=list(UNITS), default="oil")
    p.add_argument("--econ-limit", type=float, default=5.0, help="economic limit rate (per day)")
    p.add_argument("--dmin", type=float, default=0.08,
                   help="terminal effective annual decline (0 disables)")
    p.add_argument("--horizon", type=float, default=50.0, help="max forecast years")
    p.add_argument("--b-max", type=float, default=2.0, help="upper bound on hyperbolic b")
    p.add_argument("--min-days", type=int, default=10, help="drop months with fewer producing days")
    p.add_argument("--half-life", type=float, default=None,
                   help="weight recent months more (weight halves every N days back)")


def cmd_fetch_nd(args):
    from .sources.nd_dmr import load_production
    df = load_production(args.start, args.end, apis=args.api, file_nos=args.file_no,
                         pool=args.pool)
    df.to_csv(args.output, index=False)
    print(f"wrote {len(df):,} rows for {df['api'].nunique()} wells to {args.output}")


def cmd_analyze(args):
    df = _load_csv(args.csv)
    if args.well:
        df = df[(df["api"] == args.well.zfill(14)) |
                (df.get("well_name", pd.Series(dtype=str)).str.upper() == args.well.upper())]
    elif "api" in df and df["api"].nunique() > 1:
        sys.exit("multiple wells in file; pick one with --well (or use `batch`)")
    if df.empty:
        sys.exit("no rows for that well")

    well_id = args.well or (df["api"].iloc[0] if "api" in df else Path(args.csv).stem)
    name = df["well_name"].iloc[0] if "well_name" in df else well_id
    vol_unit, rate_unit = UNITS[args.phase]

    a = analyze_well(df, well_id=str(well_id),
                     cleaning=CleaningConfig(phase=args.phase, min_days_on=args.min_days),
                     settings=_settings(args), b_max=args.b_max,
                     recency_half_life=args.half_life)
    print(a.report(vol_unit))
    print()
    print(a.eur_by_model.round(3).to_string())

    if args.forecast_csv:
        a.forecast.to_csv(args.forecast_csv, index=False)
        print(f"\nforecast written to {args.forecast_csv}")
    if args.plot:
        from .plotting import plot_decline, plot_residuals
        plot_decline(a.prepared, a.fits, settings=a.settings, best=a.best.model,
                     title=f"{name} - {args.phase} decline", unit=rate_unit, path=args.plot)
        resid_path = Path(args.plot).with_name(Path(args.plot).stem + "_residuals.png")
        plot_residuals(a.prepared, a.fits, path=resid_path)
        print(f"plots written to {args.plot}, {resid_path}")


def cmd_batch(args):
    df = _load_csv(args.csv)
    summary = analyze_many(df, cleaning=CleaningConfig(phase=args.phase, min_days_on=args.min_days),
                           settings=_settings(args), b_max=args.b_max,
                           recency_half_life=args.half_life)
    summary.to_csv(args.output, index=False)
    ok = summary[summary["error"] == ""]
    print(f"analyzed {len(ok)}/{len(summary)} wells -> {args.output}")
    if len(ok):
        print(ok["best_model"].value_counts().to_string())
        print(f"EUR P90/P50/P10: " + " / ".join(
            f"{ok['eur'].quantile(p):,.0f}" for p in (0.1, 0.5, 0.9)))
    if args.plot and len(ok) >= 3:
        from .plotting import plot_eur_distribution
        plot_eur_distribution(ok, path=args.plot, unit=UNITS[args.phase][0])
        print(f"EUR histogram written to {args.plot}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dca", description="Arps decline curve analysis")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch-nd", help="download ND DMR monthly production")
    p.add_argument("--start", required=True, help="YYYY-MM")
    p.add_argument("--end", required=True, help="YYYY-MM")
    p.add_argument("--api", nargs="*", help="14-digit API numbers")
    p.add_argument("--file-no", nargs="*", type=int, help="NDIC file numbers")
    p.add_argument("--pool", help="e.g. BAKKEN")
    p.add_argument("-o", "--output", required=True)
    p.set_defaults(func=cmd_fetch_nd)

    p = sub.add_parser("analyze", help="fit, forecast and estimate EUR for one well")
    p.add_argument("csv")
    p.add_argument("--well", help="API number or well name")
    p.add_argument("--plot", help="save rate/cumulative plot to this path")
    p.add_argument("--forecast-csv", help="save monthly forecast table")
    _common(p)
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("batch", help="analyze every well in a CSV")
    p.add_argument("csv")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--plot", help="save EUR histogram to this path")
    _common(p)
    p.set_defaults(func=cmd_batch)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
