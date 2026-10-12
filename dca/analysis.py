"""High-level pipeline: raw monthly production -> fits -> forecast -> EUR."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .cleaning import CleaningConfig, decline_arrays, prepare_decline_data
from .fitting import MODEL_NAMES, FitResult, best_fit, fit_all, recency_weights
from .forecast import EURResult, ForecastSettings, estimate_eur, eur_table, monthly_forecast


@dataclass
class WellAnalysis:
    well_id: str
    prepared: pd.DataFrame
    fits: dict[str, FitResult]
    best: FitResult
    eur: EURResult
    eur_by_model: pd.DataFrame
    forecast: pd.DataFrame
    settings: ForecastSettings

    @property
    def decline_start(self) -> pd.Timestamp:
        return self.prepared.loc[self.prepared["t"] == 0, "date"].iloc[0]

    def report(self, unit: str = "bbl") -> str:
        lines = [f"Well {self.well_id}",
                 f"  decline start {self.decline_start:%Y-%m}, "
                 f"{self.best.n_points} months fitted, "
                 f"last data {self.prepared['date'].iloc[-1]:%Y-%m}",
                 ""]
        lines += ["  " + f.summary() for f in self.fits.values()]
        e = self.eur
        lines += ["",
                  f"  best model (AIC): {self.best.model}",
                  f"  cum to date : {e.cum_to_date:12,.0f} {unit}",
                  f"  remaining   : {e.remaining:12,.0f} {unit}  "
                  f"({e.years_remaining:.1f} yrs, ends at {e.end_reason})",
                  f"  EUR         : {e.eur:12,.0f} {unit}"]
        return "\n".join(lines)


def analyze_well(df: pd.DataFrame, *, well_id: str = "",
                 cleaning: CleaningConfig | None = None,
                 settings: ForecastSettings | None = None,
                 models_to_fit=MODEL_NAMES, b_max: float = 2.0,
                 recency_half_life: float | None = None) -> WellAnalysis:
    """Run the full DCA workflow on one well's monthly production."""
    settings = settings or ForecastSettings()
    prepared = prepare_decline_data(df, cleaning)
    t, q = decline_arrays(prepared)
    if len(t) < 3:
        raise ValueError(f"only {len(t)} usable months after the peak")

    weights = recency_weights(t, recency_half_life) if recency_half_life else None
    fits = fit_all(t, q, models_to_fit, b_bounds=(0.0, b_max), weights=weights)
    best = best_fit(fits)

    t_last = float(prepared["t"].iloc[-1])
    cum = float(prepared["cum"].iloc[-1])
    start = prepared.loc[prepared["t"] == 0, "date"].iloc[0]
    return WellAnalysis(
        well_id=well_id,
        prepared=prepared,
        fits=fits,
        best=best,
        eur=estimate_eur(best, t_last, cum, settings),
        eur_by_model=eur_table(fits, t_last, cum, settings),
        forecast=monthly_forecast(best, start, t_last, cum, settings),
        settings=settings,
    )


def analyze_many(production: pd.DataFrame, id_col: str = "api", **kwargs) -> pd.DataFrame:
    """Analyze every well in a long-format table; returns one summary row per well."""
    rows = []
    for well_id, well_df in production.groupby(id_col):
        try:
            a = analyze_well(well_df, well_id=str(well_id), **kwargs)
        except ValueError as exc:
            rows.append({id_col: well_id, "error": str(exc)})
            continue
        first = well_df.iloc[0]
        rows.append({
            id_col: well_id,
            "well_name": first.get("well_name", ""),
            "operator": first.get("operator", ""),
            "decline_start": a.decline_start,
            "months_fitted": a.best.n_points,
            "best_model": a.best.model,
            "qi": a.best.qi,
            "di_eff_annual": a.best.di_annual_effective,
            "b": a.best.b,
            "rmse_log": a.best.rmse_log,
            "cum_to_date": a.eur.cum_to_date,
            "remaining": a.eur.remaining,
            "eur": a.eur.eur,
            "error": "",
        })
    return pd.DataFrame(rows)
