from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd


ERROR_COLUMNS = ["id_cliente", "familia", "month", "real", "prediccion", "error"]


def ensure_month_period(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "month" in df.columns and not df.empty:
        df["month"] = pd.PeriodIndex(df["month"], freq="M")
    return df


def actuals_for_month(monthly: pd.DataFrame, families: Iterable[str]) -> pd.DataFrame:
    families = list(families)
    return monthly.melt(
        id_vars=["id_cliente", "month"],
        value_vars=families,
        var_name="familia",
        value_name="real",
    )


def update_errors(
    existing_errors: pd.DataFrame,
    prediction_history: pd.DataFrame,
    actuals: pd.DataFrame,
) -> pd.DataFrame:
    """Append errors for predictions whose real sales are already known."""
    existing_errors = ensure_month_period(existing_errors)
    prediction_history = ensure_month_period(prediction_history)
    actuals = ensure_month_period(actuals)

    comparable = prediction_history.merge(
        actuals,
        on=["id_cliente", "month", "familia"],
        how="inner",
    )
    comparable["error"] = comparable["real"] - comparable["prediccion"]

    new_errors = comparable[ERROR_COLUMNS]
    if existing_errors.empty:
        return new_errors.drop_duplicates(["id_cliente", "familia", "month"], keep="last")

    combined = pd.concat([existing_errors, new_errors], ignore_index=True)
    combined = ensure_month_period(combined)
    return combined.drop_duplicates(["id_cliente", "familia", "month"], keep="last")


def _empty_errors() -> pd.DataFrame:
    return pd.DataFrame(columns=ERROR_COLUMNS)


def _load_cached_errors(cache_path: Path) -> pd.DataFrame:
    if not cache_path.exists():
        return _empty_errors()
    cached = pd.read_csv(cache_path)
    return ensure_month_period(cached)


def _save_cached_errors(errors: pd.DataFrame, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached = errors.copy()
    cached["month"] = cached["month"].astype(str)
    cached.to_csv(cache_path, index=False)


def cache_covers_latest_month(
    cached_errors: pd.DataFrame,
    monthly: pd.DataFrame,
    backfill_months: int,
) -> bool:
    if cached_errors.empty or monthly.empty or backfill_months <= 0:
        return False

    expected_months = set(pd.period_range(end=monthly["month"].max(), periods=backfill_months, freq="M"))
    cached_months = set(cached_errors["month"].dropna().unique())
    return expected_months.issubset(cached_months)


def historical_prediction_errors_for_spc(
    model: Any,
    monthly: pd.DataFrame,
    families: Iterable[str],
    history_periods: int,
    backfill_months: int,
    build_model_inputs: Callable[..., pd.DataFrame],
    predict_client_family_sales: Callable[..., pd.DataFrame],
) -> pd.DataFrame:
    """Create retrospective model errors so SPC can run on a cold start.

    These predictions are only for SPC history. They are intentionally kept out
    of type-1 control alerts because they were not actually sent as forecasts.
    """
    if monthly.empty or backfill_months <= 0:
        return _empty_errors()

    months = pd.period_range(end=monthly["month"].max(), periods=backfill_months, freq="M")
    predictions = []
    for month in months:
        model_inputs = build_model_inputs(
            monthly=monthly,
            run_month=month,
            families=families,
            history_periods=history_periods,
        )
        if not model_inputs.empty:
            predictions.append(predict_client_family_sales(model, model_inputs, families))

    if not predictions:
        return _empty_errors()

    historical_predictions = pd.concat(predictions, ignore_index=True)
    return update_errors(_empty_errors(), historical_predictions, actuals_for_month(monthly, families))


def load_or_create_historical_spc_errors(
    model: Any,
    monthly: pd.DataFrame,
    families: Iterable[str],
    history_periods: int,
    backfill_months: int,
    cache_path: Path,
    build_model_inputs: Callable[..., pd.DataFrame],
    predict_client_family_sales: Callable[..., pd.DataFrame],
) -> pd.DataFrame:
    cached_errors = _load_cached_errors(cache_path)
    if cache_covers_latest_month(cached_errors, monthly, backfill_months):
        return cached_errors

    errors = historical_prediction_errors_for_spc(
        model=model,
        monthly=monthly,
        families=families,
        history_periods=history_periods,
        backfill_months=backfill_months,
        build_model_inputs=build_model_inputs,
        predict_client_family_sales=predict_client_family_sales,
    )
    _save_cached_errors(errors, cache_path)
    return errors
