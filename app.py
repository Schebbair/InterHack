from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Iterable
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd


RAW_DATA_PATH = Path("Datasets.xlsx")
TREATED_DATA_PATH = Path("dataset_uploader") / "treatedData.xlsx"
DATA_PATH = TREATED_DATA_PATH if TREATED_DATA_PATH.exists() else RAW_DATA_PATH
MODEL_PATH = Path("snn") / "snn_model"
ERRORS_PATH = Path("outputs") / "prediction_errors.csv"
PREDICTIONS_PATH = Path("outputs") / "predictions.csv"
SIGNALS_PATH = Path("outputs") / "signals.csv"
SIGNALS_JSON_PATH = Path("outputs") / "signals.json"

FAMILIES = ["Familia C1", "Familia C2", "Familia T1", "Familia T2"]
TYPE_1_MIN_PREDICTION = 1.0
SIGNAL_TYPE_1 = "type_1_model_prediction"
SIGNAL_TYPE_2 = "type_2_scp_client_family"
SIGNAL_TYPE_3 = "type_3_scp_family"
SIGNAL_TYPE_4 = "type_4_promiscuous_restock"
SIGNAL_TYPE_5 = "type_1_unfulfilled_prediction_control"

try:
    from SPC import SPC, SCP_familia
except ImportError:
    SPC = None
    SCP_familia = None

try:
    from signal_type4 import P_weights, days_until_exhaust
except ImportError:
    P_weights = {}
    days_until_exhaust = None

try:
    from alertas_indiv import generar_alertes_individuals
except ImportError:
    generar_alertes_individuals = None


@dataclass(frozen=True)
class PipelineConfig:
    data_path: Path = DATA_PATH
    model_path: Path = MODEL_PATH
    errors_path: Path = ERRORS_PATH
    predictions_path: Path = PREDICTIONS_PATH
    signals_path: Path = SIGNALS_PATH
    signals_json_path: Path = SIGNALS_JSON_PATH
    families: tuple[str, ...] = tuple(FAMILIES)
    history_periods: int = 60


class SNNPredictor:
    """Adapter for the SNN implementation in snn/snn.py.

    The SNN expects a temporal tensor shaped as:
    (time_steps, batch_size, 5)

    Per time step the five inputs are:
    [family_1_history, family_2_history, family_3_history, family_4_history, month_norm]
    """

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        families: Iterable[str] = FAMILIES,
        snn_module_path: Path = Path("snn") / "snn.py",
    ) -> None:
        self.model_path = Path(model_path)
        self.families = list(families)
        self.snn_module_path = Path(snn_module_path)
        self.model = None
        self.scaler = None
        self.device = None

    def _load_snn_class(self) -> Any:
        module_path = self.snn_module_path
        if not module_path.exists():
            raise FileNotFoundError(f"SNN implementation not found at {module_path}.")

        spec = importlib.util.spec_from_file_location("inibsa_snn", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load SNN module from {module_path}.")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.SNN

    def _torch_load_checkpoint(self, torch_module: Any) -> Any:
        def torch_load(source: Any) -> Any:
            try:
                return torch_module.load(source, map_location=self.device, weights_only=False)
            except TypeError:
                return torch_module.load(source, map_location=self.device)

        if self.model_path.is_dir():
            if not (self.model_path / "data.pkl").exists():
                raise FileNotFoundError(
                    f"{self.model_path} is a directory but does not contain data.pkl."
                )

            buffer = BytesIO()
            with ZipFile(buffer, mode="w", compression=ZIP_DEFLATED) as archive:
                for file_path in self.model_path.rglob("*"):
                    if file_path.is_file():
                        archive_name = (
                            Path(self.model_path.name)
                            / file_path.relative_to(self.model_path)
                        ).as_posix()
                        archive.write(file_path, archive_name)
            buffer.seek(0)
            return torch_load(buffer)

        return torch_load(self.model_path)

    def load(self) -> "SNNPredictor":
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "The SNN model requires torch and snntorch. Install those dependencies "
                "or pass a preloaded model object to run_pipeline(model=...)."
            ) from exc

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"SNN checkpoint not found at {self.model_path}. "
                "Expected a torch checkpoint path, for example snn/snn_model."
            )

        SNN = self._load_snn_class()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SNN(num_inputs=5, num_outputs=len(self.families)).to(self.device)

        checkpoint = self._torch_load_checkpoint(torch)
        if isinstance(checkpoint, dict) and "families" in checkpoint:
            checkpoint_families = list(checkpoint["families"])
            if checkpoint_families != self.families:
                raise ValueError(
                    f"Checkpoint families {checkpoint_families} do not match {self.families}."
                )

        if hasattr(checkpoint, "eval") and callable(checkpoint):
            self.model = checkpoint.to(self.device)
            self.model.eval()
            return self

        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
            self.scaler = checkpoint.get("scaler", checkpoint.get("y_scaler"))

        self.model.load_state_dict(state_dict)
        self.model.eval()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            self.load()

        import torch

        sequences = []
        history_cols = [f"historial_{family}" for family in self.families]
        for _, row in X.iterrows():
            histories = [np.asarray(row[col], dtype=float) for col in history_cols]
            max_len = max(len(history) for history in histories)
            family_matrix = []
            for history in histories:
                if len(history) < max_len:
                    history = np.pad(history, (max_len - len(history), 0), constant_values=0)
                family_matrix.append(history[-max_len:])

            sequence = np.stack(family_matrix, axis=1)
            month_norm = np.full((sequence.shape[0], 1), float(row["mes_actual"]) / 12.0)
            sequences.append(np.hstack([sequence, month_norm]))

        batch = torch.tensor(np.asarray(sequences), dtype=torch.float32, device=self.device)
        batch = batch.permute(1, 0, 2)

        with torch.no_grad():
            predictions = self.model(batch).detach().cpu().numpy()

        if self.scaler is not None:
            predictions = self.scaler.inverse_transform(predictions)

        return np.maximum(0, predictions)


def load_model(
    model_path: Path = MODEL_PATH,
    families: Iterable[str] = FAMILIES,
) -> Any:
    return SNNPredictor(model_path=model_path, families=families).load()


def clean_inputs(data_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ventas = pd.read_excel(data_path, sheet_name="Ventas")
    productos = pd.read_excel(data_path, sheet_name="Productos")

    ventas.columns = ["num_fact", "fecha", "id_cliente", "id_producto", "unidades", "valor"]
    ventas["fecha"] = pd.to_datetime(ventas["fecha"], errors="coerce")
    ventas["id_cliente"] = pd.to_numeric(ventas["id_cliente"], errors="coerce").astype("Int64")
    ventas["id_producto"] = pd.to_numeric(ventas["id_producto"], errors="coerce").astype("Int64")
    ventas["valor"] = pd.to_numeric(ventas["valor"], errors="coerce").fillna(0)
    ventas["unidades"] = pd.to_numeric(ventas["unidades"], errors="coerce").fillna(0)

    productos.columns = ["id_producto", "bloque", "categoria", "familia"]
    productos["id_producto"] = pd.to_numeric(productos["id_producto"], errors="coerce").astype("Int64")
    productos["familia"] = productos["familia"].astype("string").str.strip()

    ventas = ventas.merge(productos, on="id_producto", how="left", validate="many_to_one")
    ventas = ventas.dropna(subset=["fecha", "id_cliente", "familia"])
    return ventas, productos


def load_potential(data_path: Path) -> pd.DataFrame:
    potencial = pd.read_excel(data_path, sheet_name="Potencial")
    potencial = potencial.rename(
        columns={
            "Id.Cliente": "id_cliente",
            "Id. Cliente": "id_cliente",
            "Familia": "familia_potencial",
            "Categoria Productos": "categoria",
            "Categoria_H": "categoria",
            "Potencial_anual": "potencial_anual",
            "REAL POTENCIAL": "potencial_anual",
        }
    )
    potencial = potencial[["id_cliente", "familia_potencial", "categoria", "potencial_anual"]]
    potencial["id_cliente"] = pd.to_numeric(potencial["id_cliente"], errors="coerce").astype("Int64")
    potencial["categoria"] = potencial["categoria"].astype("string").str.strip()
    potencial["potencial_anual"] = pd.to_numeric(
        potencial["potencial_anual"],
        errors="coerce",
    ).fillna(0)
    return potencial.dropna(subset=["id_cliente", "categoria"])


def load_promiscuous_clients(data_path: Path) -> pd.DataFrame:
    xl = pd.ExcelFile(data_path)
    sheet = next((name for name in xl.sheet_names if name.lower() == "promiscuos"), None)
    if sheet is None:
        return pd.DataFrame(columns=["id_cliente", "categoria"])

    promiscuos = pd.read_excel(data_path, sheet_name=sheet)
    promiscuos = promiscuos.rename(
        columns={
            "Id.Cliente": "id_cliente",
            "Id. Cliente": "id_cliente",
            "Categoria Productos": "categoria",
            "Categoria_H": "categoria",
        }
    )
    promiscuos = promiscuos[["id_cliente", "categoria"]].copy()
    promiscuos["id_cliente"] = pd.to_numeric(promiscuos["id_cliente"], errors="coerce").astype("Int64")
    promiscuos["categoria"] = promiscuos["categoria"].astype("string").str.strip()
    return promiscuos.dropna(subset=["id_cliente", "categoria"]).drop_duplicates()


def monthly_family_sales(ventas: pd.DataFrame, families: Iterable[str]) -> pd.DataFrame:
    ventas_pos = ventas[(ventas["unidades"] > 0) & (ventas["valor"] > 0)].copy()
    ventas_pos["month"] = ventas_pos["fecha"].dt.to_period("M")

    monthly = (
        ventas_pos.groupby(["id_cliente", "month", "familia"], as_index=False)["unidades"]
        .sum()
        .pivot_table(
            index=["id_cliente", "month"],
            columns="familia",
            values="unidades",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )

    for family in families:
        if family not in monthly.columns:
            monthly[family] = 0.0

    return monthly[["id_cliente", "month", *families]].sort_values(["id_cliente", "month"])


def build_model_inputs(
    monthly: pd.DataFrame,
    run_month: pd.Period,
    families: Iterable[str],
    history_periods: int,
) -> pd.DataFrame:
    """Build one row per client with five model inputs.

    The five inputs are:
    1. purchase history for family 1,
    2. purchase history for family 2,
    3. purchase history for family 3,
    4. purchase history for family 4,
    5. current month.
    """
    families = list(families)
    rows = []

    for client_id, client_monthly in monthly.groupby("id_cliente"):
        client_monthly = client_monthly.set_index("month").sort_index()
        history_index = pd.period_range(
            end=run_month - 1,
            periods=history_periods,
            freq="M",
        )
        history = client_monthly.reindex(history_index, fill_value=0)

        row = {
            "id_cliente": client_id,
            "month": run_month,
            "mes_actual": run_month.month,
        }
        for family in families:
            row[f"historial_{family}"] = history[family].astype(float).tolist()
        rows.append(row)

    return pd.DataFrame(rows)


def model_features_for_prediction(model_inputs: pd.DataFrame, families: Iterable[str]) -> pd.DataFrame:
    """Return the model-facing feature frame.

    Keep this adapter stable for the teammate who owns the model. If their model
    expects a numpy array, nested lists, extra date encoding, or different names,
    change only this function.
    """
    feature_cols = [f"historial_{family}" for family in families] + ["mes_actual"]
    return model_inputs[feature_cols]


def predict_client_family_sales(
    model: Any,
    model_inputs: pd.DataFrame,
    families: Iterable[str],
) -> pd.DataFrame:
    families = list(families)
    X = model_features_for_prediction(model_inputs, families)
    raw_predictions = model.predict(X)
    predictions = np.asarray(raw_predictions, dtype=float)

    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, len(families))

    if predictions.shape != (len(model_inputs), len(families)):
        raise ValueError(
            "Model must return one prediction per client and one output per family. "
            f"Expected {(len(model_inputs), len(families))}, got {predictions.shape}."
        )

    wide = model_inputs[["id_cliente", "month"]].copy()
    for idx, family in enumerate(families):
        wide[family] = predictions[:, idx]

    return wide.melt(
        id_vars=["id_cliente", "month"],
        value_vars=families,
        var_name="familia",
        value_name="prediccion",
    )


def signal_type_1_from_predictions(
    predictions: pd.DataFrame,
    min_prediction: float = TYPE_1_MIN_PREDICTION,
) -> pd.DataFrame:
    signals = predictions[predictions["prediccion"] >= min_prediction].copy()
    signals["signal_type"] = SIGNAL_TYPE_1
    signals["signal"] = signals.apply(
        lambda row: (
            f"Esta semana el cliente {row['id_cliente']} podria comprar "
            f"{row['prediccion']:.0f} unidades de {row['familia']}."
        ),
        axis=1,
    )
    signals["urgency"] = pd.NA
    return signals[["signal_type", "id_cliente", "familia", "month", "prediccion", "urgency", "signal"]]


def urgency_from_signal(signal: Any, default: str | None = None) -> str | None:
    if not isinstance(signal, str):
        return default
    upper = signal.upper()
    if "ALERTA URGENTE" in upper or "URGENTE" in upper:
        return "ALERTA URGENTE"
    if "ALERTA" in upper:
        return "ALERTA"
    return default


def apply_signal_urgency(signals: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return signals
    out = signals.copy()
    if "urgency" not in out.columns:
        out["urgency"] = pd.NA
    out["urgency"] = out.apply(
        lambda row: (
            pd.NA
            if row.get("signal_type") == SIGNAL_TYPE_1
            else row.get("urgency")
            if pd.notna(row.get("urgency"))
            else urgency_from_signal(row.get("signal"), default="ALERTA")
        ),
        axis=1,
    )
    return out


def category_weight_key(category: str) -> str:
    return str(category).strip().lower().replace(" ", "_")


def build_signal_type_4(
    ventas: pd.DataFrame,
    potencial: pd.DataFrame,
    promiscuous_clients: pd.DataFrame,
    run_date: pd.Timestamp,
    horizon_days: int = 7,
    promiscuous_window_days: int = 180,
) -> pd.DataFrame:
    """Detect promiscuous commodity clients one week before expected restock.

    The dataset uploader owns promiscuity detection and writes the Promiscuos
    sheet. Here we only investigate those known promiscuous client-categories,
    estimate when the last purchase will be exhausted, then emit the alert
    during the week that starts seven days before that date.
    """
    if days_until_exhaust is None:
        raise ImportError("signal_type4.py must expose days_until_exhaust().")
    if promiscuous_clients.empty:
        return pd.DataFrame()

    run_date = pd.Timestamp(run_date).normalize()
    horizon_end = run_date + pd.Timedelta(days=horizon_days - 1)
    ventas_pos = ventas[(ventas["unidades"] > 0) & (ventas["valor"] > 0)].copy()
    recent_start = run_date - pd.Timedelta(days=promiscuous_window_days)
    recent = ventas_pos[
        (ventas_pos["bloque"] == "Commodities")
        & (ventas_pos["fecha"] >= recent_start)
    ].copy()

    if recent.empty:
        return pd.DataFrame()

    known_promiscuous = promiscuous_clients.copy()
    known_promiscuous["categoria"] = known_promiscuous["categoria"].astype("string").str.strip()
    recent = recent.merge(known_promiscuous, on=["id_cliente", "categoria"], how="inner")
    if recent.empty:
        return pd.DataFrame()

    promiscuous = recent.groupby(["id_cliente", "categoria"], as_index=False).agg(
        productos_distintos=("id_producto", "nunique"),
        valor_reciente=("valor", "sum"),
        pedidos_recientes=("num_fact", "nunique"),
    )
    top_product = (
        recent.groupby(["id_cliente", "categoria", "id_producto"], as_index=False)["valor"]
        .sum()
        .sort_values("valor", ascending=False)
        .groupby(["id_cliente", "categoria"], as_index=False)
        .first()
        .rename(columns={"valor": "valor_top_producto", "id_producto": "producto_top"})
    )
    promiscuous = promiscuous.merge(top_product, on=["id_cliente", "categoria"], how="left")
    promiscuous["share_top_producto"] = (
        promiscuous["valor_top_producto"] / promiscuous["valor_reciente"].replace(0, np.nan)
    )
    promiscuous = promiscuous[promiscuous["valor_reciente"] > 0].copy()

    if promiscuous.empty:
        return pd.DataFrame()

    potencial_categoria = (
        potencial.groupby(["id_cliente", "categoria"], as_index=False)["potencial_anual"]
        .sum()
    )
    promiscuous = promiscuous.merge(
        potencial_categoria,
        on=["id_cliente", "categoria"],
        how="left",
    )
    promiscuous["potencial_anual"] = promiscuous["potencial_anual"].fillna(0)

    last_purchase = (
        recent.sort_values("fecha")
        .groupby(["id_cliente", "categoria"], as_index=False)
        .tail(1)[
            [
                "id_cliente",
                "categoria",
                "familia",
                "id_producto",
                "fecha",
                "valor",
                "unidades",
            ]
        ]
        .rename(
            columns={
                "familia": "ultima_familia",
                "id_producto": "ultimo_producto",
                "fecha": "fecha_ultima_compra",
                "valor": "valor_ultima_compra",
                "unidades": "unidades_ultima_compra",
            }
        )
    )
    promiscuous = promiscuous.merge(last_purchase, on=["id_cliente", "categoria"], how="left")

    rows = []
    for _, row in promiscuous.iterrows():
        weight_key = category_weight_key(row["categoria"])
        weights = P_weights.get(weight_key)
        if not weights or row["potencial_anual"] <= 0 or row["valor_ultima_compra"] <= 0:
            continue

        exhaust_date = pd.Timestamp(
            days_until_exhaust(
                row["fecha_ultima_compra"],
                float(row["valor_ultima_compra"]),
                float(row["potencial_anual"]),
                weights,
            )
        ).normalize()
        alert_date = exhaust_date - pd.Timedelta(days=7)

        if not (run_date <= alert_date <= horizon_end):
            continue

        signal = (
            f"Cliente promiscuo {row['id_cliente']} cerca de reabastecerse en "
            f"{row['categoria']}: ultima compra el {row['fecha_ultima_compra'].date()} "
            f"por {row['valor_ultima_compra']:.2f}; agotamiento estimado {exhaust_date.date()}. "
            "Conviene contactar una semana antes para evitar compra a competencia."
        )
        rows.append(
            {
                "signal_type": SIGNAL_TYPE_4,
                "id_cliente": row["id_cliente"],
                "familia": row["ultima_familia"],
                "categoria": row["categoria"],
                "month": pd.Period(alert_date, freq="M"),
                "signal_date": alert_date,
                "restock_date": exhaust_date,
                "prediccion": pd.NA,
                "urgency": "ALERTA URGENTE",
                "signal": signal,
                "metadata": {
                    "productos_distintos": int(row["productos_distintos"]),
                    "share_top_producto": round(float(row["share_top_producto"]), 4),
                    "valor_reciente": round(float(row["valor_reciente"]), 2),
                    "potencial_anual": round(float(row["potencial_anual"]), 2),
                    "ultimo_producto": int(row["ultimo_producto"]),
                    "unidades_ultima_compra": float(row["unidades_ultima_compra"]),
                },
            }
        )

    return pd.DataFrame(rows)


def load_errors(errors_path: Path = ERRORS_PATH) -> pd.DataFrame:
    columns = ["id_cliente", "familia", "month", "real", "prediccion", "error"]
    if not errors_path.exists():
        return pd.DataFrame(columns=columns)

    errors = pd.read_csv(errors_path)
    errors["month"] = pd.PeriodIndex(errors["month"], freq="M")
    return errors[columns]


def ensure_month_period(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "month" in df.columns and not df.empty:
        df["month"] = pd.PeriodIndex(df["month"], freq="M")
    return df


def load_predictions(predictions_path: Path = PREDICTIONS_PATH) -> pd.DataFrame:
    columns = ["id_cliente", "month", "familia", "prediccion"]
    if not predictions_path.exists():
        return pd.DataFrame(columns=columns)

    predictions = pd.read_csv(predictions_path)
    predictions["month"] = pd.PeriodIndex(predictions["month"], freq="M")
    return predictions[columns]


def append_predictions(
    existing_predictions: pd.DataFrame,
    current_predictions: pd.DataFrame,
) -> pd.DataFrame:
    existing_predictions = ensure_month_period(existing_predictions)
    current_predictions = ensure_month_period(current_predictions)
    combined = pd.concat([existing_predictions, current_predictions], ignore_index=True)
    combined = ensure_month_period(combined)
    return combined.drop_duplicates(["id_cliente", "familia", "month"], keep="last")


def update_errors(
    existing_errors: pd.DataFrame,
    prediction_history: pd.DataFrame,
    actuals: pd.DataFrame,
) -> pd.DataFrame:
    """Append errors for predictions whose real sales are already known.

    Error definition: real value - previous prediction. In a daily run this
    usually closes errors for older predicted months, while the new future
    prediction remains only in prediction history until real sales arrive.
    """
    existing_errors = ensure_month_period(existing_errors)
    prediction_history = ensure_month_period(prediction_history)
    actuals = ensure_month_period(actuals)

    comparable = prediction_history.merge(
        actuals,
        on=["id_cliente", "month", "familia"],
        how="inner",
    )
    comparable["error"] = comparable["real"] - comparable["prediccion"]

    new_errors = comparable[["id_cliente", "familia", "month", "real", "prediccion", "error"]]
    if existing_errors.empty:
        return new_errors.drop_duplicates(["id_cliente", "familia", "month"], keep="last")

    combined = pd.concat([existing_errors, new_errors], ignore_index=True)
    combined = ensure_month_period(combined)
    return combined.drop_duplicates(["id_cliente", "familia", "month"], keep="last")


def actuals_for_month(monthly: pd.DataFrame, families: Iterable[str]) -> pd.DataFrame:
    families = list(families)
    return monthly.melt(
        id_vars=["id_cliente", "month"],
        value_vars=families,
        var_name="familia",
        value_name="real",
    )


def latest_non_negative_errors(errors: pd.DataFrame) -> pd.DataFrame:
    if errors.empty:
        return errors

    latest_idx = errors.sort_values("month").groupby(["id_cliente", "familia"])["month"].idxmax()
    latest = errors.loc[latest_idx]
    return latest[latest["error"] >= 0]


def scp_client_family(id_cliente: Any, familia: str, error_history: pd.Series) -> str | None:
    if SPC is None:
        raise ImportError("SPC.py must expose SPC(lista_errores, id_cliente, id_producto).")
    return SPC(error_history.tolist(), id_cliente, familia)


def scp_family(familia: str, aggregated_error_history: pd.Series) -> str | None:
    if SCP_familia is None:
        raise ImportError("SPC.py must expose SCP_familia(agregado_lista_errores, id_producto).")
    return SCP_familia(aggregated_error_history.tolist(), familia)


def build_signal_type_2(
    errors: pd.DataFrame,
    scp_func: Callable[[Any, str, pd.Series], str],
) -> pd.DataFrame:
    """Run SCP only when the latest client-family error is zero or positive."""
    allowed = latest_non_negative_errors(errors)
    rows = []

    for _, latest in allowed.iterrows():
        history = (
            errors[
                (errors["id_cliente"] == latest["id_cliente"])
                & (errors["familia"] == latest["familia"])
            ]
            .sort_values("month")["error"]
            .reset_index(drop=True)
        )
        signal = scp_func(latest["id_cliente"], latest["familia"], history)
        if signal:
            rows.append(
                {
                    "signal_type": SIGNAL_TYPE_2,
                    "id_cliente": latest["id_cliente"],
                    "familia": latest["familia"],
                    "month": latest["month"],
                    "urgency": urgency_from_signal(signal),
                    "signal": signal,
                }
            )

    return pd.DataFrame(rows)


def build_signal_type_3(
    errors: pd.DataFrame,
    scp_prod_func: Callable[[str, pd.Series], str],
) -> pd.DataFrame:
    rows = []

    for family, family_errors in errors.groupby("familia"):
        aggregated_history = (
            family_errors.groupby("month", as_index=True)["error"]
            .sum()
            .sort_index()
            .reset_index(drop=True)
        )
        signal = scp_prod_func(family, aggregated_history)
        if signal:
            rows.append(
                {
                    "signal_type": SIGNAL_TYPE_3,
                    "id_cliente": pd.NA,
                    "familia": family,
                    "month": family_errors["month"].max(),
                    "urgency": urgency_from_signal(signal),
                    "signal": signal,
                }
            )

    return pd.DataFrame(rows)


def build_signal_type_5(errors: pd.DataFrame) -> pd.DataFrame:
    """Alert when a previous type-1 prediction was not fulfilled.

    Type 1 signals are the model predictions. Once real sales arrive for a
    predicted client-family-month, this function compares real vs predicted and
    delegates the alert text/urgency to alertas_indiv.py.
    """
    if generar_alertes_individuals is None:
        raise ImportError(
            "alertas_indiv.py must expose generar_alertes_individuals()."
        )
    if errors.empty:
        return pd.DataFrame()

    rows = []
    comparable = errors[errors["real"] != errors["prediccion"]].copy()
    for _, row in comparable.iterrows():
        signal = generar_alertes_individuals(
            row["id_cliente"],
            row["familia"],
            row["familia"],
            row["prediccion"],
            row["real"],
        )
        if "URGENTE" not in signal.upper():
            signal = signal.replace("ALERTA", "ALERTA URGENTE", 1)
        rows.append(
            {
                "signal_type": SIGNAL_TYPE_5,
                "id_cliente": row["id_cliente"],
                "familia": row["familia"],
                "month": row["month"],
                "real": row["real"],
                "prediccion": row["prediccion"],
                "error": row["error"],
                "urgency": "ALERTA URGENTE",
                "signal": signal,
                "metadata": {
                    "source_signal_type": SIGNAL_TYPE_1,
                    "control": "type_1_prediction_not_fulfilled",
                },
            }
        )

    return pd.DataFrame(rows)


def json_safe(value: Any) -> Any:
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, pd.Period):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def frontend_signals_payload(signals: pd.DataFrame) -> dict[str, Any]:
    if signals.empty:
        return {"signals": [], "by_type": {}, "calendar": {}}

    safe = signals.copy()
    for column in ["month", "signal_date", "restock_date"]:
        if column not in safe.columns:
            safe[column] = pd.NA
    if "categoria" not in safe.columns:
        safe["categoria"] = pd.NA
    if "metadata" not in safe.columns:
        safe["metadata"] = pd.NA
    if "urgency" not in safe.columns:
        safe["urgency"] = pd.NA

    records = []
    for idx, row in safe.reset_index(drop=True).iterrows():
        record = {
            "id": f"signal-{idx + 1}",
            "type": json_safe(row.get("signal_type")),
            "clientId": json_safe(row.get("id_cliente")),
            "family": json_safe(row.get("familia")),
            "category": json_safe(row.get("categoria")),
            "month": json_safe(row.get("month")),
            "urgency": json_safe(row.get("urgency")),
            "signalDate": json_safe(row.get("signal_date")),
            "restockDate": json_safe(row.get("restock_date")),
            "prediction": json_safe(row.get("prediccion")),
            "message": json_safe(row.get("signal")),
            "metadata": json_safe(row.get("metadata")) or {},
        }
        records.append(record)

    by_type: dict[str, list[dict[str, Any]]] = {}
    calendar: dict[str, list[dict[str, Any]]] = {}
    by_urgency: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_type.setdefault(record["type"], []).append(record)
        if record["urgency"]:
            by_urgency.setdefault(record["urgency"], []).append(record)
        if record["signalDate"]:
            calendar.setdefault(record["signalDate"], []).append(record)

    return {
        "signals": records,
        "by_type": by_type,
        "by_urgency": by_urgency,
        "calendar": calendar,
        "counts": {
            "total": len(records),
            "by_type": {signal_type: len(items) for signal_type, items in by_type.items()},
            "by_urgency": {urgency: len(items) for urgency, items in by_urgency.items()},
            "calendar_days": len(calendar),
        },
    }


def save_outputs(
    signals: pd.DataFrame,
    errors: pd.DataFrame,
    predictions: pd.DataFrame,
    config: PipelineConfig,
) -> None:
    import json

    config.signals_path.parent.mkdir(parents=True, exist_ok=True)
    signals_to_save = signals.copy()
    errors_to_save = errors.copy()
    predictions_to_save = predictions.copy()
    for df in [signals_to_save, errors_to_save, predictions_to_save]:
        if "month" in df.columns:
            df["month"] = df["month"].astype(str)

    signals_to_save.to_csv(config.signals_path, index=False)
    errors_to_save.to_csv(config.errors_path, index=False)
    predictions_to_save.to_csv(config.predictions_path, index=False)
    config.signals_json_path.write_text(
        json.dumps(frontend_signals_payload(signals), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def placeholder_scp(id_cliente: Any, familia: str, error_history: pd.Series) -> str:
    return (
        f"SCP pendiente para cliente {id_cliente}, familia {familia}. "
        f"Historial de errores: {error_history.tolist()}."
    )


def placeholder_scp_prod(familia: str, aggregated_error_history: pd.Series) -> str:
    return (
        f"SCP_Prod pendiente para familia {familia}. "
        f"Historial agregado de errores: {aggregated_error_history.tolist()}."
    )


def run_pipeline(
    config: PipelineConfig = PipelineConfig(),
    model: Any | None = None,
    scp_func: Callable[[Any, str, pd.Series], str | None] = scp_client_family,
    scp_prod_func: Callable[[str, pd.Series], str | None] = scp_family,
    run_month: pd.Period | None = None,
    run_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    ventas, _ = clean_inputs(config.data_path)
    potencial = load_potential(config.data_path)
    promiscuous_clients = load_promiscuous_clients(config.data_path)
    monthly = monthly_family_sales(ventas, config.families)

    if run_month is None:
        run_month = monthly["month"].max() + 1
    if run_date is None:
        run_date = monthly["month"].dt.to_timestamp().max() + pd.offsets.MonthBegin(1)
    run_date = pd.Timestamp(run_date).normalize()

    model_inputs = build_model_inputs(
        monthly=monthly,
        run_month=run_month,
        families=config.families,
        history_periods=config.history_periods,
    )

    if model is None:
        model = load_model(config.model_path, config.families)

    predictions = predict_client_family_sales(model, model_inputs, config.families)
    type_1 = signal_type_1_from_predictions(predictions)

    existing_predictions = load_predictions(config.predictions_path)
    prediction_history = append_predictions(existing_predictions, predictions)

    existing_errors = load_errors(config.errors_path)
    actuals = actuals_for_month(monthly, config.families)
    errors = update_errors(existing_errors, prediction_history, actuals)

    type_2 = build_signal_type_2(errors, scp_func)
    type_3 = build_signal_type_3(errors, scp_prod_func)
    type_4 = build_signal_type_4(
        ventas,
        potencial,
        promiscuous_clients,
        run_date=run_date,
    )
    type_5 = build_signal_type_5(errors)

    signals = pd.concat([type_1, type_2, type_3, type_4, type_5], ignore_index=True, sort=False)
    signals = apply_signal_urgency(signals)
    save_outputs(signals, errors, prediction_history, config)
    return signals


if __name__ == "__main__":
    signals_df = run_pipeline()
    print(f"Generated {len(signals_df)} signals at {SIGNALS_PATH}")
