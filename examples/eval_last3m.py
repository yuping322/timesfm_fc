import os
import sys
import io
import numpy as np
import pandas as pd
import matplotlib

# Use non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error

# Ensure local src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from timesfm import TimesFM_2p5_200M_torch  # noqa: E402
from examples.data import load_oss_stocks  # noqa: E402


def compute_feature_vector(context: np.ndarray, base_pred_price: float) -> np.ndarray:
    last_price = float(context[-1])
    mean_price = float(np.mean(context))
    pct_change = float((context[-1] - context[0]) / context[0]) if context[0] != 0 else 0.0
    volatility = float(np.std(context))
    return np.array([base_pred_price, last_price, mean_price, pct_change, volatility], dtype=np.float32)


def load_adapter_state(adapter_path: str):
    """Load optional residual adapter trained by examples/finetune_stock.py."""
    import torch

    if not adapter_path or not os.path.exists(adapter_path):
        return None
    try:
        state = torch.load(adapter_path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    required = {"adapter_coef", "scaler_mean", "scaler_scale"}
    if not required.issubset(state.keys()):
        return None
    # Normalize dtypes
    state["adapter_coef"] = state["adapter_coef"].astype(np.float32)
    state["scaler_mean"] = state["scaler_mean"].astype(np.float32)
    state["scaler_scale"] = state["scaler_scale"].astype(np.float32)
    trained_codes = state.get("trained_codes")
    if trained_codes is None and "stock_code" in state:
        trained_codes = [state["stock_code"]]
    if isinstance(trained_codes, np.ndarray):
        trained_codes = trained_codes.tolist()
    if isinstance(trained_codes, tuple):
        trained_codes = list(trained_codes)
    state["trained_codes"] = trained_codes
    return state


def apply_adapter_if_available(code: str, base_pred_price: float, context: np.ndarray, adapter_state: dict | None) -> float:
    if adapter_state is None:
        return base_pred_price
    coef = adapter_state["adapter_coef"]
    mean = adapter_state["scaler_mean"]
    scale = adapter_state["scaler_scale"]
    features = compute_feature_vector(context, base_pred_price)
    scaled = (features - mean) / np.where(scale == 0, 1.0, scale)
    residual = float(np.dot(scaled, coef[:-1]) + coef[-1])
    return float(base_pred_price + residual)


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def forecast_one_step_base(model: TimesFM_2p5_200M_torch, context: np.ndarray) -> float:
    # Follow examples/fc_timesfm_score.py normalization
    norm_context = (context / context[0]).astype(np.float32)
    forecasts = model.model.forecast_naive(horizon=1, inputs=[norm_context])
    # Compatible with shape (1, 1) or (1, q)
    if forecasts[0].ndim == 1:
        f = forecasts[0][0]
    elif forecasts[0].shape[1] == 1:
        f = forecasts[0][0]
    else:
        f = forecasts[0][0, 1]
    return float(f * context[0])


def evaluate_stock_over_last_3_months(
    model: TimesFM_2p5_200M_torch,
    stock_code: str,
    context_len: int = 60,
    use_adapter: bool = True,
    adapter_state: dict | None = None,
):
    today = pd.Timestamp.today().normalize().date()
    end_date = (pd.Timestamp(today) - pd.Timedelta(days=1)).date()
    start_date = (pd.Timestamp(end_date) - pd.DateOffset(months=3)).date()
    # Fetch enough history for contexts
    fetch_start = (pd.Timestamp(start_date) - pd.Timedelta(days=context_len * 3)).date()

    df = load_oss_stocks(codes=[stock_code], start=fetch_start, end=end_date)
    if stock_code not in df.columns:
        raise RuntimeError(f"No data for {stock_code} in the specified range.")
    series = df[stock_code].dropna()
    dates = series.index.to_pydatetime()
    prices = series.to_numpy(dtype=np.float32)

    preds_base = []
    preds_adapted = []
    targets = []
    target_dates = []

    for idx in range(context_len, len(prices)):
        target_date = pd.Timestamp(dates[idx]).date()
        if not (start_date <= target_date <= end_date):
            continue
        context = prices[idx - context_len: idx]
        base_pred = forecast_one_step_base(model, context)
        if use_adapter and adapter_state is not None:
            adapted_pred = apply_adapter_if_available(stock_code, base_pred, context, adapter_state)
        else:
            adapted_pred = base_pred

        preds_base.append(base_pred)
        preds_adapted.append(adapted_pred)
        targets.append(float(prices[idx]))
        target_dates.append(pd.Timestamp(dates[idx]).date())

    preds_base = np.array(preds_base, dtype=np.float32)
    preds_adapted = np.array(preds_adapted, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)

    metrics = {
        "base_mae": mean_absolute_error(targets, preds_base) if len(targets) else float("nan"),
        "base_mape": safe_mape(targets, preds_base) if len(targets) else float("nan"),
        "adapt_mae": mean_absolute_error(targets, preds_adapted) if len(targets) else float("nan"),
        "adapt_mape": safe_mape(targets, preds_adapted) if len(targets) else float("nan"),
        "num_points": int(len(targets)),
        "start_date": start_date,
        "end_date": end_date,
    }

    return target_dates, targets, preds_base, preds_adapted, metrics


def plot_predictions(stock_code: str, dates, targets, preds_base, preds_adapted, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(11, 4))
    plt.plot(dates, targets, label="Actual", color="#333333")
    plt.plot(dates, preds_base, label="Pred (Base)", color="#1f77b4")
    plt.plot(dates, preds_adapted, label="Pred (Adapted)", color="#ff7f0e", alpha=0.8)
    plt.xticks(rotation=30, ha="right")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(f"{stock_code} Next-day Forecast vs Actual (Last 3 Months)")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{stock_code}_last3m.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def main():
    stock_codes = ["601006", "603993", "000100"]
    context_len = 60
    use_adapter = True
    adapter_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../finetuned_adapter.pth"))
    adapter_state = load_adapter_state(adapter_path) if use_adapter else None

    print("Loading local TimesFM 2.5 model...")
    tfm = TimesFM_2p5_200M_torch.from_pretrained("./local_timesfm_model")

    results = {}
    for code in stock_codes:
        try:
            dates, y_true, y_base, y_adapt, metrics = evaluate_stock_over_last_3_months(
                tfm, code, context_len=context_len, use_adapter=use_adapter, adapter_state=adapter_state
            )
            fig_path = plot_predictions(code, dates, y_true, y_base, y_adapt, output_dir=os.path.join(os.path.dirname(__file__), "../outputs"))
            results[code] = {"metrics": metrics, "figure": os.path.abspath(fig_path)}
            print(
                f"{code}: N={metrics['num_points']} | Base MAE={metrics['base_mae']:.4f}, MAPE={metrics['base_mape']:.2f}% | "
                f"Adapt MAE={metrics['adapt_mae']:.4f}, MAPE={metrics['adapt_mape']:.2f}% | Fig: {fig_path}"
            )
        except Exception as exc:
            print(f"Failed to evaluate {code}: {exc}")

    return results


if __name__ == "__main__":
    main()


