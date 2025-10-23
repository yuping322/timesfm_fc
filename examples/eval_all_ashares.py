import os
import sys
import io
import json
import random
import numpy as np
import pandas as pd
import matplotlib

# Use non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

# Ensure local src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from timesfm import TimesFM_2p5_200M_torch  # noqa: E402
from examples.data import load_oss_stocks  # noqa: E402


def generate_ashare_codes(sample_size: int | None = None, random_seed: int = 42) -> list[str]:
    """Load A-share stock codes from CSV file or return a random sample.

    Args:
        sample_size: If provided, return a random sample of this size.
                     If None, return all codes from CSV.
        random_seed: Random seed for sampling.

    Returns:
        List of stock codes like ["000001", "600000", ...]
    """
    csv_path = os.path.join(os.path.dirname(__file__), "../all_a_stocks.csv")

    try:
        df = pd.read_csv(csv_path, dtype=str)
        all_codes = df['code'].tolist()
        print(f"Loaded {len(all_codes)} stock codes from {csv_path}")
    except Exception as e:
        print(f"Failed to load stock codes from CSV: {e}")
        print("Falling back to generated codes...")
        # Fallback to generated codes if CSV fails
        all_codes = []

        # Shanghai Stock Exchange (上交所)
        for prefix in ["600", "601", "603", "688", "689"]:
            for suffix in range(1000):
                all_codes.append(f"{prefix}{suffix:03d}")

        # Shenzhen Stock Exchange (深交所)
        for prefix in ["000", "001", "002", "003", "300", "301"]:
            for suffix in range(1000):
                all_codes.append(f"{prefix}{suffix:03d}")

        # Beijing Stock Exchange (北交所)
        for prefix in ["400", "800", "820", "830", "870", "880"]:
            for suffix in range(1000):
                all_codes.append(f"{prefix}{suffix:03d}")

    if sample_size is None or sample_size >= len(all_codes):
        return all_codes

    random.seed(random_seed)
    return random.sample(all_codes, sample_size)


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
    min_data_points: int = 10,
):
    """Evaluate one stock over the last 3 months.

    Returns:
        dict with metrics or None if insufficient data.
    """
    today = pd.Timestamp.today().normalize().date()
    end_date = (pd.Timestamp(today) - pd.Timedelta(days=1)).date()
    start_date = (pd.Timestamp(end_date) - pd.DateOffset(months=3)).date()
    # Fetch enough history for contexts
    fetch_start = (pd.Timestamp(start_date) - pd.Timedelta(days=context_len * 3)).date()

    try:
        df = load_oss_stocks(codes=[stock_code], start=fetch_start, end=end_date)
        if stock_code not in df.columns:
            return None
        series = df[stock_code].dropna()
        if len(series) < context_len + min_data_points:
            return None

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

        if len(targets) < min_data_points:
            return None

        metrics = {
            "base_mae": mean_absolute_error(targets, preds_base),
            "base_mape": safe_mape(targets, preds_base),
            "adapt_mae": mean_absolute_error(targets, preds_adapted),
            "adapt_mape": safe_mape(targets, preds_adapted),
            "num_points": int(len(targets)),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        }

        return metrics

    except Exception as exc:
        return None


def save_summary_results(results: dict, output_dir: str):
    """Save summary statistics to JSON and CSV."""
    os.makedirs(output_dir, exist_ok=True)

    # Filter out None results
    valid_results = {k: v for k, v in results.items() if v is not None}

    if not valid_results:
        print("No valid results to save.")
        return

    # Summary statistics
    summary = {
        "total_stocks_evaluated": len(results),
        "stocks_with_data": len(valid_results),
        "evaluation_period": f"{list(valid_results.values())[0]['start_date']} to {list(valid_results.values())[0]['end_date']}",
        "overall_metrics": {},
    }

    # Aggregate metrics across all stocks
    all_base_mae = [v["base_mae"] for v in valid_results.values() if not np.isnan(v["base_mae"])]
    all_base_mape = [v["base_mape"] for v in valid_results.values() if not np.isnan(v["base_mape"])]
    all_adapt_mae = [v["adapt_mae"] for v in valid_results.values() if not np.isnan(v["adapt_mae"])]
    all_adapt_mape = [v["adapt_mape"] for v in valid_results.values() if not np.isnan(v["adapt_mape"])]

    if all_base_mae:
        summary["overall_metrics"]["base_mae_mean"] = float(np.mean(all_base_mae))
        summary["overall_metrics"]["base_mae_median"] = float(np.median(all_base_mae))
        summary["overall_metrics"]["base_mae_std"] = float(np.std(all_base_mae))

    if all_base_mape:
        summary["overall_metrics"]["base_mape_mean"] = float(np.mean(all_base_mape))
        summary["overall_metrics"]["base_mape_median"] = float(np.median(all_base_mape))
        summary["overall_metrics"]["base_mape_std"] = float(np.std(all_base_mape))

    if all_adapt_mae:
        summary["overall_metrics"]["adapt_mae_mean"] = float(np.mean(all_adapt_mae))
        summary["overall_metrics"]["adapt_mae_median"] = float(np.median(all_adapt_mae))
        summary["overall_metrics"]["adapt_mae_std"] = float(np.std(all_adapt_mae))

    if all_adapt_mape:
        summary["overall_metrics"]["adapt_mape_mean"] = float(np.mean(all_adapt_mape))
        summary["overall_metrics"]["adapt_mape_median"] = float(np.median(all_adapt_mape))
        summary["overall_metrics"]["adapt_mape_std"] = float(np.std(all_adapt_mape))

    # Save summary JSON
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Save detailed CSV
    df_data = []
    for code, metrics in valid_results.items():
        df_data.append({
            "stock_code": code,
            **metrics
        })

    df = pd.DataFrame(df_data)
    df.to_csv(os.path.join(output_dir, "detailed_results.csv"), index=False, encoding="utf-8-sig")

    print(f"\nSummary saved to {output_dir}/summary.json and detailed_results.csv")
    print(f"Evaluated {len(valid_results)}/{len(results)} stocks with sufficient data.")


def evaluate_three_modes(tfm, stock_codes, context_len, adapter_state):
    """Evaluate all stocks in three modes: no_adapter, auto, global"""
    results = {"no_adapter": {}, "auto": {}, "global": {}}
    
    for code in tqdm(stock_codes, desc="Evaluating stocks"):
        # Mode 1: No adapter
        metrics_base = evaluate_stock_over_last_3_months(
            tfm, code, context_len=context_len, use_adapter=False, adapter_state=None
        )
        results["no_adapter"][code] = metrics_base
        
        # Mode 2: Auto mode (use adapter only for trained stocks)
        if adapter_state is not None:
            trained_codes = adapter_state.get("trained_codes", [])
            use_adapter = code in trained_codes if trained_codes else False
        else:
            use_adapter = False
        metrics_auto = evaluate_stock_over_last_3_months(
            tfm, code, context_len=context_len, use_adapter=use_adapter, adapter_state=adapter_state
        )
        results["auto"][code] = metrics_auto
        
        # Mode 3: Global mode (use adapter for all stocks)
        metrics_global = evaluate_stock_over_last_3_months(
            tfm, code, context_len=context_len, use_adapter=True, adapter_state=adapter_state
        )
        results["global"][code] = metrics_global
    
    return results


def main():
    # Configuration
    sample_size = 500  # Sample 500 stocks for broader testing; set to None for all ~5400 stocks
    context_len = 60
    adapter_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../finetuned_adapter_all_99stocks.pth"))
    adapter_state = load_adapter_state(adapter_path)
    output_dir = os.path.join(os.path.dirname(__file__), "../outputs_all_ashares_comparison")

    print(f"Loading local TimesFM 2.5 model...")
    tfm = TimesFM_2p5_200M_torch.from_pretrained("./local_timesfm_model")

    print(f"Generating A-share stock codes (sample_size={sample_size})...")
    stock_codes = generate_ashare_codes(sample_size=sample_size)

    print(f"Evaluating {len(stock_codes)} stocks in three modes: No Adapter, Auto, Global...")
    results_all_modes = evaluate_three_modes(tfm, stock_codes, context_len, adapter_state)

    # Print comparison summary
    print("\n" + "="*80)
    print("COMPARISON OF THREE MODES")
    print("="*80)
    
    for mode_name, results in results_all_modes.items():
        valid_results = {k: v for k, v in results.items() if v is not None}
        if not valid_results:
            print(f"\n{mode_name.upper()}: No valid results")
            continue
            
        all_mape = [v["adapt_mape"] for v in valid_results.values() if not np.isnan(v["adapt_mape"])]
        if all_mape:
            print(f"\n{mode_name.upper()} MODE:")
            print(f"  Stocks evaluated: {len(valid_results)}")
            print(f"  Mean MAPE: {np.mean(all_mape):.2f}%")
            print(f"  Median MAPE: {np.median(all_mape):.2f}%")
            print(f"  Std MAPE: {np.std(all_mape):.2f}%")
    
    # Save detailed results for each mode
    os.makedirs(output_dir, exist_ok=True)
    for mode_name, results in results_all_modes.items():
        save_summary_results(results, os.path.join(output_dir, mode_name))
    
    print(f"\nDetailed results saved to {output_dir}/")


if __name__ == "__main__":
    main()
