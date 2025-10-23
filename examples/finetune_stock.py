import sys
import os
import random
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import torch
from tqdm import tqdm

# 将本地 src 加入路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from timesfm import TimesFM_2p5_200M_torch  # noqa: E402
from examples.data import load_oss_stocks  # noqa: E402


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_ashare_codes(sample_size: int | None = None, random_seed: int = 42) -> list[str]:
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
        print("Falling back to hardcoded codes...")
        # Fallback to original hardcoded codes if CSV fails
        all_codes = ["601006", "603993", "000100"]

    if sample_size is None or sample_size >= len(all_codes):
        return all_codes

    random.seed(random_seed)
    return random.sample(all_codes, sample_size)


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def compute_feature_vector(context: np.ndarray, base_pred: float) -> np.ndarray:
    last_price = context[-1]
    mean_price = float(np.mean(context))
    pct_change = float((last_price - context[0]) / context[0]) if context[0] != 0 else 0.0
    volatility = float(np.std(context))
    return np.array([base_pred, last_price, mean_price, pct_change, volatility], dtype=np.float32)


@dataclass
class DatasetSplits:
    train_X: np.ndarray
    train_y: np.ndarray
    train_base: np.ndarray
    val_X: np.ndarray
    val_y: np.ndarray
    val_base: np.ndarray


def build_dataset(
    model: TimesFM_2p5_200M_torch,
    stock_code: str,
    start_date: str,
    end_date: str,
    context_len: int,
    horizon_len: int,
    train_ratio: float = 0.8,
) -> DatasetSplits:
    df = load_oss_stocks(codes=[stock_code], start=start_date, end=end_date)
    if stock_code not in df.columns:
        raise ValueError(f"No data found for stock {stock_code}.")
    prices = df[stock_code].dropna().to_numpy(dtype=np.float32)
    if len(prices) < context_len + 10:
        raise ValueError("Not enough data points for the specified context length.")

    features = []
    base_preds = []
    targets = []
    for idx in range(context_len, len(prices) - horizon_len + 1):
        context = prices[idx - context_len:idx]
        target = prices[idx]
        if not np.isfinite(target) or not np.all(np.isfinite(context)):
            continue
        scale = context[0] if abs(context[0]) > 1e-6 else 1.0
        norm_context = (context / scale).astype(np.float32)
        try:
            forecast = model.model.forecast_naive(horizon=horizon_len, inputs=[norm_context])[0]
        except Exception as exc:
            print(f"Skipping sample at index {idx} due to forecast error: {exc}")
            continue
        if forecast.ndim == 1:
            base_point = float(forecast[0])
        else:
            base_point = float(forecast[0, 0])
        base_price = base_point * scale
        if not np.isfinite(base_price):
            continue
        features.append(compute_feature_vector(context, base_price))
        base_preds.append(base_price)
        targets.append(target)

    features = np.array(features, dtype=np.float32)
    base_preds = np.array(base_preds, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)

    if len(targets) < 50:
        raise ValueError("Dataset is too small after filtering valid samples.")

    split_idx = int(len(targets) * train_ratio)
    train_X, val_X = features[:split_idx], features[split_idx:]
    train_y, val_y = targets[:split_idx], targets[split_idx:]
    train_base, val_base = base_preds[:split_idx], base_preds[split_idx:]

    return DatasetSplits(train_X, train_y, train_base, val_X, val_y, val_base)


def fit_linear_adapter(train_X: np.ndarray, train_residuals: np.ndarray) -> np.ndarray:
    """Returns linear coefficients (including bias) solving least squares."""
    ones = np.ones((train_X.shape[0], 1), dtype=np.float32)
    X_aug = np.concatenate([train_X, ones], axis=1)
    coef, *_ = np.linalg.lstsq(X_aug, train_residuals, rcond=None)
    return coef.astype(np.float32)


def apply_linear_adapter(features: np.ndarray, coef: np.ndarray) -> np.ndarray:
    ones = np.ones((features.shape[0], 1), dtype=np.float32)
    X_aug = np.concatenate([features, ones], axis=1)
    return X_aug @ coef


def main():
    set_seed(42)

    # Configuration - use 100 stocks for better diversity
    sample_size = 100  # Use 100 stocks for training; set to None for all ~5400 stocks
    stock_codes = load_ashare_codes(sample_size=sample_size)

    start_date = "2022-01-01"
    end_date = "2025-10-10"
    context_len = 60
    horizon_len = 1
    train_ratio = 0.8
    min_samples_per_stock = 50  # Minimum samples required per stock

    print(f"加载本地模型...")
    tfm = TimesFM_2p5_200M_torch.from_pretrained("./local_timesfm_model")

    print(f"开始为 {len(stock_codes)} 只股票构建数据集...")
    collected = []
    successful_stocks = 0

    for code in tqdm(stock_codes, desc="Processing stocks"):
        try:
            splits = build_dataset(
                tfm,
                stock_code=code,
                start_date=start_date,
                end_date=end_date,
                context_len=context_len,
                horizon_len=horizon_len,
                train_ratio=train_ratio,
            )

            # Only include stocks with sufficient samples
            if len(splits.train_y) >= min_samples_per_stock and len(splits.val_y) >= 10:
                collected.append((code, splits))
                successful_stocks += 1
                if successful_stocks <= 5:  # Print first few for progress
                    print(
                        f"股票 {code} 样本数: 训练 {len(splits.train_y)} / 验证 {len(splits.val_y)}"
                    )
            else:
                if successful_stocks <= 5:
                    print(f"股票 {code} 样本不足，跳过")

        except Exception as exc:
            if successful_stocks <= 5:
                print(f"警告: 构建股票 {code} 数据集失败: {exc}")

    print(f"\n成功构建数据集的股票数量: {len(collected)}/{len(stock_codes)}")

    if not collected:
        raise RuntimeError("无法为任何股票构建数据集，请检查数据源")

    print("合并所有股票的数据集...")
    train_X = np.concatenate([splits.train_X for _, splits in collected], axis=0)
    train_y = np.concatenate([splits.train_y for _, splits in collected], axis=0)
    train_base = np.concatenate([splits.train_base for _, splits in collected], axis=0)
    val_X = np.concatenate([splits.val_X for _, splits in collected], axis=0)
    val_y = np.concatenate([splits.val_y for _, splits in collected], axis=0)
    val_base = np.concatenate([splits.val_base for _, splits in collected], axis=0)

    print(
        "有效样本总数: {}\n训练集: {}  验证集: {}".format(
            len(train_y) + len(val_y), len(train_y), len(val_y)
        )
    )

    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_X)
    val_X_scaled = scaler.transform(val_X)

    base_mae = mean_absolute_error(val_y, val_base)
    base_mape = safe_mape(val_y, val_base)
    print(f"基模型验证集  MAE: {base_mae:.4f}, MAPE: {base_mape:.2f}%")

    print("训练线性残差适配器...")
    train_residuals = train_y - train_base
    coef = fit_linear_adapter(train_X_scaled, train_residuals)

    val_residuals = apply_linear_adapter(val_X_scaled, coef)
    val_preds = val_base + val_residuals
    ft_mae = mean_absolute_error(val_y, val_preds)
    ft_mape = safe_mape(val_y, val_preds)
    print(f"微调后模型验证集 MAE: {ft_mae:.4f}, MAPE: {ft_mape:.2f}%")

    improvement_mae = base_mae - ft_mae
    improvement_mape = base_mape - ft_mape
    print("\n=== 结果对比 ===")
    print(f"MAE 改善: {improvement_mae:+.4f}")
    print(f"MAPE 改善: {improvement_mape:+.2f}%")

    # Save the new comprehensive adapter
    adapter_filename = f"finetuned_adapter_all_{len(collected)}stocks.pth"
    torch.save(
        {
            "adapter_coef": coef,
            "scaler_mean": scaler.mean_.astype(np.float32),
            "scaler_scale": scaler.scale_.astype(np.float32),
            "feature_names": ["base_pred", "last_price", "mean_price", "pct_change", "volatility"],
            "stock_code": None,
            "trained_codes": [code for code, _ in collected],
            "context_len": context_len,
            "horizon_len": horizon_len,
            "total_stocks": len(collected),
            "total_samples": len(train_y) + len(val_y),
        },
        os.path.join(os.path.dirname(__file__), f"../{adapter_filename}"),
    )
    print(f"已将适配器权重保存到 {adapter_filename}")

    return adapter_filename


if __name__ == "__main__":
    # df = load_oss_stocks(codes=["430017"], start="2022-01-01", end="2025-10-10")
    main()
