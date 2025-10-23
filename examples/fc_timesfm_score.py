import matplotlib.pyplot as plt
def plot_history_and_forecast(prices, context, pred_price, stock_code):
    plt.figure(figsize=(10, 4))
    x_hist = range(len(prices) - len(context), len(prices))
    plt.plot(x_hist, context, label='history price', marker='o')
    plt.plot([x_hist[-1], x_hist[-1] + 1], [context[-1], pred_price], 'ro--', label='forecast')
    plt.xlabel('time number')
    plt.ylabel('prices:')
    plt.title(f'{stock_code} history and forecast')
    plt.legend()
    plt.tight_layout()
    plt.show()
import json
import numpy as np
import pandas as pd
import os
import sys
import torch

# 适配本地依赖
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from timesfm import TimesFM_2p5_200M_torch
from examples.data import load_oss_stocks

tfm = TimesFM_2p5_200M_torch.from_pretrained(
    "./local_timesfm_model"
)

_ADAPTER_CACHE = None

CONFIG = {
    "use_adapter": True,
    "adapter_path": os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../finetuned_adapter_all_99stocks.pth")
    ),
    # adapter_scope: "auto" obeys metadata, "match" forces metadata matching,
    # "global" applies adapter to all stocks regardless of metadata.
    "adapter_scope": "global",  # Use global mode for best overall performance
    # Timing control: Only trade low-risk stocks
    "min_confidence_threshold": 2.0,  # Only trade stocks with MAPE < 2%
}


def _load_adapter():
    global _ADAPTER_CACHE
    if _ADAPTER_CACHE is not None:
        return _ADAPTER_CACHE
    if not CONFIG.get("use_adapter", True):
        _ADAPTER_CACHE = None
        return None
    adapter_path = CONFIG.get("adapter_path")
    if not adapter_path:
        _ADAPTER_CACHE = None
        return None
    if not os.path.exists(adapter_path):
        _ADAPTER_CACHE = None
        return None
    try:
        state = torch.load(adapter_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"Warning: failed to load adapter from {adapter_path}: {exc}")
        _ADAPTER_CACHE = None
        return None
    required = {"adapter_coef", "scaler_mean", "scaler_scale"}
    if not required.issubset(state.keys()):
        _ADAPTER_CACHE = None
        return None
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
    _ADAPTER_CACHE = state
    return _ADAPTER_CACHE

def get_stock_score(stock_code, context_len=60):
    try:
        # 取今天日期
        today = pd.Timestamp.today().normalize()
        # 预测目标日为今天
        predict_day = today
        end_day = predict_day - pd.Timedelta(days=1)
        start_day = end_day - pd.Timedelta(days=context_len*3)
        df = load_oss_stocks(codes=[stock_code], start=start_day, end=end_day)
        prices = df[stock_code].dropna().to_numpy(dtype=np.float32)
        
        if len(prices) < context_len:
            print(f"实际抓取到{len(prices)}条数据，要求{context_len}条")
            return None
        context = prices[-context_len:]
        norm_context = context / context[0]

        forecasts = tfm.model.forecast_naive(horizon=1, inputs=[norm_context])
        # 兼容 shape (1, 1) 或 (1, 3)
        if forecasts[0].shape[1] == 1:
            forecast = forecasts[0][0, 0]
        else:
            forecast = forecasts[0][0, 1]
        base_pred_price = forecast * context[0]

        adapter_state = _load_adapter()
        apply_adapter = False
        if adapter_state is not None:
            scope = CONFIG.get("adapter_scope", "auto")
            trained_codes = adapter_state.get("trained_codes")
            if scope == "global":
                apply_adapter = True
            elif scope == "match":
                apply_adapter = trained_codes is not None and stock_code in trained_codes
            else:  # auto
                if not trained_codes:
                    apply_adapter = True
                else:
                    apply_adapter = stock_code in trained_codes

        if apply_adapter and adapter_state is not None:
            coef = adapter_state["adapter_coef"]
            mean = adapter_state["scaler_mean"]
            scale = adapter_state["scaler_scale"]
            features = np.array(
                [
                    base_pred_price,
                    context[-1],
                    float(np.mean(context)),
                    float((context[-1] - context[0]) / context[0]) if context[0] != 0 else 0.0,
                    float(np.std(context)),
                ],
                dtype=np.float32,
            )
            scaled_features = (features - mean) / np.where(scale == 0, 1.0, scale)
            residual = float(np.dot(scaled_features, coef[:-1]) + coef[-1])
            pred_price = base_pred_price + residual
        else:
            pred_price = base_pred_price
        last_price = context[-1]
        rise = (pred_price - last_price) / last_price
        score = max(0, min(1, rise / 0.1))
        # 可选：画图
        plot_history_and_forecast(prices, context, pred_price, stock_code)
        return score
    except Exception as e:
        print(f"Failed to load data for code {stock_code}: {e}")
        return None

def handler(event, context):
    try:
        body = event if isinstance(event, dict) else json.loads(event["body"])
        codes = body.get("codes", [])
        if not codes:
            return {"statusCode": 400, "body": json.dumps({"error": "Missing codes"})}
        results = {}
        for code in codes:
            score = get_stock_score(code)
            if score is not None:
                results[code] = float(score)
            else:
                results[code] = -1
        return {"statusCode": 200, "body": json.dumps(results)}
    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}


if __name__ == "__main__":
    # 本地测试
    test_event = {
        "codes": ["601006", "603993","000100"]
    }
    result = handler(test_event, None)
    print("Test result:", result)
