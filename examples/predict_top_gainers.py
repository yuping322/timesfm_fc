import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# 适配本地依赖
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from timesfm import TimesFM_2p5_200M_torch
from examples.data import load_oss_stocks


def load_adapter_state(adapter_path: str):
    """Load adapter state"""
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
    state["adapter_coef"] = state["adapter_coef"].astype(np.float32)
    state["scaler_mean"] = state["scaler_mean"].astype(np.float32)
    state["scaler_scale"] = state["scaler_scale"].astype(np.float32)
    return state


def compute_feature_vector(context: np.ndarray, base_pred_price: float) -> np.ndarray:
    last_price = float(context[-1])
    mean_price = float(np.mean(context))
    pct_change = float((context[-1] - context[0]) / context[0]) if context[0] != 0 else 0.0
    volatility = float(np.std(context))
    return np.array([base_pred_price, last_price, mean_price, pct_change, volatility], dtype=np.float32)


def apply_adapter(base_pred_price: float, context: np.ndarray, adapter_state) -> float:
    if adapter_state is None:
        return base_pred_price
    coef = adapter_state["adapter_coef"]
    mean = adapter_state["scaler_mean"]
    scale = adapter_state["scaler_scale"]
    features = compute_feature_vector(context, base_pred_price)
    scaled = (features - mean) / np.where(scale == 0, 1.0, scale)
    residual = float(np.dot(scaled, coef[:-1]) + coef[-1])
    return float(base_pred_price + residual)


def forecast_one_step(model, context: np.ndarray) -> float:
    """Forecast next price from context"""
    norm_context = (context / context[0]).astype(np.float32)
    forecasts = model.model.forecast_naive(horizon=1, inputs=[norm_context])
    if forecasts[0].ndim == 1:
        f = forecasts[0][0]
    elif forecasts[0].shape[1] == 1:
        f = forecasts[0][0]
    else:
        f = forecasts[0][0, 1]
    return float(f * context[0])


def predict_stock_price(model, stock_code: str, adapter_state, context_len: int = 60):
    """Predict tomorrow's price for a stock"""
    try:
        today = pd.Timestamp.today().normalize().date()
        predict_day = today
        end_day = predict_day - pd.Timedelta(days=1)
        start_day = end_day - pd.Timedelta(days=context_len * 3)
        
        df = load_oss_stocks(codes=[stock_code], start=start_day, end=end_day)
        if stock_code not in df.columns:
            return None
        
        prices = df[stock_code].dropna().to_numpy(dtype=np.float32)
        if len(prices) < context_len:
            return None
        
        context = prices[-context_len:]
        last_price = context[-1]
        
        # Base prediction
        base_pred = forecast_one_step(model, context)
        
        # Apply adapter
        pred_price = apply_adapter(base_pred, context, adapter_state)
        
        # Calculate expected rise
        expected_rise = (pred_price - last_price) / last_price if last_price != 0 else 0
        
        return {
            "stock_code": stock_code,
            "last_price": float(last_price),
            "pred_price": float(pred_price),
            "expected_rise": float(expected_rise),
            "context": context.tolist()
        }
    except Exception as e:
        return None


def load_all_stock_codes():
    """Load all A-share stock codes from CSV"""
    csv_path = os.path.join(os.path.dirname(__file__), "../all_a_stocks.csv")
    try:
        df = pd.read_csv(csv_path, dtype=str)
        return df['code'].tolist()
    except Exception as e:
        print(f"Failed to load stock codes: {e}")
        return []


def plot_top_gainers(top_predictions, output_dir):
    """Plot top gainers prediction chart"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subplot
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Expected rise distribution
    rises = [p['expected_rise'] * 100 for p in top_predictions]
    stock_codes = [p['stock_code'] for p in top_predictions]
    
    axes[0].barh(stock_codes, rises, color='green' if rises[0] > 0 else 'red')
    axes[0].set_xlabel('Expected Rise (%)')
    axes[0].set_title('Top 20 Stocks - Expected Daily Rise')
    axes[0].grid(axis='x', alpha=0.3)
    
    # Plot 2: Price chart for top 5 stocks
    for i, pred in enumerate(top_predictions[:5]):
        context = np.array(pred['context'])
        last_price = pred['last_price']
        pred_price = pred['pred_price']
        
        x_hist = range(len(context))
        axes[1].plot(x_hist, context, label=f"{pred['stock_code']} (Last: {last_price:.2f})", marker='o', markersize=3)
        axes[1].plot([x_hist[-1], x_hist[-1] + 1], [last_price, pred_price], 
                     'ro--', linewidth=2, markersize=8)
    
    axes[1].set_xlabel('Trading Days (Last 60)')
    axes[1].set_ylabel('Price')
    axes[1].set_title('Top 5 Stocks - Historical Price & Prediction')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "top_gainers_prediction.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def main():
    # Configuration
    context_len = 60
    top_n = 20  # Top N stocks to show
    adapter_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../finetuned_adapter_all_99stocks.pth"))
    output_dir = os.path.join(os.path.dirname(__file__), "../outputs_top_gainers")
    
    print("Loading TimesFM 2.5 model...")
    model = TimesFM_2p5_200M_torch.from_pretrained("./local_timesfm_model")
    
    print("Loading adapter...")
    adapter_state = load_adapter_state(adapter_path)
    
    print("Loading all A-share stock codes...")
    all_codes = load_all_stock_codes()
    print(f"Total stocks: {len(all_codes)}")
    
    print("Predicting tomorrow's prices for all stocks...")
    predictions = []
    
    for code in tqdm(all_codes, desc="Predicting"):
        result = predict_stock_price(model, code, adapter_state, context_len)
        if result is not None:
            predictions.append(result)
    
    print(f"\nSuccessfully predicted {len(predictions)} stocks")
    
    # Sort by expected rise
    predictions_sorted = sorted(predictions, key=lambda x: x['expected_rise'], reverse=True)
    
    # Statistics
    all_rises = [p['expected_rise'] for p in predictions]
    rising_count = sum(1 for r in all_rises if r > 0)
    falling_count = sum(1 for r in all_rises if r < 0)
    
    print(f"\n{'='*80}")
    print(f"MARKET PREDICTION SUMMARY")
    print(f"{'='*80}")
    print(f"Total stocks predicted: {len(predictions)}")
    print(f"Predicted to rise (>0%): {rising_count} ({rising_count/len(predictions)*100:.1f}%)")
    print(f"Predicted to fall (<0%): {falling_count} ({falling_count/len(predictions)*100:.1f}%)")
    print(f"Average expected change: {np.mean(all_rises)*100:.2f}%")
    print(f"Median expected change: {np.median(all_rises)*100:.2f}%")
    
    # Get top gainers
    top_gainers = predictions_sorted[:top_n]
    
    # Print results
    print(f"\n{'='*80}")
    print(f"TOP {top_n} EXPECTED GAINERS (Tomorrow)")
    print(f"{'='*80}")
    print(f"{'Rank':<6} {'Code':<10} {'Last Price':<12} {'Pred Price':<12} {'Expected Rise':<15}")
    print(f"{'-'*80}")
    
    for i, pred in enumerate(top_gainers, 1):
        print(f"{i:<6} {pred['stock_code']:<10} {pred['last_price']:<12.2f} "
              f"{pred['pred_price']:<12.2f} {pred['expected_rise']*100:<15.2f}%")
    
    # Plot chart
    print("\nGenerating visualization...")
    chart_path = plot_top_gainers(top_gainers, output_dir)
    print(f"Chart saved to: {chart_path}")
    
    # Save results to JSON
    json_path = os.path.join(output_dir, "top_gainers.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(top_gainers, f, ensure_ascii=False, indent=2)
    print(f"Results saved to: {json_path}")
    
    # Save to CSV
    csv_path = os.path.join(output_dir, "top_gainers.csv")
    df = pd.DataFrame(top_gainers)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()

