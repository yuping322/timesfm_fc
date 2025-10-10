from typing import Union, List
import datetime as dt
import pandas as pd

import oss2   # 假设 bucket 已全局初始化好
import io
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # 未安装 python-dotenv 时忽略
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # 未安装 python-dotenv 时忽略


# 从本地环境变量读取 OSS 密钥和 bucket
OSS_ACCESS_KEY_ID = os.environ.get("OSS_ACCESS_KEY_ID", "")
OSS_ACCESS_KEY_SECRET = os.environ.get("OSS_ACCESS_KEY_SECRET", "")
OSS_ENDPOINT = os.environ.get("OSS_ENDPOINT", "https://oss-cn-hangzhou.aliyuncs.com")
OSS_BUCKET = os.environ.get("OSS_BUCKET", "test123432")

auth = oss2.Auth(OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET)

bucket = oss2.Bucket(
    auth,
    OSS_ENDPOINT,  # 可通过环境变量覆盖
    OSS_BUCKET     # 可通过环境变量覆盖
)


def load_oss_stocks(
    codes: Union[str, List[str]] = None,
    start: str = None,
    end: str = None,
) -> pd.DataFrame:
    """
    从 OSS 目录 hangqing/daily_data/ 拉取 start~end 区间内所有股票的日线行情，
    返回 DataFrame(index=date, columns=股票代码, values=收盘价)。
    """
    # ---------------- 日期范围处理 ----------------
    if start is None:
        start = dt.date(2000, 1, 1)
    else:
        start = pd.to_datetime(start).date()

    if end is None:
        end = dt.date.today()
    else:
        end = pd.to_datetime(end).date()

    # ---------------- 股票代码过滤 ----------------
    if codes is not None:
        if isinstance(codes, str):
            codes = [codes]
        codes = [c.zfill(6) for c in codes]

    # ---------------- 遍历 OSS 目录 ----------------
    prefix = "hangqing/daily_data/"
    frames = []
    def add_prefix(code: str) -> str:
        code = str(code).zfill(6)
        if code.startswith("6"):
            return "sh" + code
        elif code.startswith(("0", "3")):
            return "sz" + code
        elif code.startswith(("4", "8")):
            return "bj" + code
        else:
            return code

    for code in codes:
        try:
            fname = add_prefix(code) + ".csv"
            content = bucket.get_object(prefix + fname).read()
            # 只把可能出问题的列强制 str，close 让 pandas 自己推断
            df = pd.read_csv(io.BytesIO(content),
                            dtype={"代码": str, "日期": str})
        except Exception as e:
            print(f"Failed to load data for code {code}: {e}")
            continue

        # 日期、价格列统一命名
        df["date"] = pd.to_datetime(df["日期"])
        df["close"] = pd.to_numeric(df["close"], errors="coerce")

        # 日期过滤
        mask = (df["date"].dt.date >= start) & (df["date"].dt.date <= end)
        df = df.loc[mask, ["date", "close"]]
        df["asset"] = code
        frames.append(df)

    if not frames:
        return pd.DataFrame(dtype=float)

    df_all = pd.concat(frames, ignore_index=True)
    prices = (
        df_all
        .drop_duplicates(subset=["date", "asset"], keep="last")
        .pivot(index="date", columns="asset", values="close")
        .sort_index()
    )
    # # ---------------- 转宽表 ----------------
    # prices = (
    #     df_all
    #     .pivot(index="date", columns="asset", values="close")
    #     .sort_index()
    # )
    return prices

if __name__ == "__main__":
    # Example usage:
    # Load data for specific stocks within a date range
    print("--- Attempting to load data... ---")
    stocks_df = load_oss_stocks(codes=['000001', '600519'], start='2023-01-01', end='2023-01-31')
    if stocks_df.empty:
        print("--- Data loading resulted in an empty DataFrame. ---")
    else:
        print(stocks_df)
    print("--- Script finished. ---")
