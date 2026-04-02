"""
数据加载模块：
- 从 stock_lookalike 的 parquet 缓存读取股票数据
- 获取上证指数（增量缓存）
- 多股票对齐到统一日期索引
"""

from __future__ import annotations

from pathlib import Path

import akshare as ak
import pandas as pd

# stock_lookalike 的原始数据目录（默认值，可通过参数覆盖）
DEFAULT_RAW_DIR = Path("/Users/10259879/stock_lookalike/data/raw")
DEFAULT_CACHE_DIR = Path("/Users/10259879/portfolio_analyzer/data")


class StockNotFoundError(FileNotFoundError):
    pass


class InsufficientDataError(ValueError):
    pass


def load_stock(
    code: str,
    start_date: str,
    end_date: str,
    raw_dir: Path = DEFAULT_RAW_DIR,
) -> pd.Series:
    """
    从 parquet 缓存读取单只股票的收盘价序列。

    Returns:
        pd.Series，index 为 DatetimeIndex，name 为股票代码
    Raises:
        StockNotFoundError: parquet 文件不存在
        InsufficientDataError: 指定区间内无数据
    """
    path = Path(raw_dir) / f"{code}.parquet"
    if not path.exists():
        raise StockNotFoundError(f"股票 {code} 的数据文件不存在：{path}")

    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    df = df.loc[start:end, ["close"]]

    if df.empty:
        raise InsufficientDataError(
            f"股票 {code} 在 {start_date} ~ {end_date} 区间内无数据"
        )

    series = df["close"].rename(code)
    return series


def fetch_index(
    symbol: str = "sh000001",
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force_refresh: bool = False,
) -> pd.Series:
    """
    获取指数日线数据，并增量缓存到本地 parquet。

    Returns:
        pd.Series，index 为 DatetimeIndex，name 为 symbol（收盘价）
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{symbol}.parquet"

    today = pd.Timestamp.today().normalize()

    # 尝试使用缓存
    if cache_path.exists() and not force_refresh:
        cached = pd.read_parquet(cache_path)
        cached["date"] = pd.to_datetime(cached["date"])
        cached_end = cached["date"].max()
        if cached_end >= today - pd.Timedelta(days=1):
            return cached.set_index("date")["close"].rename(symbol)

    # 拉取全量数据
    try:
        df_new = ak.stock_zh_index_daily(symbol=symbol)
    except Exception as e:
        if cache_path.exists():
            import warnings
            warnings.warn(f"指数数据拉取失败（{e}），使用本地缓存")
            cached = pd.read_parquet(cache_path)
            cached["date"] = pd.to_datetime(cached["date"])
            return cached.set_index("date")["close"].rename(symbol)
        raise

    df_new = df_new.rename(columns={"date": "date"})
    df_new["date"] = pd.to_datetime(df_new["date"])
    df_new = df_new[["date", "close"]].drop_duplicates("date").sort_values("date")

    # 合并旧缓存（若存在）
    if cache_path.exists() and not force_refresh:
        cached = pd.read_parquet(cache_path)
        cached["date"] = pd.to_datetime(cached["date"])
        df_all = (
            pd.concat([cached[["date", "close"]], df_new])
            .drop_duplicates("date")
            .sort_values("date")
            .reset_index(drop=True)
        )
    else:
        df_all = df_new.reset_index(drop=True)

    df_all.to_parquet(cache_path, index=False)
    return df_all.set_index("date")["close"].rename(symbol)


def load_stocks_aligned(
    codes: list[str],
    start_date: str,
    end_date: str,
    raw_dir: Path = DEFAULT_RAW_DIR,
) -> pd.DataFrame:
    """
    读取多只股票，对齐到所有股票的日期并集，停牌日为 NaN。

    Returns:
        DataFrame，index 为 DatetimeIndex，列为股票代码（收盘价）
    """
    series_list = []
    for code in codes:
        s = load_stock(code, start_date, end_date, raw_dir)
        series_list.append(s)

    price_df = pd.concat(series_list, axis=1)
    price_df = price_df.sort_index()
    return price_df
