"""
组合收益率与净值曲线构建。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_returns(
    price_df: pd.DataFrame,
    weights: list[float],
) -> pd.Series:
    """
    计算组合日收益率。停牌日（NaN）的该股收益率计为 0，权重不变。

    Parameters:
        price_df: 各股票收盘价，index=DatetimeIndex，列=股票代码
        weights:  与 price_df 列顺序一致的权重列表（应归一化为 1）

    Returns:
        pd.Series，组合日收益率，index=DatetimeIndex
    """
    daily_rets = price_df.pct_change()
    # 停牌 / 缺失数据收益率视为 0
    daily_rets = daily_rets.fillna(0.0)
    # 去掉第一行（pct_change 产生的 NaN 行）
    daily_rets = daily_rets.iloc[1:]

    w = np.array(weights, dtype=float)
    portfolio_ret = (daily_rets.values * w).sum(axis=1)
    return pd.Series(portfolio_ret, index=daily_rets.index, name="portfolio")


def build_nav(returns: pd.Series) -> pd.Series:
    """
    从日收益率构建累积净值曲线（初始值 = 1.0）。

    Returns:
        pd.Series，NAV 曲线，index=DatetimeIndex
    """
    return (1 + returns).cumprod().rename("nav")


def individual_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算各股票日收益率（停牌计 0）。

    Returns:
        DataFrame，同 price_df 结构，index=DatetimeIndex
    """
    rets = price_df.pct_change().fillna(0.0).iloc[1:]
    return rets
