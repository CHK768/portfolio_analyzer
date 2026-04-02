"""
风险指标计算：
- 年化波动率
- 最大回撤
- 最大连续下跌天数
- Beta 系数（vs 基准指数）
- 个股相关性矩阵
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def annualized_volatility(returns: pd.Series, trading_days: int = 252) -> float:
    """年化波动率 = daily_std × √trading_days"""
    return float(returns.std() * math.sqrt(trading_days))


def max_drawdown(nav: pd.Series) -> float:
    """
    最大回撤 = max((peak - trough) / peak)

    Returns:
        float，范围 [0, 1]，例如 0.20 表示 20% 回撤
    """
    peak = nav.cummax()
    drawdown = (peak - nav) / peak
    return float(drawdown.max())


def max_consecutive_down_days(returns: pd.Series) -> int:
    """
    最大连续下跌天数：连续出现日收益率 < 0 的最长天数。
    """
    is_down = (returns < 0).values
    max_streak = 0
    current = 0
    for v in is_down:
        if v:
            current += 1
            if current > max_streak:
                max_streak = current
        else:
            current = 0
    return max_streak


def beta(
    portfolio_returns: pd.Series,
    index_returns: pd.Series,
    min_periods: int = 20,
) -> float:
    """
    Beta = Cov(R_p, R_m) / Var(R_m)

    两个 Series 按 inner join 对齐后计算。
    公共数据点 < min_periods 时返回 nan。
    """
    rp, rm = portfolio_returns.align(index_returns, join="inner")
    combined = pd.concat([rp, rm], axis=1).dropna()
    if len(combined) < min_periods:
        return float("nan")
    cov_mat = combined.cov()
    return float(cov_mat.iloc[0, 1] / cov_mat.iloc[1, 1])


def correlation_matrix(individual_rets: pd.DataFrame) -> pd.DataFrame:
    """
    计算各股票日收益率的皮尔逊相关系数矩阵。

    Parameters:
        individual_rets: DataFrame，列为股票代码，行为交易日

    Returns:
        DataFrame，对称相关矩阵，行列均为股票代码
    """
    return individual_rets.corr(method="pearson")
