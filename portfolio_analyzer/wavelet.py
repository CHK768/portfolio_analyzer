"""
小波分解模块：将组合日收益率的总方差分解到 5 个时间尺度。

支持两种方法：
  - modwt（默认）：最大重叠离散小波变换（平稳小波变换，pywt.swt）
    统计性质好，各成分方差之和严格等于总方差，适合金融时间序列。
  - cwt：连续小波变换（Morlet 小波，pywt.cwt）
    可精确对应指定天数（5/10/30），同时输出各尺度的时间平均能量占比。

时间尺度分组（交易日）：
  Band 1：1–2  天（cD1 / scales~1）     → "日内噪声"
  Band 2：2–8  天（cD2 / scales~2-8）   → "周频波动（≤5天）"
  Band 3：8–16 天（cD3 / scales~8-16）  → "双周波动（≤10天）"
  Band 4：16–64天（cD4+cD5 / ~16-64）  → "月频波动（≤30天）"
  Band 5：>64  天（cA5 / >64）          → "长期趋势（>30天）"
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pywt

BAND_NAMES = [
    "日内噪声   (≈1-2天)",
    "周频波动   (≈2-8天, ≤5天)",
    "双周波动   (≈8-16天, ≤10天)",
    "月频波动   (≈16-64天, ≤30天)",
    "长期趋势   (>64天, >30天)",
]

# MODWT 5 级分解所需的最小样本数（db4 滤波器长度=8，5级 → 8*2^5=256，保守取 64）
_MIN_SAMPLES_MODWT = 64
# 如果样本量不足以做 5 级，自动降级；最低 1 级需要 ~16 个样本
_MIN_SAMPLES_FLOOR = 16

# CWT 对应 5 个频段的 scale 边界（交易日）
_CWT_BAND_EDGES = [1, 2, 8, 16, 64]  # [s0, s1, s2, s3, s4, +inf]


def _max_modwt_level(signal: np.ndarray, target_level: int = 5) -> int:
    """
    计算给定信号长度下可执行的最大 SWT 层数（不超过 target_level）。
    SWT 要求信号长度是 2^level 的倍数，通过 swt_max_level 获取上限。
    如信号需要，自动 zero-pad 到最近的合法长度。
    """
    n = len(signal)
    max_level = pywt.swt_max_level(n)
    return min(max_level, target_level)


def wavelet_decompose_modwt(
    returns: pd.Series,
    wavelet: str = "db4",
) -> dict[str, float]:
    """
    用 MODWT（平稳小波变换）分解日收益率，返回各频段方差占比（%）。

    统计严格性：各频段方差之和 ≈ 信号总方差（pywt.swt 的正交分解保证）。

    Returns:
        dict {band_name: variance_pct}，值为百分比（0~100）
    """
    signal = returns.dropna().values.astype(float)
    n = len(signal)

    if n < _MIN_SAMPLES_FLOOR:
        raise ValueError(
            f"数据点数（{n}）过少，无法进行小波分解（至少需要 {_MIN_SAMPLES_FLOOR} 个交易日）"
        )

    # 去均值
    signal = signal - signal.mean()

    # 计算可用层数；若信号长度不是 2^5 的倍数，自动 pad 到最近合法长度
    target_level = 5
    required_len = 2 ** target_level
    if n % required_len != 0:
        pad_len = required_len - (n % required_len)
        signal = np.pad(signal, (0, pad_len), mode="wrap")

    level = _max_modwt_level(signal, target_level)

    if level < 5:
        warnings.warn(
            f"数据点数（{n}）不足以进行 5 级分解，自动降级为 {level} 级。"
            f"部分频段将合并或缺失。",
            stacklevel=2,
        )

    # pywt.swt 返回 [(cA_n, cD_n), ..., (cA_1, cD_1)]，从低频到高频排列
    # 即 coeffs[0] = (cAn, cDn)，coeffs[-1] = (cA1, cD1)
    coeffs = pywt.swt(signal, wavelet=wavelet, level=level, norm=True)
    # norm=True 使各层系数的 L2 范数与原信号一致，方差分解更准确

    # 提取各层 detail 系数（cD）及最终 approximation（cA）
    # coeffs[i] = (cA_{level-i}, cD_{level-i})
    # 层编号：coeffs[0] → level 级（最低频 detail），coeffs[-1] → 1 级（最高频 detail）
    # 重新整理为 [cD1, cD2, ..., cDn, cAn]（从高频到低频）
    details = [c[1] for c in reversed(coeffs)]   # cD1, cD2, ..., cDlevel
    approx = coeffs[0][0]                          # cA_level（最终近似）

    # 各成分方差
    variances = [float(np.var(d)) for d in details] + [float(np.var(approx))]
    total_var = sum(variances)

    if total_var < 1e-15:
        warnings.warn("组合收益率方差接近 0，小波分解结果无意义。", stacklevel=2)
        return {name: 0.0 for name in BAND_NAMES}

    # 将层级映射到 5 个频段（level 不足时合并）
    band_vars = _aggregate_bands(variances[:-1], variances[-1], level)
    return {name: v / total_var * 100 for name, v in zip(BAND_NAMES, band_vars)}


def _aggregate_bands(
    detail_vars: list[float],
    approx_var: float,
    level: int,
) -> list[float]:
    """
    将 level 层的方差聚合为固定的 5 个频段。

    level=5 时：
      Band1 = cD1, Band2 = cD2, Band3 = cD3, Band4 = cD4+cD5, Band5 = cA5

    level<5 时，高频层缺失部分方差归入相邻频段，低频 approx 保持。
    """
    # 补齐到 5 个 detail（不足的高频层填 0）
    padded = [0.0] * (5 - level) + list(detail_vars)  # index 0=cD1, 1=cD2, ...
    # padded[0] = cD1（最高频），padded[4] = cD5（最低频 detail）

    band1 = padded[0]               # cD1
    band2 = padded[1]               # cD2
    band3 = padded[2]               # cD3
    band4 = padded[3] + padded[4]   # cD4 + cD5
    band5 = approx_var              # cA5

    return [band1, band2, band3, band4, band5]


def wavelet_decompose_cwt(
    returns: pd.Series,
    scales: np.ndarray | None = None,
) -> dict[str, float]:
    """
    用 CWT（连续小波变换，Morlet）分解日收益率，返回各频段能量占比（%）。

    CWT 各频段有重叠（非正交），因此这里计算的是"能量贡献度"而非严格方差分解。
    输出值之和可能略不等于 100%，但比例关系有参考意义。

    Parameters:
        scales: 自定义 scale 数组（对应交易日周期），默认为 1~128 天

    Returns:
        dict {band_name: energy_pct}
    """
    signal = returns.dropna().values.astype(float)
    n = len(signal)

    if n < 32:
        raise ValueError(f"数据点数（{n}）过少，CWT 至少需要 32 个交易日")

    signal = signal - signal.mean()

    if scales is None:
        scales = np.arange(1, min(128, n // 2) + 1, dtype=float)

    # pywt.cwt 返回 (coef_matrix, freqs)
    # coef_matrix shape: (len(scales), n)
    coef, _ = pywt.cwt(signal, scales=scales, wavelet="morl")

    # 各 scale 的时间平均能量（功率）
    power = np.mean(np.abs(coef) ** 2, axis=1)  # shape: (len(scales),)

    # 按频段边界分组
    band_edges = _CWT_BAND_EDGES  # [1, 2, 8, 16, 64]
    band_power = []
    prev = 0.0
    for i, edge in enumerate(band_edges):
        mask = (scales >= (band_edges[i - 1] if i > 0 else 0)) & (scales < edge)
        band_power.append(float(power[mask].sum()) if mask.any() else 0.0)
    # 最后一段：scales >= 64
    mask_last = scales >= band_edges[-1]
    band_power.append(float(power[mask_last].sum()) if mask_last.any() else 0.0)

    total_power = sum(band_power)
    if total_power < 1e-15:
        return {name: 0.0 for name in BAND_NAMES}

    return {name: p / total_power * 100 for name, p in zip(BAND_NAMES, band_power)}


def wavelet_decompose(
    returns: pd.Series,
    method: str = "modwt",
    wavelet: str = "db4",
) -> dict[str, float]:
    """
    统一入口：选择 modwt 或 cwt 方法进行小波分解。

    Parameters:
        returns: 组合日收益率
        method:  "modwt"（默认，统计严谨）或 "cwt"（精确对应天数）
        wavelet: MODWT 使用的小波类型（cwt 固定使用 morl）

    Returns:
        dict {band_name: pct}，百分比
    """
    method = method.lower()
    if method == "modwt":
        return wavelet_decompose_modwt(returns, wavelet=wavelet)
    elif method == "cwt":
        return wavelet_decompose_cwt(returns)
    else:
        raise ValueError(f"未知的小波方法：{method}，可选 'modwt' 或 'cwt'")
