"""
结果格式化输出：控制台表格 或 JSON。
"""

from __future__ import annotations

import json
import math

from tabulate import tabulate


def _pct(v: float, decimals: int = 2) -> str:
    return f"{v * 100:.{decimals}f}%"


def _na(v: float, fmt: str = ".4f") -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return format(v, fmt)


def format_table(result: dict) -> str:
    lines = []

    # ── 组合构成 ──────────────────────────────────────────────
    portfolio = result["portfolio"]
    lines.append("=" * 52)
    lines.append("  组合分析报告")
    lines.append("=" * 52)
    lines.append(f"  分析区间：{portfolio['start']} ~ {portfolio['end']}")
    lines.append(f"  实际交易日：{portfolio['trading_days']} 天")
    lines.append("")

    rows = [
        [code, f"{w * 100:.1f}%"]
        for code, w in zip(portfolio["codes"], portfolio["weights"])
    ]
    lines.append(tabulate(rows, headers=["股票代码", "权重"], tablefmt="simple"))
    lines.append("")

    # ── 风险指标 ──────────────────────────────────────────────
    m = result["metrics"]
    lines.append("── 风险指标 " + "─" * 40)
    risk_rows = [
        ["年化波动率",          _pct(m["volatility"])],
        ["最大回撤",            _pct(m["max_drawdown"])],
        ["最大连续下跌天数",    f"{m['max_consecutive_down_days']} 天"],
        ["Beta（vs 上证指数）", _na(m["beta"])],
    ]
    lines.append(tabulate(risk_rows, tablefmt="simple"))
    lines.append("")

    # ── 个股相关性矩阵 ────────────────────────────────────────
    corr = result.get("correlation")
    if corr is not None:
        lines.append("── 个股相关性矩阵（日收益率皮尔逊相关系数）" + "─" * 10)
        codes = portfolio["codes"]
        headers = [""] + codes
        corr_rows = [
            [code] + [f"{corr[code][other]:.3f}" for other in codes]
            for code in codes
        ]
        lines.append(tabulate(corr_rows, headers=headers, tablefmt="simple"))
        lines.append("")

    # ── 小波分解 ──────────────────────────────────────────────
    wavelet = result.get("wavelet")
    if wavelet is not None:
        method_label = result.get("wavelet_method", "modwt").upper()
        lines.append(f"── 波动来源分析（小波分解，{method_label}）" + "─" * 20)
        wv_rows = [
            [band, f"{pct:.1f}%"]
            for band, pct in wavelet.items()
        ]
        lines.append(tabulate(wv_rows, headers=["时间尺度", "方差占比"], tablefmt="simple"))
        lines.append("")
    elif result.get("wavelet_skipped"):
        lines.append(f"[提示] 小波分解已跳过：{result['wavelet_skipped']}")
        lines.append("")

    lines.append("=" * 52)
    return "\n".join(lines)


def format_json(result: dict) -> str:
    return json.dumps(result, ensure_ascii=False, indent=2, default=str)
