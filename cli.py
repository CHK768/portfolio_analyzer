#!/usr/bin/env python3
"""
组合质量分析工具

示例：
  python cli.py analyze --stocks 000001 600519 300750 \\
                        --weights 0.3 0.4 0.3 \\
                        --start 2024-01-01 --end 2026-03-01

  python cli.py analyze --stocks 000001 600519 \\
                        --start 2024-01-01 --format json

  python cli.py analyze --stocks 000001 600519 300750 \\
                        --start 2024-01-01 --wavelet-method cwt
"""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path

import click

from portfolio_analyzer.data_loader import (
    DEFAULT_CACHE_DIR,
    DEFAULT_RAW_DIR,
    InsufficientDataError,
    StockNotFoundError,
    fetch_index,
    load_stocks_aligned,
)
from portfolio_analyzer.metrics import (
    annualized_volatility,
    beta,
    correlation_matrix,
    max_consecutive_down_days,
    max_drawdown,
)
from portfolio_analyzer.portfolio import build_nav, build_returns, individual_returns
from portfolio_analyzer.reporter import format_json, format_table
from portfolio_analyzer.wavelet import wavelet_decompose


@click.group()
def cli():
    """组合质量分析工具 — 分析股票组合的风险特征与波动来源。"""


@cli.command("analyze")
@click.option(
    "--stocks", "-s",
    required=True,
    multiple=True,
    help="股票代码，可多次指定（-s 000001 -s 600519）或空格分隔（--stocks '000001 600519'）",
)
@click.option(
    "--weights", "-w",
    required=False,
    multiple=True,
    type=float,
    help="各股票权重（与 --stocks 顺序一致）。不指定则等权重。",
)
@click.option("--start", required=True, help="分析起始日期，格式 YYYY-MM-DD")
@click.option("--end", default=None, help="分析结束日期，格式 YYYY-MM-DD（默认今天）")
@click.option(
    "--wavelet-method",
    default="modwt",
    show_default=True,
    type=click.Choice(["modwt", "cwt"], case_sensitive=False),
    help="小波分解方法：modwt（统计严谨，推荐）| cwt（精确对应天数）",
)
@click.option(
    "--wavelet-type",
    default="db4",
    show_default=True,
    help="MODWT 使用的小波类型（db4/db6/haar 等，仅 modwt 方法有效）",
)
@click.option(
    "--index-symbol",
    default="sh000001",
    show_default=True,
    help="基准指数代码（用于 Beta 计算）",
)
@click.option(
    "--format", "fmt",
    default="table",
    show_default=True,
    type=click.Choice(["table", "json"], case_sensitive=False),
    help="输出格式",
)
@click.option(
    "--raw-dir",
    default=str(DEFAULT_RAW_DIR),
    show_default=True,
    help="股票 parquet 数据目录",
)
@click.option(
    "--cache-dir",
    default=str(DEFAULT_CACHE_DIR),
    show_default=True,
    help="指数缓存目录",
)
@click.option(
    "--force-refresh",
    is_flag=True,
    default=False,
    help="强制重新拉取指数数据",
)
@click.option(
    "--no-wavelet",
    is_flag=True,
    default=False,
    help="跳过小波分解",
)
def analyze(
    stocks, weights, start, end, wavelet_method, wavelet_type,
    index_symbol, fmt, raw_dir, cache_dir, force_refresh, no_wavelet,
):
    """分析股票组合的风险指标与波动来源。"""

    # ── 1. 参数处理 ────────────────────────────────────────────
    codes = [c for token in stocks for c in token.split()]
    if not codes:
        raise click.UsageError("至少需要指定一只股票（--stocks）")

    if weights:
        weights = list(weights)
        if len(weights) != len(codes):
            raise click.UsageError(
                f"权重数量（{len(weights)}）与股票数量（{len(codes)}）不匹配"
            )
        total_w = sum(weights)
        if abs(total_w - 1.0) > 1e-6:
            click.echo(
                f"[警告] 权重之和为 {total_w:.4f}，已自动归一化。", err=True
            )
            weights = [w / total_w for w in weights]
    else:
        weights = [1.0 / len(codes)] * len(codes)

    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    # ── 2. 加载股票数据 ────────────────────────────────────────
    try:
        price_df = load_stocks_aligned(codes, start, end, Path(raw_dir))
    except StockNotFoundError as e:
        raise click.ClickException(str(e))
    except InsufficientDataError as e:
        raise click.ClickException(str(e))

    # ── 3. 构建组合收益率 & NAV ─────────────────────────────────
    port_returns = build_returns(price_df, weights)
    nav = build_nav(port_returns)
    indiv_rets = individual_returns(price_df)

    trading_days = len(port_returns)

    # ── 4. 风险指标 ────────────────────────────────────────────
    vol = annualized_volatility(port_returns)
    mdd = max_drawdown(nav)
    mcdd = max_consecutive_down_days(port_returns)

    # Beta：获取基准指数
    beta_val = float("nan")
    try:
        index_close = fetch_index(
            symbol=index_symbol,
            cache_dir=Path(cache_dir),
            force_refresh=force_refresh,
        )
        index_rets = index_close.pct_change().dropna()
        beta_val = beta(port_returns, index_rets)
    except Exception as e:
        click.echo(f"[警告] Beta 计算失败（{e}），结果显示 N/A。", err=True)

    # ── 5. 相关性矩阵 ───────────────────────────────────────────
    corr = None
    if len(codes) > 1:
        corr_df = correlation_matrix(indiv_rets)
        corr = corr_df.to_dict()

    # ── 6. 小波分解 ────────────────────────────────────────────
    wavelet_result = None
    wavelet_skipped = None

    if not no_wavelet:
        try:
            wavelet_result = wavelet_decompose(
                port_returns,
                method=wavelet_method,
                wavelet=wavelet_type,
            )
        except ValueError as e:
            wavelet_skipped = str(e)
            click.echo(f"[警告] 小波分解跳过：{e}", err=True)
        except Exception as e:
            wavelet_skipped = str(e)
            click.echo(f"[警告] 小波分解失败：{e}", err=True)

    # ── 7. 组装结果 ────────────────────────────────────────────
    result = {
        "portfolio": {
            "codes": codes,
            "weights": weights,
            "start": start,
            "end": end,
            "trading_days": trading_days,
        },
        "metrics": {
            "volatility": vol,
            "max_drawdown": mdd,
            "max_consecutive_down_days": mcdd,
            "beta": beta_val,
        },
        "correlation": corr,
        "wavelet": wavelet_result,
        "wavelet_method": wavelet_method,
        "wavelet_skipped": wavelet_skipped,
    }

    # ── 8. 输出 ────────────────────────────────────────────────
    if fmt == "table":
        click.echo(format_table(result))
    else:
        click.echo(format_json(result))


if __name__ == "__main__":
    cli()
