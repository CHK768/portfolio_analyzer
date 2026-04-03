"""FastAPI 服务：组合质量分析（局域网访问）"""
from __future__ import annotations

import logging
import math
import warnings
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator, model_validator

from portfolio_analyzer.data_loader import (
    DEFAULT_CACHE_DIR,
    DEFAULT_RAW_DIR,
    InsufficientDataError,
    StockNotFoundError,
    fetch_index,
    load_stocks_aligned,
)
from portfolio_analyzer.metrics import (
    alpha,
    annualized_volatility,
    beta,
    correlation_matrix,
    max_consecutive_down_days,
    max_drawdown,
)
from portfolio_analyzer.portfolio import build_nav, build_returns, individual_returns
from portfolio_analyzer.wavelet import wavelet_decompose

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pa-server")

RAW_DIR = DEFAULT_RAW_DIR
CACHE_DIR = DEFAULT_CACHE_DIR
STOCK_LIST_CSV = Path("/Users/10259879/stock_lookalike/data/stock_list.csv")

STOCK_NAMES: dict[str, str] = {}   # code -> name，启动时加载
STOCKS_ALL:  list[dict]    = []    # [{code, name, pinyin}]，供前端全量搜索


# ─── 请求/响应模型 ──────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    codes: List[str]
    weights: Optional[List[float]] = None
    start: str
    end: Optional[str] = None
    wavelet_method: str = "modwt"
    wavelet_type: str = "db4"
    index_symbol: str = "sh000001"
    no_wavelet: bool = False
    force_refresh: bool = False
    include_nav: bool = False

    @field_validator("codes")
    @classmethod
    def codes_not_empty(cls, v):
        codes = [c for token in v for c in token.split()]
        if not codes:
            raise ValueError("至少需要一只股票")
        return codes

    @field_validator("wavelet_method")
    @classmethod
    def valid_method(cls, v):
        if v.lower() not in ("modwt", "cwt"):
            raise ValueError("wavelet_method 可选 modwt 或 cwt")
        return v.lower()

    @model_validator(mode="after")
    def normalize_weights(self):
        codes = self.codes
        if self.weights is None:
            self.weights = [1.0 / len(codes)] * len(codes)
        else:
            if len(self.weights) != len(codes):
                raise ValueError(
                    f"weights 数量（{len(self.weights)}）与 codes 数量（{len(codes)}）不匹配"
                )
            total = sum(self.weights)
            if abs(total - 1.0) > 1e-6:
                self.weights = [w / total for w in self.weights]
        if self.end is None:
            self.end = datetime.today().strftime("%Y-%m-%d")
        return self


# ─── 应用 ──────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global STOCK_NAMES
    logger.info(f"数据目录: {RAW_DIR}")
    n = len(list(RAW_DIR.glob("*.parquet"))) if RAW_DIR.exists() else 0
    logger.info(f"可用股票数: {n}")
    if STOCK_LIST_CSV.exists():
        try:
            from pypinyin import lazy_pinyin, Style
            df_names = pd.read_csv(STOCK_LIST_CSV, dtype=str)
            for _, row in df_names.iterrows():
                code = str(row["code"]).strip()
                name = str(row["name"]).strip()
                pinyin = "".join(lazy_pinyin(name, style=Style.FIRST_LETTER)).lower()
                STOCK_NAMES[code] = name
                STOCKS_ALL.append({"code": code, "name": name, "pinyin": pinyin})
            logger.info(f"股票名称及拼音加载完成: {len(STOCK_NAMES)} 条")
        except Exception as e:
            logger.warning(f"股票名称加载失败: {e}")
    yield


app = FastAPI(
    title="Portfolio Analyzer",
    description=(
        "A股组合质量分析服务\n\n"
        "## 功能\n"
        "- **年化波动率** — `std × √252`\n"
        "- **最大回撤** — `(峰值 − 谷值) / 峰值`\n"
        "- **最大连续下跌天数**\n"
        "- **Beta 系数** — `Cov(Rp, Rm) / Var(Rm)`，基准默认上证指数\n"
        "- **个股相关性矩阵** — 皮尔逊相关系数\n"
        "- **小波分解** — MODWT（统计严谨，默认）或 CWT（Morlet）\n\n"
        "## 小波频段\n"
        "| 频段 | 周期 |\n"
        "|------|------|\n"
        "| 日内噪声 | ≈1-2天 |\n"
        "| 周频波动 | ≈2-8天，≤5天 |\n"
        "| 双周波动 | ≈8-16天，≤10天 |\n"
        "| 月频波动 | ≈16-64天，≤30天 |\n"
        "| 长期趋势 | >64天，>30天 |\n"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ─── 端点 ──────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    n = len(list(RAW_DIR.glob("*.parquet"))) if RAW_DIR.exists() else 0
    return {"status": "ok", "raw_dir": str(RAW_DIR), "stocks_available": n}


@app.get("/stocks")
def list_stocks(q: Optional[str] = None):
    """
    列出可用的股票代码。

    - **q**: 可选过滤前缀（如 `q=600` 只返回 600 开头的代码）
    """
    if not RAW_DIR.exists():
        return {"stocks": [], "codes": [], "count": 0}
    codes = sorted(p.stem for p in RAW_DIR.glob("*.parquet"))
    if q:
        codes = [c for c in codes if c.startswith(q)]
    stocks = [{"code": c, "name": STOCK_NAMES.get(c, "")} for c in codes]
    return {"stocks": stocks, "codes": codes, "count": len(codes)}


@app.get("/stocks/all", include_in_schema=False)
def all_stocks():
    """返回全量股票列表（含拼音缩写），供前端客户端搜索。"""
    return {"stocks": STOCKS_ALL, "count": len(STOCKS_ALL)}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    """
    分析股票组合的风险指标与波动来源。

    **请求示例：**
    ```json
    {
      "codes": ["000001", "600519", "300750"],
      "weights": [0.3, 0.4, 0.3],
      "start": "2024-01-01",
      "end": "2026-03-01"
    }
    ```

    - **codes**: 股票代码列表
    - **weights**: 各股票权重（不填则等权重；非归一化时自动归一化）
    - **start / end**: 分析区间（end 不填默认今天）
    - **wavelet_method**: `modwt`（默认，统计严谨）或 `cwt`（精确对应天数）
    - **wavelet_type**: MODWT 小波类型，默认 `db4`
    - **index_symbol**: Beta 基准指数，默认 `sh000001`（上证）
    - **no_wavelet**: `true` 则跳过小波分解
    - **force_refresh**: `true` 则强制重新拉取指数数据
    - **include_nav**: `true` 则在结果中附带净值曲线序列
    """
    codes = req.codes
    weights = req.weights

    # 1. 加载股票数据
    try:
        price_df = load_stocks_aligned(codes, req.start, req.end, RAW_DIR)
    except StockNotFoundError as e:
        raise HTTPException(404, detail=str(e))
    except InsufficientDataError as e:
        raise HTTPException(422, detail=str(e))

    # 2. 构建组合收益率 & NAV
    port_returns = build_returns(price_df, weights)
    nav = build_nav(port_returns)
    indiv_rets = individual_returns(price_df)
    trading_days = len(port_returns)

    # 3. 风险指标
    vol = annualized_volatility(port_returns)
    mdd = max_drawdown(nav)
    mcdd = max_consecutive_down_days(port_returns)

    beta_val = float("nan")
    alpha_val = float("nan")
    beta_error = None
    index_rets = None
    try:
        index_close = fetch_index(
            symbol=req.index_symbol,
            cache_dir=CACHE_DIR,
            force_refresh=req.force_refresh,
        )
        index_rets = index_close.pct_change().dropna()
        beta_val = beta(port_returns, index_rets)
        alpha_val = alpha(port_returns, index_rets)
    except Exception as e:
        beta_error = str(e)
        logger.warning(f"Beta 计算失败: {e}")

    # 4. 相关性矩阵
    corr = None
    if len(codes) > 1:
        corr_df = correlation_matrix(indiv_rets)
        corr = {
            col: {row: round(v, 4) for row, v in corr_df[col].items()}
            for col in corr_df.columns
        }

    # 5. 小波分解
    wavelet_result = None
    wavelet_index_result = None
    wavelet_skipped = None
    if not req.no_wavelet:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                wavelet_result = wavelet_decompose(
                    port_returns,
                    method=req.wavelet_method,
                    wavelet=req.wavelet_type,
                )
            if caught:
                logger.warning(f"小波分解警告: {caught[-1].message}")
        except ValueError as e:
            wavelet_skipped = str(e)
        except Exception as e:
            wavelet_skipped = str(e)
            logger.error(f"小波分解失败: {e}")
        # 5b. 指数小波分解（作为参照基准）
        if index_rets is not None:
            try:
                idx_aligned = index_rets.reindex(port_returns.index).fillna(0)
                wavelet_index_result = wavelet_decompose(
                    idx_aligned,
                    method=req.wavelet_method,
                    wavelet=req.wavelet_type,
                )
            except Exception as e:
                logger.warning(f"指数小波分解失败: {e}")

    # 6. 组装结果
    result = {
        "portfolio": {
            "codes": codes,
            "names": {c: STOCK_NAMES.get(c, "") for c in codes},
            "weights": [round(w, 6) for w in weights],
            "start": port_returns.index[0].strftime("%Y-%m-%d") if trading_days else req.start,
            "end": port_returns.index[-1].strftime("%Y-%m-%d") if trading_days else req.end,
            "trading_days": trading_days,
        },
        "metrics": {
            "volatility": _fmt(vol),
            "max_drawdown": _fmt(mdd),
            "max_consecutive_down_days": mcdd,
            "beta": _fmt(beta_val),
            "alpha": _fmt(alpha_val),
            "beta_index": req.index_symbol,
            "beta_error": beta_error,
        },
        "correlation": corr,
        "wavelet": {
            "method": req.wavelet_method,
            "bands": {k: round(v, 2) for k, v in wavelet_result.items()} if wavelet_result else None,
            "index_bands": {k: round(v, 2) for k, v in wavelet_index_result.items()} if wavelet_index_result else None,
            "index_symbol": req.index_symbol,
            "skipped": wavelet_skipped,
        },
    }

    if req.include_nav:
        result["nav"] = {
            d.strftime("%Y-%m-%d"): round(v, 6)
            for d, v in nav.items()
        }

    return JSONResponse(result)


def _fmt(v: float) -> float | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return round(v, 6)


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/static/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8001, reload=False)
