"""
Quantitative Analysis
---------------------
Factor exposure, risk metrics (Sharpe, Sortino, VaR, CVaR, max drawdown), beta.
"""

import pandas as pd
import numpy as np
from scipy import stats


def returns_from_prices(prices: pd.Series) -> pd.Series:
    return prices.pct_change().dropna()


def log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices / prices.shift(1)).dropna()


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    if len(returns) == 0:
        return 0
    cum = (1 + returns).prod()
    years = len(returns) / periods_per_year
    if years <= 0 or cum <= 0:
        return 0
    return cum ** (1 / years) - 1


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    return returns.std() * np.sqrt(periods_per_year)


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.045, periods_per_year: int = 252) -> float:
    if returns.std() == 0:
        return 0
    excess = returns - risk_free / periods_per_year
    return excess.mean() / returns.std() * np.sqrt(periods_per_year)


def sortino_ratio(returns: pd.Series, risk_free: float = 0.045, periods_per_year: int = 252) -> float:
    excess = returns - risk_free / periods_per_year
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0
    return excess.mean() / downside.std() * np.sqrt(periods_per_year)


def max_drawdown(prices: pd.Series) -> dict:
    cummax = prices.cummax()
    drawdown = (prices - cummax) / cummax
    max_dd = drawdown.min()
    trough_date = drawdown.idxmin()
    peak_date = prices[:trough_date].idxmax() if trough_date else None
    return {"max_drawdown": max_dd, "peak_date": peak_date, "trough_date": trough_date, "drawdown_series": drawdown}


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR."""
    if len(returns) == 0:
        return 0
    return np.percentile(returns, (1 - confidence) * 100)


def conditional_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """CVaR / Expected Shortfall — average loss in tail."""
    var = value_at_risk(returns, confidence)
    tail = returns[returns <= var]
    return tail.mean() if len(tail) > 0 else var


def beta_alpha(stock_returns: pd.Series, market_returns: pd.Series) -> dict:
    """OLS regression of stock vs market for beta and alpha."""
    aligned = pd.concat([stock_returns, market_returns], axis=1, join="inner").dropna()
    if len(aligned) < 30:
        return {"beta": None, "alpha": None, "r_squared": None}
    aligned.columns = ["stock", "market"]
    slope, intercept, r_value, _, _ = stats.linregress(aligned["market"], aligned["stock"])
    return {
        "beta": slope,
        "alpha": intercept * 252,  # annualized alpha
        "r_squared": r_value ** 2,
        "correlation": aligned["stock"].corr(aligned["market"]),
    }


def calmar_ratio(returns: pd.Series, prices: pd.Series, periods_per_year: int = 252) -> float:
    ann_ret = annualized_return(returns, periods_per_year)
    mdd = max_drawdown(prices)["max_drawdown"]
    if mdd == 0:
        return 0
    return ann_ret / abs(mdd)


def factor_exposure(stock_prices: pd.Series, market_prices: pd.Series, info: dict) -> dict:
    """Approximate factor scoring (value, quality, momentum, low-vol, size)."""
    stock_ret = returns_from_prices(stock_prices)
    market_ret = returns_from_prices(market_prices)

    # Momentum: 12-1 month return
    if len(stock_prices) >= 252:
        twelve_one = (stock_prices.iloc[-21] / stock_prices.iloc[-252]) - 1
        market_twelve_one = (market_prices.iloc[-21] / market_prices.iloc[-252]) - 1
        momentum_excess = twelve_one - market_twelve_one
    else:
        twelve_one = None
        momentum_excess = None

    # Low-vol: realized 1Y vol
    realized_vol = annualized_volatility(stock_ret) if len(stock_ret) > 0 else None

    # Value: composite of low P/E, low P/B, low EV/EBITDA
    pe = info.get("trailingPE")
    pb = info.get("priceToBook")
    ev_ebitda = info.get("enterpriseToEbitda")
    value_signals = []
    if pe and pe > 0:
        value_signals.append("low" if pe < 15 else "high" if pe > 30 else "neutral")
    if pb and pb > 0:
        value_signals.append("low" if pb < 2 else "high" if pb > 5 else "neutral")
    if ev_ebitda and ev_ebitda > 0:
        value_signals.append("low" if ev_ebitda < 10 else "high" if ev_ebitda > 20 else "neutral")
    value_score = "Cheap (value)" if value_signals.count("low") > value_signals.count("high") else "Expensive (growth)" if value_signals.count("high") > value_signals.count("low") else "Neutral"

    # Quality: ROE, margins, debt
    roe = info.get("returnOnEquity")
    op_margin = info.get("operatingMargins")
    quality_signals = []
    if roe and roe > 0.15:
        quality_signals.append("high")
    elif roe and roe > 0.05:
        quality_signals.append("med")
    else:
        quality_signals.append("low")
    if op_margin and op_margin > 0.20:
        quality_signals.append("high")
    elif op_margin and op_margin > 0.08:
        quality_signals.append("med")
    else:
        quality_signals.append("low")
    quality_score = "High Quality" if quality_signals.count("high") >= 1 else "Average Quality" if quality_signals.count("med") >= 1 else "Low Quality"

    # Size
    market_cap = info.get("marketCap", 0)
    if market_cap > 200e9:
        size = "Mega Cap"
    elif market_cap > 10e9:
        size = "Large Cap"
    elif market_cap > 2e9:
        size = "Mid Cap"
    elif market_cap > 300e6:
        size = "Small Cap"
    else:
        size = "Micro Cap"

    return {
        "momentum_12_1": twelve_one,
        "momentum_excess_vs_market": momentum_excess,
        "realized_volatility": realized_vol,
        "value": value_score,
        "quality": quality_score,
        "size": size,
        "market_cap": market_cap,
    }


def full_risk_metrics(prices: pd.Series, market_prices: pd.Series, risk_free: float = 0.045) -> dict:
    """Compute the full risk metrics suite."""
    if len(prices) < 30:
        return {"insufficient_data": True}

    returns = returns_from_prices(prices)
    market_returns = returns_from_prices(market_prices) if len(market_prices) > 30 else pd.Series()

    metrics = {
        "annualized_return": annualized_return(returns),
        "annualized_volatility": annualized_volatility(returns),
        "sharpe_ratio": sharpe_ratio(returns, risk_free),
        "sortino_ratio": sortino_ratio(returns, risk_free),
        "calmar_ratio": calmar_ratio(returns, prices),
        "max_drawdown": max_drawdown(prices)["max_drawdown"],
        "var_95": value_at_risk(returns, 0.95),
        "cvar_95": conditional_var(returns, 0.95),
        "var_99": value_at_risk(returns, 0.99),
        "cvar_99": conditional_var(returns, 0.99),
        "skewness": returns.skew(),
        "kurtosis": returns.kurtosis(),
    }
    if len(market_returns) > 30:
        ba = beta_alpha(returns, market_returns)
        metrics.update(ba)
    return metrics
