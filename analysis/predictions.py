"""
Fundamentals Analysis
---------------------
DCF valuation with bear/base/bull scenarios, ratio analysis, quality scoring.
"""

import pandas as pd
import numpy as np


def safe_get(info: dict, *keys, default=None):
    """Try multiple keys, return first non-None."""
    for k in keys:
        v = info.get(k)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            return v
    return default


def calculate_wacc(info: dict, risk_free_rate: float, market_premium: float = 0.055) -> dict:
    """Compute WACC using CAPM for cost of equity."""
    beta = safe_get(info, "beta", default=1.0)
    cost_of_equity = risk_free_rate + beta * market_premium

    market_cap = safe_get(info, "marketCap", default=0)
    total_debt = safe_get(info, "totalDebt", default=0)
    total_value = market_cap + total_debt

    if total_value == 0:
        return {"wacc": cost_of_equity, "cost_of_equity": cost_of_equity, "cost_of_debt": 0, "weight_equity": 1, "weight_debt": 0}

    weight_equity = market_cap / total_value
    weight_debt = total_debt / total_value

    # Estimate cost of debt from interest expense / total debt, fallback to risk-free + spread
    cost_of_debt = risk_free_rate + 0.02  # 200bp spread default
    tax_rate = 0.21  # US corporate

    wacc = (weight_equity * cost_of_equity) + (weight_debt * cost_of_debt * (1 - tax_rate))

    return {
        "wacc": wacc,
        "cost_of_equity": cost_of_equity,
        "cost_of_debt": cost_of_debt,
        "weight_equity": weight_equity,
        "weight_debt": weight_debt,
        "beta": beta,
    }


def estimate_fcf(cashflow: pd.DataFrame, info: dict) -> float:
    """Get most recent free cash flow. Try multiple sources."""
    # Try yfinance freeCashflow first
    fcf = safe_get(info, "freeCashflow")
    if fcf:
        return float(fcf)

    # Calculate from cashflow statement: CFO - CapEx
    if cashflow is not None and not cashflow.empty:
        try:
            cfo_keys = ["Operating Cash Flow", "Total Cash From Operating Activities", "Cash Flow From Continuing Operating Activities"]
            capex_keys = ["Capital Expenditure", "Capital Expenditures", "Purchase Of PPE"]

            cfo = None
            capex = None
            for k in cfo_keys:
                if k in cashflow.index:
                    cfo = cashflow.loc[k].iloc[0]
                    break
            for k in capex_keys:
                if k in cashflow.index:
                    capex = cashflow.loc[k].iloc[0]
                    break

            if cfo is not None and capex is not None:
                return float(cfo) + float(capex)  # capex is usually negative
            if cfo is not None:
                return float(cfo)
        except Exception:
            pass
    return 0.0


def dcf_valuation(
    base_fcf: float,
    wacc: float,
    growth_high: float,
    growth_terminal: float,
    years_high: int = 5,
    shares_outstanding: float = 0,
    net_debt: float = 0,
) -> dict:
    """Two-stage DCF. Returns intrinsic equity value and per-share value."""
    if base_fcf <= 0 or wacc <= growth_terminal or shares_outstanding <= 0:
        return {"valid": False, "intrinsic_per_share": None, "enterprise_value": None}

    # Stage 1: explicit forecast period
    pv_explicit = 0
    for year in range(1, years_high + 1):
        fcf_t = base_fcf * (1 + growth_high) ** year
        pv_explicit += fcf_t / (1 + wacc) ** year

    # Stage 2: terminal value (Gordon Growth)
    fcf_terminal = base_fcf * (1 + growth_high) ** years_high * (1 + growth_terminal)
    terminal_value = fcf_terminal / (wacc - growth_terminal)
    pv_terminal = terminal_value / (1 + wacc) ** years_high

    enterprise_value = pv_explicit + pv_terminal
    equity_value = enterprise_value - net_debt
    per_share = equity_value / shares_outstanding

    return {
        "valid": True,
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "intrinsic_per_share": per_share,
        "pv_explicit": pv_explicit,
        "pv_terminal": pv_terminal,
        "terminal_pct": pv_terminal / enterprise_value if enterprise_value > 0 else 0,
    }


def dcf_scenarios(info: dict, cashflow: pd.DataFrame, wacc: float) -> dict:
    """Run bear/base/bull DCF scenarios."""
    base_fcf = estimate_fcf(cashflow, info)
    shares = safe_get(info, "sharesOutstanding", "impliedSharesOutstanding", default=0)
    total_debt = safe_get(info, "totalDebt", default=0)
    cash = safe_get(info, "totalCash", default=0)
    net_debt = total_debt - cash

    # Use historical revenue growth as anchor for base case
    base_growth_anchor = safe_get(info, "revenueGrowth", default=0.05)
    if base_growth_anchor is None or base_growth_anchor < -0.5 or base_growth_anchor > 1.0:
        base_growth_anchor = 0.05

    scenarios = {
        "bear": {"growth_high": max(base_growth_anchor - 0.05, -0.02), "growth_terminal": 0.02},
        "base": {"growth_high": base_growth_anchor, "growth_terminal": 0.025},
        "bull": {"growth_high": base_growth_anchor + 0.05, "growth_terminal": 0.03},
    }

    results = {}
    for name, params in scenarios.items():
        results[name] = dcf_valuation(
            base_fcf=base_fcf,
            wacc=wacc,
            growth_high=params["growth_high"],
            growth_terminal=params["growth_terminal"],
            shares_outstanding=shares,
            net_debt=net_debt,
        )
        results[name]["growth_high"] = params["growth_high"]
        results[name]["growth_terminal"] = params["growth_terminal"]

    results["inputs"] = {
        "base_fcf": base_fcf,
        "wacc": wacc,
        "shares_outstanding": shares,
        "net_debt": net_debt,
        "base_growth_anchor": base_growth_anchor,
    }
    return results


def key_ratios(info: dict) -> dict:
    """Extract and compute key financial ratios."""
    return {
        "valuation": {
            "P/E (TTM)": safe_get(info, "trailingPE"),
            "P/E (Forward)": safe_get(info, "forwardPE"),
            "PEG Ratio": safe_get(info, "pegRatio", "trailingPegRatio"),
            "P/S": safe_get(info, "priceToSalesTrailing12Months"),
            "P/B": safe_get(info, "priceToBook"),
            "EV/EBITDA": safe_get(info, "enterpriseToEbitda"),
            "EV/Revenue": safe_get(info, "enterpriseToRevenue"),
        },
        "profitability": {
            "Gross Margin": safe_get(info, "grossMargins"),
            "Operating Margin": safe_get(info, "operatingMargins"),
            "Net Margin": safe_get(info, "profitMargins"),
            "ROE": safe_get(info, "returnOnEquity"),
            "ROA": safe_get(info, "returnOnAssets"),
        },
        "growth": {
            "Revenue Growth (YoY)": safe_get(info, "revenueGrowth"),
            "Earnings Growth (YoY)": safe_get(info, "earningsGrowth"),
            "Earnings Quarterly Growth": safe_get(info, "earningsQuarterlyGrowth"),
        },
        "balance_sheet": {
            "Debt/Equity": safe_get(info, "debtToEquity"),
            "Current Ratio": safe_get(info, "currentRatio"),
            "Quick Ratio": safe_get(info, "quickRatio"),
            "Total Cash": safe_get(info, "totalCash"),
            "Total Debt": safe_get(info, "totalDebt"),
        },
        "dividend": {
            "Dividend Yield": safe_get(info, "dividendYield"),
            "Payout Ratio": safe_get(info, "payoutRatio"),
            "5Y Avg Dividend Yield": safe_get(info, "fiveYearAvgDividendYield"),
        },
    }


def quality_score(info: dict, ratios: dict) -> dict:
    """Composite quality score 0-100 based on profitability, balance sheet, and growth."""
    score = 0
    max_score = 0
    breakdown = {}

    # Profitability (40 points)
    roe = ratios["profitability"]["ROE"]
    if roe is not None:
        max_score += 15
        if roe > 0.20:
            pts = 15
        elif roe > 0.15:
            pts = 12
        elif roe > 0.10:
            pts = 8
        elif roe > 0.05:
            pts = 4
        else:
            pts = 0
        score += pts
        breakdown["ROE"] = f"{pts}/15 (ROE: {roe*100:.1f}%)"

    op_margin = ratios["profitability"]["Operating Margin"]
    if op_margin is not None:
        max_score += 15
        if op_margin > 0.25:
            pts = 15
        elif op_margin > 0.15:
            pts = 12
        elif op_margin > 0.08:
            pts = 8
        elif op_margin > 0:
            pts = 4
        else:
            pts = 0
        score += pts
        breakdown["Operating Margin"] = f"{pts}/15 ({op_margin*100:.1f}%)"

    net_margin = ratios["profitability"]["Net Margin"]
    if net_margin is not None:
        max_score += 10
        if net_margin > 0.20:
            pts = 10
        elif net_margin > 0.10:
            pts = 8
        elif net_margin > 0.05:
            pts = 5
        elif net_margin > 0:
            pts = 2
        else:
            pts = 0
        score += pts
        breakdown["Net Margin"] = f"{pts}/10 ({net_margin*100:.1f}%)"

    # Balance sheet (30 points)
    de = ratios["balance_sheet"]["Debt/Equity"]
    if de is not None:
        max_score += 15
        # yfinance returns D/E as a percentage sometimes (e.g. 150 = 1.5)
        de_norm = de / 100 if de > 5 else de
        if de_norm < 0.3:
            pts = 15
        elif de_norm < 0.6:
            pts = 12
        elif de_norm < 1.0:
            pts = 8
        elif de_norm < 2.0:
            pts = 4
        else:
            pts = 0
        score += pts
        breakdown["Debt/Equity"] = f"{pts}/15 ({de_norm:.2f})"

    cr = ratios["balance_sheet"]["Current Ratio"]
    if cr is not None:
        max_score += 15
        if cr > 2.0:
            pts = 15
        elif cr > 1.5:
            pts = 12
        elif cr > 1.0:
            pts = 8
        elif cr > 0.7:
            pts = 4
        else:
            pts = 0
        score += pts
        breakdown["Current Ratio"] = f"{pts}/15 ({cr:.2f})"

    # Growth (30 points)
    rev_growth = ratios["growth"]["Revenue Growth (YoY)"]
    if rev_growth is not None:
        max_score += 15
        if rev_growth > 0.20:
            pts = 15
        elif rev_growth > 0.10:
            pts = 12
        elif rev_growth > 0.05:
            pts = 8
        elif rev_growth > 0:
            pts = 4
        else:
            pts = 0
        score += pts
        breakdown["Revenue Growth"] = f"{pts}/15 ({rev_growth*100:.1f}%)"

    earn_growth = ratios["growth"]["Earnings Growth (YoY)"]
    if earn_growth is not None:
        max_score += 15
        if earn_growth > 0.20:
            pts = 15
        elif earn_growth > 0.10:
            pts = 12
        elif earn_growth > 0:
            pts = 8
        elif earn_growth > -0.10:
            pts = 4
        else:
            pts = 0
        score += pts
        breakdown["Earnings Growth"] = f"{pts}/15 ({earn_growth*100:.1f}%)"

    final = (score / max_score * 100) if max_score > 0 else 0
    return {
        "score": final,
        "raw_score": score,
        "max_possible": max_score,
        "breakdown": breakdown,
        "grade": grade_from_score(final),
    }


def grade_from_score(score: float) -> str:
    if score >= 85:
        return "A (Exceptional)"
    if score >= 70:
        return "B (Strong)"
    if score >= 55:
        return "C (Average)"
    if score >= 40:
        return "D (Weak)"
    return "F (Poor)"
