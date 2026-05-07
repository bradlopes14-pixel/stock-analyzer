"""
Predictions Module
------------------
Generates short-term (days/weeks) and medium-term (3-12 months) forecasts.
Combines multiple methods rather than relying on any single one.

IMPORTANT: All predictions are probabilistic ranges with explicit assumptions.
Markets are dominated by noise; treat all outputs as scenarios, not certainties.
"""

import pandas as pd
import numpy as np
from .technicals import atr, technical_signal_summary, support_resistance, rsi
from .quant import returns_from_prices, annualized_volatility


def short_term_forecast(df: pd.DataFrame, horizon_days: int = 21) -> dict:
    """
    Short-term forecast (days to weeks).
    Methods combined:
      1. ATR-based volatility envelope around current price
      2. Technical composite score adjustment
      3. Support/resistance asymmetry
      4. Recent momentum and mean-reversion blend
    """
    if len(df) < 50:
        return {"available": False, "reason": "Insufficient data"}

    current_price = float(df["Close"].iloc[-1])
    daily_vol = df["Close"].pct_change().std()
    horizon_vol = daily_vol * np.sqrt(horizon_days)

    # 1-sigma move over horizon
    one_sigma_pct = horizon_vol
    atr_val = atr(df).iloc[-1]
    atr_pct = atr_val / current_price

    # Tech signal adjustment: tilt the central forecast
    tech = technical_signal_summary(df)
    tech_tilt = tech["score"] / 100 * 0.02  # max ±2% tilt for ~21 days

    # Mean reversion factor: if RSI extreme, fade it
    rsi_now = rsi(df["Close"]).iloc[-1]
    if rsi_now > 75:
        mean_rev = -0.01
    elif rsi_now < 25:
        mean_rev = 0.01
    else:
        mean_rev = 0

    expected_return = tech_tilt + mean_rev
    central = current_price * (1 + expected_return)

    # Build scenario bands using normal distribution
    scenarios = {
        "bear_low": current_price * (1 + expected_return - 2 * one_sigma_pct),
        "bear": current_price * (1 + expected_return - one_sigma_pct),
        "base": central,
        "bull": current_price * (1 + expected_return + one_sigma_pct),
        "bull_high": current_price * (1 + expected_return + 2 * one_sigma_pct),
    }

    # Probability assignments based on technical bias
    # Neutral case = 50/30/20 split or thereabouts
    if tech["score"] > 30:
        prob = {"bear": 0.20, "base": 0.45, "bull": 0.35}
    elif tech["score"] < -30:
        prob = {"bear": 0.35, "base": 0.45, "bull": 0.20}
    else:
        prob = {"bear": 0.27, "base": 0.46, "bull": 0.27}

    return {
        "available": True,
        "horizon_days": horizon_days,
        "current_price": current_price,
        "scenarios": scenarios,
        "probabilities": prob,
        "expected_return_pct": expected_return * 100,
        "one_sigma_move_pct": one_sigma_pct * 100,
        "atr_pct": atr_pct * 100,
        "tech_score": tech["score"],
        "tech_interpretation": tech["interpretation"],
        "rsi": rsi_now,
        "method_notes": [
            f"Volatility envelope: ±{one_sigma_pct*100:.1f}% (1σ) over {horizon_days} days",
            f"Technical composite tilt: {tech_tilt*100:+.2f}%",
            f"Mean-reversion adjustment: {mean_rev*100:+.2f}%",
        ],
    }


def medium_term_forecast(
    df: pd.DataFrame,
    dcf_results: dict,
    quality_score: float,
    risk_metrics: dict,
    horizon_months: int = 9,
) -> dict:
    """
    Medium-term forecast (3-12 months).
    Methods combined:
      1. DCF-anchored fair value (bear/base/bull)
      2. Quality score adjustment (high quality = price reverts toward DCF; low quality = discount persists)
      3. Mean reversion to long-term return
      4. Momentum carry (12-1 month)
      5. Volatility-scaled confidence interval
    """
    if len(df) < 100:
        return {"available": False, "reason": "Insufficient data"}

    current_price = float(df["Close"].iloc[-1])

    # Pull DCF values
    bear_fv = dcf_results.get("bear", {}).get("intrinsic_per_share")
    base_fv = dcf_results.get("base", {}).get("intrinsic_per_share")
    bull_fv = dcf_results.get("bull", {}).get("intrinsic_per_share")

    has_dcf = all(v is not None and v > 0 for v in [bear_fv, base_fv, bull_fv])

    # Quality scaling: how much does price actually revert to DCF?
    # Higher quality companies tend to track fair value more closely
    if quality_score >= 70:
        reversion_speed = 0.6  # 60% closure of price-FV gap over horizon
    elif quality_score >= 55:
        reversion_speed = 0.4
    elif quality_score >= 40:
        reversion_speed = 0.25
    else:
        reversion_speed = 0.15

    # Time-scaling: more time = more reversion
    time_factor = horizon_months / 12

    # Long-run market expected return baseline
    market_baseline = 0.08 * (horizon_months / 12)  # 8% annual

    # Beta-adjusted baseline
    beta = risk_metrics.get("beta") or 1.0
    beta_adjusted_baseline = market_baseline * beta

    # Momentum carry (12-1 month return decays)
    if len(df) >= 252:
        twelve_one = (df["Close"].iloc[-21] / df["Close"].iloc[-252]) - 1
        # Empirically momentum has half-life around 3-6 months
        momentum_carry = twelve_one * 0.15 * time_factor
    else:
        momentum_carry = 0

    if has_dcf:
        # Scenario targets blend: weight(DCF reversion) + weight(market baseline) + momentum
        bear_target = current_price + (bear_fv - current_price) * reversion_speed * time_factor
        base_target = current_price + (base_fv - current_price) * reversion_speed * time_factor + momentum_carry * current_price
        bull_target = current_price + (bull_fv - current_price) * reversion_speed * time_factor + momentum_carry * current_price * 1.2

        # Add macro baseline so non-DCF drift is captured
        bear_target += beta_adjusted_baseline * current_price * 0.3
        base_target += beta_adjusted_baseline * current_price * 0.6
        bull_target += beta_adjusted_baseline * current_price

        method = "DCF-anchored with quality-scaled reversion"
    else:
        # Fallback: technical + momentum + baseline
        ann_vol = risk_metrics.get("annualized_volatility", 0.30)
        sigma_h = ann_vol * np.sqrt(horizon_months / 12)
        center = current_price * (1 + beta_adjusted_baseline + momentum_carry)
        bear_target = center * (1 - sigma_h)
        base_target = center
        bull_target = center * (1 + sigma_h)
        method = "Volatility-baseline (DCF unavailable)"

    # Probability assignment based on tech tilt + quality
    tech = technical_signal_summary(df)
    if tech["score"] > 30 and quality_score >= 60:
        prob = {"bear": 0.18, "base": 0.50, "bull": 0.32}
    elif tech["score"] > 30:
        prob = {"bear": 0.22, "base": 0.48, "bull": 0.30}
    elif tech["score"] < -30 and quality_score < 50:
        prob = {"bear": 0.40, "base": 0.45, "bull": 0.15}
    elif tech["score"] < -30:
        prob = {"bear": 0.32, "base": 0.48, "bull": 0.20}
    else:
        prob = {"bear": 0.25, "base": 0.50, "bull": 0.25}

    expected_value = (
        prob["bear"] * bear_target
        + prob["base"] * base_target
        + prob["bull"] * bull_target
    )
    expected_return_pct = (expected_value / current_price - 1) * 100

    # Recommendation logic
    if expected_return_pct > 25:
        rec = "STRONG BUY"
    elif expected_return_pct > 12:
        rec = "BUY"
    elif expected_return_pct > -5:
        rec = "HOLD"
    elif expected_return_pct > -15:
        rec = "REDUCE"
    else:
        rec = "SELL"

    return {
        "available": True,
        "horizon_months": horizon_months,
        "current_price": current_price,
        "scenarios": {
            "bear": bear_target,
            "base": base_target,
            "bull": bull_target,
        },
        "probabilities": prob,
        "expected_value": expected_value,
        "expected_return_pct": expected_return_pct,
        "recommendation": rec,
        "method": method,
        "inputs": {
            "quality_score": quality_score,
            "reversion_speed": reversion_speed,
            "beta": beta,
            "momentum_carry": momentum_carry,
            "dcf_available": has_dcf,
        },
    }


def monte_carlo_paths(
    df: pd.DataFrame,
    horizon_days: int = 252,
    n_simulations: int = 5000,
    use_log: bool = True,
) -> dict:
    """Monte Carlo simulation using historical return distribution (geometric Brownian motion)."""
    if len(df) < 60:
        return {"available": False}

    rets = np.log(df["Close"] / df["Close"].shift(1)).dropna() if use_log else df["Close"].pct_change().dropna()
    mu = rets.mean()
    sigma = rets.std()
    current = float(df["Close"].iloc[-1])

    # Vectorized simulation
    rng = np.random.default_rng(seed=42)
    shocks = rng.normal(mu, sigma, size=(n_simulations, horizon_days))

    if use_log:
        log_paths = np.cumsum(shocks, axis=1)
        terminals = current * np.exp(log_paths[:, -1])
    else:
        cumulative = np.cumprod(1 + shocks, axis=1)
        terminals = current * cumulative[:, -1]

    return {
        "available": True,
        "horizon_days": horizon_days,
        "n_simulations": n_simulations,
        "current": current,
        "terminals": terminals,
        "p10": np.percentile(terminals, 10),
        "p25": np.percentile(terminals, 25),
        "p50": np.percentile(terminals, 50),
        "p75": np.percentile(terminals, 75),
        "p90": np.percentile(terminals, 90),
        "mean": terminals.mean(),
        "prob_above_current": (terminals > current).mean(),
    }
