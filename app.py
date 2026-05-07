"""
Technical Analysis
------------------
Indicators, pattern detection, signal generation.
Pure Python/NumPy implementations to avoid TA-Lib dependency.
"""

import pandas as pd
import numpy as np


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return {"macd": macd_line, "signal": signal_line, "histogram": histogram}


def bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0) -> dict:
    middle = sma(series, period)
    std = series.rolling(window=period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    bandwidth = (upper - lower) / middle
    pct_b = (series - lower) / (upper - lower)
    return {"upper": upper, "middle": middle, "lower": lower, "bandwidth": bandwidth, "pct_b": pct_b}


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> dict:
    low_min = df["Low"].rolling(window=k_period).min()
    high_max = df["High"].rolling(window=k_period).max()
    k = 100 * (df["Close"] - low_min) / (high_max - low_min)
    d = k.rolling(window=d_period).mean()
    return {"k": k, "d": d}


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index — measures trend strength."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    plus_dm[(plus_dm - minus_dm) <= 0] = 0
    minus_dm[(minus_dm - plus_dm) <= 0] = 0

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_v = tr.ewm(alpha=1 / period, adjust=False).mean()

    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_v
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_v
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(df["Close"].diff()).fillna(0)
    return (direction * df["Volume"]).cumsum()


def vwap(df: pd.DataFrame) -> pd.Series:
    """Volume-Weighted Average Price (rolling)."""
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    return (typical * df["Volume"]).cumsum() / df["Volume"].cumsum()


def fibonacci_levels(df: pd.DataFrame, lookback: int = 60) -> dict:
    """Fibonacci retracement from recent swing high/low."""
    recent = df.tail(lookback)
    high = recent["High"].max()
    low = recent["Low"].min()
    diff = high - low
    return {
        "high": high,
        "low": low,
        "23.6%": high - 0.236 * diff,
        "38.2%": high - 0.382 * diff,
        "50.0%": high - 0.500 * diff,
        "61.8%": high - 0.618 * diff,
        "78.6%": high - 0.786 * diff,
    }


def support_resistance(df: pd.DataFrame, lookback: int = 90, num_levels: int = 3) -> dict:
    """Identify support and resistance via local extrema clustering."""
    recent = df.tail(lookback)
    highs = recent["High"].nlargest(num_levels * 2).round(2).unique()[:num_levels]
    lows = recent["Low"].nsmallest(num_levels * 2).round(2).unique()[:num_levels]
    current = df["Close"].iloc[-1]
    return {
        "resistance": sorted([h for h in highs if h > current])[:num_levels],
        "support": sorted([l for l in lows if l < current], reverse=True)[:num_levels],
        "current": current,
    }


def detect_patterns(df: pd.DataFrame) -> list:
    """Detect basic chart patterns. Returns list of detected patterns."""
    if len(df) < 50:
        return []

    patterns = []
    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values
    recent = df.tail(50)

    # Golden cross / Death cross (50 vs 200 SMA)
    if len(df) >= 200:
        sma50 = sma(df["Close"], 50)
        sma200 = sma(df["Close"], 200)
        if sma50.iloc[-1] > sma200.iloc[-1] and sma50.iloc[-5] <= sma200.iloc[-5]:
            patterns.append({"name": "Golden Cross (50/200 SMA)", "signal": "bullish", "strength": "strong"})
        elif sma50.iloc[-1] < sma200.iloc[-1] and sma50.iloc[-5] >= sma200.iloc[-5]:
            patterns.append({"name": "Death Cross (50/200 SMA)", "signal": "bearish", "strength": "strong"})

    # Higher highs + higher lows (uptrend) / lower lows + lower highs (downtrend)
    if len(recent) >= 20:
        first_half_high = recent["High"].iloc[:25].max()
        second_half_high = recent["High"].iloc[25:].max()
        first_half_low = recent["Low"].iloc[:25].min()
        second_half_low = recent["Low"].iloc[25:].min()

        if second_half_high > first_half_high and second_half_low > first_half_low:
            patterns.append({"name": "Uptrend (HH + HL)", "signal": "bullish", "strength": "moderate"})
        elif second_half_high < first_half_high and second_half_low < first_half_low:
            patterns.append({"name": "Downtrend (LH + LL)", "signal": "bearish", "strength": "moderate"})

    # Bollinger squeeze
    bb = bollinger_bands(df["Close"])
    bw = bb["bandwidth"]
    if not bw.empty and not pd.isna(bw.iloc[-1]):
        recent_bw = bw.tail(60).dropna()
        if len(recent_bw) > 20 and bw.iloc[-1] < recent_bw.quantile(0.2):
            patterns.append({"name": "Bollinger Squeeze (volatility compression)", "signal": "neutral", "strength": "moderate", "note": "expansion likely"})

    # Double bottom (rough heuristic)
    if len(recent) >= 30:
        recent_lows = recent["Low"].values
        min_idx_1 = np.argmin(recent_lows[:15])
        min_idx_2 = 15 + np.argmin(recent_lows[15:])
        v1 = recent_lows[min_idx_1]
        v2 = recent_lows[min_idx_2]
        if abs(v1 - v2) / v1 < 0.03 and recent["Close"].iloc[-1] > max(recent_lows[min_idx_1:min_idx_2 + 1]):
            patterns.append({"name": "Possible Double Bottom", "signal": "bullish", "strength": "moderate"})

    # Double top (rough heuristic)
    if len(recent) >= 30:
        recent_highs = recent["High"].values
        max_idx_1 = np.argmax(recent_highs[:15])
        max_idx_2 = 15 + np.argmax(recent_highs[15:])
        h1 = recent_highs[max_idx_1]
        h2 = recent_highs[max_idx_2]
        if abs(h1 - h2) / h1 < 0.03 and recent["Close"].iloc[-1] < min(recent_highs[max_idx_1:max_idx_2 + 1]):
            patterns.append({"name": "Possible Double Top", "signal": "bearish", "strength": "moderate"})

    # RSI divergence (price up, RSI down — or vice versa) over last 20 bars
    rsi_series = rsi(df["Close"]).tail(20)
    price_series = df["Close"].tail(20)
    if not rsi_series.empty and len(rsi_series) >= 20:
        price_change = price_series.iloc[-1] - price_series.iloc[0]
        rsi_change = rsi_series.iloc[-1] - rsi_series.iloc[0]
        if price_change > 0 and rsi_change < -5:
            patterns.append({"name": "Bearish RSI Divergence", "signal": "bearish", "strength": "moderate"})
        elif price_change < 0 and rsi_change > 5:
            patterns.append({"name": "Bullish RSI Divergence", "signal": "bullish", "strength": "moderate"})

    return patterns


def technical_signal_summary(df: pd.DataFrame) -> dict:
    """Aggregate technical indicators into a composite directional score (-100 to +100)."""
    if len(df) < 50:
        return {"score": 0, "signals": [], "interpretation": "Insufficient data"}

    signals = []
    score = 0
    weight_total = 0

    close = df["Close"]
    current = close.iloc[-1]

    # Trend: SMA 20/50/200
    if len(df) >= 200:
        sma20 = sma(close, 20).iloc[-1]
        sma50 = sma(close, 50).iloc[-1]
        sma200 = sma(close, 200).iloc[-1]
        if current > sma20 > sma50 > sma200:
            signals.append(("Price above all major SMAs (20/50/200)", "bullish", 20))
            score += 20
        elif current < sma20 < sma50 < sma200:
            signals.append(("Price below all major SMAs", "bearish", -20))
            score -= 20
        elif current > sma50:
            signals.append(("Price above 50 SMA", "bullish", 8))
            score += 8
        else:
            signals.append(("Price below 50 SMA", "bearish", -8))
            score -= 8
        weight_total += 20
    elif len(df) >= 50:
        sma50 = sma(close, 50).iloc[-1]
        if current > sma50:
            signals.append(("Price above 50 SMA", "bullish", 10))
            score += 10
        else:
            signals.append(("Price below 50 SMA", "bearish", -10))
            score -= 10
        weight_total += 10

    # Momentum: RSI
    rsi_val = rsi(close).iloc[-1]
    if not pd.isna(rsi_val):
        if rsi_val > 70:
            signals.append((f"RSI overbought ({rsi_val:.1f})", "bearish", -10))
            score -= 10
        elif rsi_val < 30:
            signals.append((f"RSI oversold ({rsi_val:.1f})", "bullish", 10))
            score += 10
        elif rsi_val > 55:
            signals.append((f"RSI bullish ({rsi_val:.1f})", "bullish", 5))
            score += 5
        elif rsi_val < 45:
            signals.append((f"RSI bearish ({rsi_val:.1f})", "bearish", -5))
            score -= 5
        else:
            signals.append((f"RSI neutral ({rsi_val:.1f})", "neutral", 0))
        weight_total += 10

    # MACD
    macd_data = macd(close)
    macd_line = macd_data["macd"].iloc[-1]
    signal_line = macd_data["signal"].iloc[-1]
    hist = macd_data["histogram"].iloc[-1]
    hist_prev = macd_data["histogram"].iloc[-2] if len(macd_data["histogram"]) > 1 else hist
    if not pd.isna(macd_line) and not pd.isna(signal_line):
        if macd_line > signal_line and hist > hist_prev:
            signals.append(("MACD bullish (line > signal, expanding)", "bullish", 12))
            score += 12
        elif macd_line > signal_line:
            signals.append(("MACD bullish (line > signal)", "bullish", 6))
            score += 6
        elif macd_line < signal_line and hist < hist_prev:
            signals.append(("MACD bearish (line < signal, expanding)", "bearish", -12))
            score -= 12
        else:
            signals.append(("MACD bearish (line < signal)", "bearish", -6))
            score -= 6
        weight_total += 12

    # Bollinger position
    bb = bollinger_bands(close)
    pct_b = bb["pct_b"].iloc[-1]
    if not pd.isna(pct_b):
        if pct_b > 1.0:
            signals.append((f"Above upper Bollinger band (%B={pct_b:.2f})", "bearish", -8))
            score -= 8
        elif pct_b < 0.0:
            signals.append((f"Below lower Bollinger band (%B={pct_b:.2f})", "bullish", 8))
            score += 8
        elif pct_b > 0.8:
            signals.append((f"Near upper band (%B={pct_b:.2f})", "neutral", 0))
        elif pct_b < 0.2:
            signals.append((f"Near lower band (%B={pct_b:.2f})", "neutral", 0))
        weight_total += 8

    # ADX (trend strength)
    adx_val = adx(df).iloc[-1]
    if not pd.isna(adx_val):
        if adx_val > 25:
            signals.append((f"Strong trend (ADX={adx_val:.1f})", "trend confirmed", 0))
        else:
            signals.append((f"Weak / no trend (ADX={adx_val:.1f})", "ranging", 0))

    # Volume confirmation
    vol_ma = df["Volume"].rolling(20).mean().iloc[-1]
    last_vol = df["Volume"].iloc[-1]
    if vol_ma > 0:
        vol_ratio = last_vol / vol_ma
        price_change = (close.iloc[-1] / close.iloc[-2] - 1) if len(close) > 1 else 0
        if vol_ratio > 1.5 and price_change > 0:
            signals.append((f"High volume rally (vol {vol_ratio:.1f}x avg)", "bullish", 8))
            score += 8
        elif vol_ratio > 1.5 and price_change < 0:
            signals.append((f"High volume sell-off (vol {vol_ratio:.1f}x avg)", "bearish", -8))
            score -= 8
        weight_total += 8

    # Normalize to -100 to +100
    if weight_total > 0:
        normalized = (score / weight_total) * 100
    else:
        normalized = 0

    if normalized > 50:
        interp = "Strongly Bullish"
    elif normalized > 20:
        interp = "Bullish"
    elif normalized > -20:
        interp = "Neutral"
    elif normalized > -50:
        interp = "Bearish"
    else:
        interp = "Strongly Bearish"

    return {
        "score": normalized,
        "raw_score": score,
        "max_weight": weight_total,
        "signals": signals,
        "interpretation": interp,
    }
