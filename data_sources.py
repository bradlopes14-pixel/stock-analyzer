"""
Institutional-Grade Stock Analyzer
==================================
Streamlit application for comprehensive equity analysis.

Run with:
    streamlit run app.py

Sources: yfinance (primary), Stooq (price cross-check), SEC EDGAR (filings).
All predictions are probabilistic ranges, not certainties.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

from data_sources import (
    fetch_yfinance_core,
    fetch_stooq_price,
    fetch_sec_filings,
    fetch_market_benchmark,
    fetch_risk_free_rate,
    cross_check_price,
)
from analysis.fundamentals import (
    calculate_wacc,
    dcf_scenarios,
    key_ratios,
    quality_score,
    safe_get,
)
from analysis.technicals import (
    sma, ema, rsi, macd, bollinger_bands, atr, stochastic, adx, obv, vwap,
    fibonacci_levels, support_resistance, detect_patterns, technical_signal_summary,
)
from analysis.quant import (
    full_risk_metrics,
    factor_exposure,
    returns_from_prices,
)
from analysis.predictions import (
    short_term_forecast,
    medium_term_forecast,
    monte_carlo_paths,
)

# ---------------- Page config ----------------
st.set_page_config(
    page_title="Stock Analyzer Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
    .main-header {font-size: 2.2rem; font-weight: 700; color: #1a1a2e;}
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem; border-radius: 10px; margin-bottom: 0.5rem;
    }
    .bullish {color: #16a34a; font-weight: 600;}
    .bearish {color: #dc2626; font-weight: 600;}
    .neutral {color: #6b7280; font-weight: 600;}
    .source-tag {
        background: #e5e7eb; padding: 2px 8px; border-radius: 4px;
        font-size: 0.75rem; color: #374151;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- Helpers ----------------
def fmt_money(v, decimals=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if abs(v) >= 1e12:
        return f"${v/1e12:.{decimals}f}T"
    if abs(v) >= 1e9:
        return f"${v/1e9:.{decimals}f}B"
    if abs(v) >= 1e6:
        return f"${v/1e6:.{decimals}f}M"
    return f"${v:,.{decimals}f}"

def fmt_pct(v, decimals=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v*100:.{decimals}f}%"

def fmt_num(v, decimals=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{decimals}f}"

def signal_color(score):
    if score > 30: return "#16a34a"
    if score > 0: return "#65a30d"
    if score > -30: return "#f59e0b"
    return "#dc2626"

# ---------------- Sidebar ----------------
st.sidebar.markdown("# 📊 Stock Analyzer Pro")
st.sidebar.markdown("Institutional-grade equity analysis")
st.sidebar.markdown("---")

ticker_input = st.sidebar.text_input("Ticker Symbol", value="AAPL", help="e.g., AAPL, MSFT, NVDA").upper().strip()
analyze_btn = st.sidebar.button("🔍 Analyze", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### Settings")
horizon_short = st.sidebar.slider("Short-term horizon (days)", 5, 60, 21)
horizon_medium = st.sidebar.slider("Medium-term horizon (months)", 3, 18, 9)
mc_simulations = st.sidebar.select_slider("Monte Carlo simulations", [1000, 2500, 5000, 10000], value=5000)

st.sidebar.markdown("---")
st.sidebar.markdown("### Data Sources")
st.sidebar.markdown("""
- **Primary:** Yahoo Finance (yfinance)
- **Cross-check:** Stooq
- **Filings:** SEC EDGAR
- **Risk-free rate:** ^TNX (10Y Treasury)
""")
st.sidebar.markdown("---")
st.sidebar.caption("⚠️ Analytical tool only. Not investment advice. All forecasts are probabilistic.")

# ---------------- Main ----------------
st.markdown('<div class="main-header">📊 Stock Analyzer Pro</div>', unsafe_allow_html=True)
st.caption("Institutional-grade equity analysis with cross-referenced data and probabilistic forecasts")

if not analyze_btn and "last_ticker" not in st.session_state:
    st.info("👈 Enter a ticker symbol and click **Analyze** to begin.")
    st.markdown("""
    ### What this tool does
    **Comprehensive analysis across four dimensions:**

    1. **Fundamentals** — DCF valuation (bear/base/bull), key ratios, quality score
    2. **Technical** — 15+ indicators, pattern detection, signal aggregation
    3. **Quant Risk** — Sharpe, Sortino, VaR, CVaR, max drawdown, factor exposure
    4. **Predictions** — Short-term (technical-heavy) and medium-term (DCF-anchored) side-by-side, plus Monte Carlo

    **Honest about limitations:** No tool reliably predicts markets. This builds professional-grade scenario analysis with explicit assumptions you can stress-test.
    """)
    st.stop()

if analyze_btn:
    st.session_state["last_ticker"] = ticker_input

ticker = st.session_state.get("last_ticker", ticker_input)

# ---------------- Fetch data ----------------
with st.spinner(f"Fetching data for {ticker} from multiple sources..."):
    yf_data = fetch_yfinance_core(ticker)
    stooq_df = fetch_stooq_price(ticker)
    market_df = fetch_market_benchmark()
    risk_free = fetch_risk_free_rate()
    sec_data = fetch_sec_filings(ticker)

if not yf_data["success"] or not yf_data["info"] or yf_data["hist_1y"].empty:
    st.error(f"❌ Could not fetch data for **{ticker}**. Verify the ticker symbol.")
    if yf_data.get("error"):
        st.code(yf_data["error"])
    st.stop()

info = yf_data["info"]
hist_1y = yf_data["hist_1y"]
hist_5y = yf_data["hist_5y"]

# Header
col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
with col1:
    st.markdown(f"### {info.get('longName', ticker)} ({ticker})")
    st.caption(f"{info.get('sector', 'N/A')} · {info.get('industry', 'N/A')} · {info.get('exchange', 'N/A')}")
with col2:
    current_price = info.get("currentPrice") or hist_1y["Close"].iloc[-1]
    prev_close = info.get("previousClose") or hist_1y["Close"].iloc[-2]
    change = current_price - prev_close
    change_pct = change / prev_close * 100
    st.metric("Price", f"${current_price:.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
with col3:
    st.metric("Market Cap", fmt_money(info.get("marketCap")))
with col4:
    st.metric("52W Range", f"${info.get('fiftyTwoWeekLow', 0):.0f}–${info.get('fiftyTwoWeekHigh', 0):.0f}")

# Cross-check
xcheck = cross_check_price(current_price, stooq_df)
if xcheck["checked"]:
    if xcheck["match"]:
        st.success(f"✅ Price cross-check: Yahoo ${xcheck['yf_price']:.2f} vs Stooq ${xcheck['stooq_price']:.2f} (diff {xcheck['diff_pct']:.2f}%)")
    else:
        st.warning(f"⚠️ Price discrepancy: Yahoo ${xcheck['yf_price']:.2f} vs Stooq ${xcheck['stooq_price']:.2f} (diff {xcheck['diff_pct']:.2f}%)")
else:
    st.info("ℹ️ Stooq cross-check unavailable")

# ---------------- Run analyses ----------------
ratios = key_ratios(info)
qs = quality_score(info, ratios)
wacc_data = calculate_wacc(info, risk_free)
dcf = dcf_scenarios(info, yf_data["cashflow"], wacc_data["wacc"])

market_prices = market_df["Close"] if not market_df.empty else pd.Series()
risk = full_risk_metrics(hist_5y["Close"] if not hist_5y.empty else hist_1y["Close"], market_prices, risk_free)
factors = factor_exposure(hist_5y["Close"] if not hist_5y.empty else hist_1y["Close"], market_prices, info)

tech_summary = technical_signal_summary(hist_1y)
patterns = detect_patterns(hist_1y)
sr_levels = support_resistance(hist_1y)
fib = fibonacci_levels(hist_1y)

short_pred = short_term_forecast(hist_1y, horizon_short)
medium_pred = medium_term_forecast(hist_1y, dcf, qs["score"], risk, horizon_medium)
mc = monte_carlo_paths(hist_1y, horizon_days=horizon_medium * 21, n_simulations=mc_simulations)

# ---------------- Tabs ----------------
tab_overview, tab_predict, tab_fund, tab_tech, tab_quant, tab_filings = st.tabs([
    "📋 Overview", "🔮 Predictions", "💼 Fundamentals", "📈 Technical", "🎯 Quant & Risk", "📑 Filings"
])

# ============ OVERVIEW TAB ============
with tab_overview:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 🎯 Quality Score")
        st.metric("Composite", f"{qs['score']:.1f}/100", qs["grade"])
    with c2:
        st.markdown("#### 📊 Technical Signal")
        st.metric("Score", f"{tech_summary['score']:+.1f}", tech_summary["interpretation"])
    with c3:
        st.markdown("#### 🔮 Medium-Term Recommendation")
        if medium_pred["available"]:
            st.metric(
                f"{horizon_medium}-month outlook",
                medium_pred["recommendation"],
                f"{medium_pred['expected_return_pct']:+.1f}% expected",
            )
        else:
            st.info("Insufficient data")

    st.markdown("---")
    st.markdown("### 📈 Price Chart with Key Indicators")

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2],
        vertical_spacing=0.03,
        subplot_titles=("Price + SMAs + Bollinger Bands", "Volume", "RSI"),
    )

    # Candles
    fig.add_trace(go.Candlestick(
        x=hist_1y.index, open=hist_1y["Open"], high=hist_1y["High"],
        low=hist_1y["Low"], close=hist_1y["Close"], name="Price",
    ), row=1, col=1)

    sma20 = sma(hist_1y["Close"], 20)
    sma50 = sma(hist_1y["Close"], 50)
    sma200 = sma(hist_1y["Close"], 200) if len(hist_1y) >= 200 else None
    bb = bollinger_bands(hist_1y["Close"])

    fig.add_trace(go.Scatter(x=hist_1y.index, y=sma20, name="SMA 20", line=dict(color="orange", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist_1y.index, y=sma50, name="SMA 50", line=dict(color="blue", width=1)), row=1, col=1)
    if sma200 is not None:
        fig.add_trace(go.Scatter(x=hist_1y.index, y=sma200, name="SMA 200", line=dict(color="red", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist_1y.index, y=bb["upper"], name="BB Upper", line=dict(color="gray", width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist_1y.index, y=bb["lower"], name="BB Lower", line=dict(color="gray", width=1, dash="dot"), fill="tonexty", fillcolor="rgba(128,128,128,0.05)"), row=1, col=1)

    # Volume
    colors = ["green" if c >= o else "red" for o, c in zip(hist_1y["Open"], hist_1y["Close"])]
    fig.add_trace(go.Bar(x=hist_1y.index, y=hist_1y["Volume"], name="Volume", marker_color=colors), row=2, col=1)

    # RSI
    rsi_series = rsi(hist_1y["Close"])
    fig.add_trace(go.Scatter(x=hist_1y.index, y=rsi_series, name="RSI", line=dict(color="purple")), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    fig.update_layout(height=750, showlegend=True, xaxis_rangeslider_visible=False)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])

    st.plotly_chart(fig, use_container_width=True)

    # Snapshot row
    st.markdown("### 📋 Snapshot")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown("**Valuation**")
        st.write(f"P/E: {fmt_num(ratios['valuation']['P/E (TTM)'])}")
        st.write(f"Fwd P/E: {fmt_num(ratios['valuation']['P/E (Forward)'])}")
        st.write(f"P/S: {fmt_num(ratios['valuation']['P/S'])}")
        st.write(f"EV/EBITDA: {fmt_num(ratios['valuation']['EV/EBITDA'])}")
    with s2:
        st.markdown("**Profitability**")
        st.write(f"Gross Margin: {fmt_pct(ratios['profitability']['Gross Margin'])}")
        st.write(f"Op Margin: {fmt_pct(ratios['profitability']['Operating Margin'])}")
        st.write(f"Net Margin: {fmt_pct(ratios['profitability']['Net Margin'])}")
        st.write(f"ROE: {fmt_pct(ratios['profitability']['ROE'])}")
    with s3:
        st.markdown("**Risk**")
        st.write(f"Beta: {fmt_num(risk.get('beta'))}")
        st.write(f"Ann. Vol: {fmt_pct(risk.get('annualized_volatility'))}")
        st.write(f"Max DD: {fmt_pct(risk.get('max_drawdown'))}")
        st.write(f"Sharpe: {fmt_num(risk.get('sharpe_ratio'))}")
    with s4:
        st.markdown("**Factor Profile**")
        st.write(f"Size: {factors['size']}")
        st.write(f"Value: {factors['value']}")
        st.write(f"Quality: {factors['quality']}")
        if factors.get("momentum_12_1") is not None:
            st.write(f"Momentum 12-1: {fmt_pct(factors['momentum_12_1'])}")

# ============ PREDICTIONS TAB ============
with tab_predict:
    st.markdown("## 🔮 Forecasts — Side by Side")
    st.caption("All forecasts are probabilistic scenarios. Markets are dominated by noise; treat outputs as decision-support, not certainty.")

    col_short, col_medium = st.columns(2)

    with col_short:
        st.markdown(f"### Short-Term ({horizon_short} days)")
        st.caption("Technical-heavy: ATR volatility envelope + signal score + mean reversion")
        if short_pred["available"]:
            st.metric(
                "Expected Return",
                f"{short_pred['expected_return_pct']:+.2f}%",
                f"1σ band ±{short_pred['one_sigma_move_pct']:.1f}%"
            )

            sc = short_pred["scenarios"]
            scenario_df = pd.DataFrame({
                "Scenario": ["Bear (-2σ)", "Bear (-1σ)", "Base", "Bull (+1σ)", "Bull (+2σ)"],
                "Price Target": [sc["bear_low"], sc["bear"], sc["base"], sc["bull"], sc["bull_high"]],
                "% from Now": [
                    (sc["bear_low"]/short_pred["current_price"] - 1) * 100,
                    (sc["bear"]/short_pred["current_price"] - 1) * 100,
                    (sc["base"]/short_pred["current_price"] - 1) * 100,
                    (sc["bull"]/short_pred["current_price"] - 1) * 100,
                    (sc["bull_high"]/short_pred["current_price"] - 1) * 100,
                ],
            })
            scenario_df["Price Target"] = scenario_df["Price Target"].apply(lambda x: f"${x:.2f}")
            scenario_df["% from Now"] = scenario_df["% from Now"].apply(lambda x: f"{x:+.2f}%")
            st.dataframe(scenario_df, use_container_width=True, hide_index=True)

            st.markdown("**Probability Distribution**")
            prob = short_pred["probabilities"]
            pcol1, pcol2, pcol3 = st.columns(3)
            pcol1.metric("Bearish", f"{prob['bear']*100:.0f}%")
            pcol2.metric("Base case", f"{prob['base']*100:.0f}%")
            pcol3.metric("Bullish", f"{prob['bull']*100:.0f}%")

            with st.expander("Method details"):
                for note in short_pred["method_notes"]:
                    st.write(f"- {note}")
                st.write(f"- Technical composite score: **{short_pred['tech_score']:+.1f}** ({short_pred['tech_interpretation']})")
                st.write(f"- RSI: **{short_pred['rsi']:.1f}**")
        else:
            st.warning(short_pred.get("reason", "Unavailable"))

    with col_medium:
        st.markdown(f"### Medium-Term ({horizon_medium} months)")
        st.caption("DCF-anchored: fair-value reversion scaled by quality + momentum carry")
        if medium_pred["available"]:
            st.metric(
                "Expected Return",
                f"{medium_pred['expected_return_pct']:+.2f}%",
                medium_pred["recommendation"]
            )

            mc_scenarios = medium_pred["scenarios"]
            mc_df = pd.DataFrame({
                "Scenario": ["Bear", "Base", "Bull"],
                "Price Target": [mc_scenarios["bear"], mc_scenarios["base"], mc_scenarios["bull"]],
                "% from Now": [
                    (mc_scenarios["bear"]/medium_pred["current_price"] - 1) * 100,
                    (mc_scenarios["base"]/medium_pred["current_price"] - 1) * 100,
                    (mc_scenarios["bull"]/medium_pred["current_price"] - 1) * 100,
                ],
                "Probability": [
                    f"{medium_pred['probabilities']['bear']*100:.0f}%",
                    f"{medium_pred['probabilities']['base']*100:.0f}%",
                    f"{medium_pred['probabilities']['bull']*100:.0f}%",
                ],
            })
            mc_df["Price Target"] = mc_df["Price Target"].apply(lambda x: f"${x:.2f}")
            mc_df["% from Now"] = mc_df["% from Now"].apply(lambda x: f"{x:+.2f}%")
            st.dataframe(mc_df, use_container_width=True, hide_index=True)

            st.metric("Probability-Weighted Target", f"${medium_pred['expected_value']:.2f}")

            with st.expander("Method details"):
                inp = medium_pred["inputs"]
                st.write(f"- Method: **{medium_pred['method']}**")
                st.write(f"- Quality score: **{inp['quality_score']:.1f}/100**")
                st.write(f"- Mean-reversion speed (FV gap closure over horizon): **{inp['reversion_speed']*100:.0f}%**")
                st.write(f"- Beta-adjusted baseline: **{inp['beta']:.2f}** beta")
                st.write(f"- Momentum carry: **{inp['momentum_carry']*100:+.2f}%**")
                st.write(f"- DCF inputs available: **{inp['dcf_available']}**")
        else:
            st.warning(medium_pred.get("reason", "Unavailable"))

    # Monte Carlo
    st.markdown("---")
    st.markdown(f"### 🎲 Monte Carlo Simulation ({mc_simulations:,} paths over {horizon_medium*21} trading days)")
    st.caption("Geometric Brownian Motion using historical return distribution. Useful for sanity-checking the scenarios above.")

    if mc["available"]:
        mc_col1, mc_col2 = st.columns([2, 1])
        with mc_col1:
            # Histogram of terminal prices
            fig_mc = go.Figure()
            fig_mc.add_trace(go.Histogram(
                x=mc["terminals"], nbinsx=80, name="Simulated terminal prices",
                marker_color="lightblue", opacity=0.75,
            ))
            fig_mc.add_vline(x=mc["current"], line_dash="dash", line_color="black", annotation_text="Current")
            fig_mc.add_vline(x=mc["p50"], line_dash="dash", line_color="green", annotation_text="Median")
            fig_mc.add_vline(x=mc["p10"], line_dash="dot", line_color="red", annotation_text="P10")
            fig_mc.add_vline(x=mc["p90"], line_dash="dot", line_color="green", annotation_text="P90")
            fig_mc.update_layout(
                title=f"Distribution of Possible Prices in {horizon_medium} Months",
                xaxis_title="Terminal Price ($)", yaxis_title="Frequency",
                height=400, showlegend=False,
            )
            st.plotly_chart(fig_mc, use_container_width=True)
        with mc_col2:
            st.markdown("**Percentile Outcomes**")
            mc_pct_df = pd.DataFrame({
                "Percentile": ["P10 (worst 10%)", "P25", "P50 (median)", "P75", "P90 (best 10%)"],
                "Price": [f"${mc['p10']:.2f}", f"${mc['p25']:.2f}", f"${mc['p50']:.2f}", f"${mc['p75']:.2f}", f"${mc['p90']:.2f}"],
                "Return": [
                    f"{(mc['p10']/mc['current']-1)*100:+.1f}%",
                    f"{(mc['p25']/mc['current']-1)*100:+.1f}%",
                    f"{(mc['p50']/mc['current']-1)*100:+.1f}%",
                    f"{(mc['p75']/mc['current']-1)*100:+.1f}%",
                    f"{(mc['p90']/mc['current']-1)*100:+.1f}%",
                ],
            })
            st.dataframe(mc_pct_df, use_container_width=True, hide_index=True)
            st.metric("P(price > current)", f"{mc['prob_above_current']*100:.1f}%")
    else:
        st.info("Monte Carlo unavailable (insufficient history)")

# ============ FUNDAMENTALS TAB ============
with tab_fund:
    st.markdown("## 💼 Fundamental Analysis")

    # Quality
    qcol1, qcol2 = st.columns([1, 2])
    with qcol1:
        st.metric("Quality Score", f"{qs['score']:.1f}/100", qs["grade"])
    with qcol2:
        st.markdown("**Score Breakdown:**")
        for k, v in qs["breakdown"].items():
            st.write(f"- **{k}:** {v}")

    st.markdown("---")
    st.markdown("### 📊 Key Ratios")
    rcol1, rcol2 = st.columns(2)
    with rcol1:
        st.markdown("**Valuation**")
        st.dataframe(pd.DataFrame(list(ratios["valuation"].items()), columns=["Metric", "Value"]).assign(
            Value=lambda d: d["Value"].apply(lambda x: fmt_num(x) if x is not None else "—")
        ), use_container_width=True, hide_index=True)
        st.markdown("**Profitability**")
        st.dataframe(pd.DataFrame(list(ratios["profitability"].items()), columns=["Metric", "Value"]).assign(
            Value=lambda d: d["Value"].apply(lambda x: fmt_pct(x) if x is not None else "—")
        ), use_container_width=True, hide_index=True)
    with rcol2:
        st.markdown("**Growth**")
        st.dataframe(pd.DataFrame(list(ratios["growth"].items()), columns=["Metric", "Value"]).assign(
            Value=lambda d: d["Value"].apply(lambda x: fmt_pct(x) if x is not None else "—")
        ), use_container_width=True, hide_index=True)
        st.markdown("**Balance Sheet**")
        bs_display = []
        for k, v in ratios["balance_sheet"].items():
            if "Cash" in k or "Debt" in k:
                bs_display.append([k, fmt_money(v) if v else "—"])
            else:
                bs_display.append([k, fmt_num(v) if v else "—"])
        st.dataframe(pd.DataFrame(bs_display, columns=["Metric", "Value"]), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 💰 DCF Valuation (Bear / Base / Bull)")
    st.caption(f"WACC: {wacc_data['wacc']*100:.2f}% (Cost of Equity: {wacc_data['cost_of_equity']*100:.2f}%, Beta: {wacc_data['beta']:.2f})")

    dcf_inputs = dcf["inputs"]
    st.write(f"**Inputs:** Base FCF: {fmt_money(dcf_inputs['base_fcf'])} · Shares: {fmt_money(dcf_inputs['shares_outstanding'], 0).replace('$','')} · Net Debt: {fmt_money(dcf_inputs['net_debt'])}")

    dcf_rows = []
    for scenario in ["bear", "base", "bull"]:
        s = dcf[scenario]
        if s["valid"]:
            dcf_rows.append({
                "Scenario": scenario.title(),
                "Growth (5Y)": f"{s['growth_high']*100:.1f}%",
                "Terminal Growth": f"{s['growth_terminal']*100:.1f}%",
                "Intrinsic Value/Share": f"${s['intrinsic_per_share']:.2f}",
                "Margin of Safety": f"{(s['intrinsic_per_share']/current_price - 1)*100:+.1f}%",
                "Terminal % of EV": f"{s['terminal_pct']*100:.0f}%",
            })
    if dcf_rows:
        st.dataframe(pd.DataFrame(dcf_rows), use_container_width=True, hide_index=True)
    else:
        st.warning("DCF could not be computed (missing FCF or shares data)")

# ============ TECHNICAL TAB ============
with tab_tech:
    st.markdown("## 📈 Technical Analysis")

    st.markdown("### 🎯 Composite Signal")
    sig_col1, sig_col2 = st.columns([1, 2])
    with sig_col1:
        st.metric(
            "Composite Score",
            f"{tech_summary['score']:+.1f} / 100",
            tech_summary["interpretation"],
        )
    with sig_col2:
        st.markdown("**Active signals:**")
        for sig, direction, _ in tech_summary["signals"]:
            color = "🟢" if direction == "bullish" else "🔴" if direction == "bearish" else "⚪"
            st.write(f"{color} {sig}")

    st.markdown("---")
    st.markdown("### 🔍 Detected Patterns")
    if patterns:
        for p in patterns:
            icon = "🟢" if p["signal"] == "bullish" else "🔴" if p["signal"] == "bearish" else "⚪"
            note = f" — _{p.get('note', '')}_" if p.get("note") else ""
            st.write(f"{icon} **{p['name']}** ({p['signal']}, {p['strength']}){note}")
    else:
        st.info("No major patterns detected in current window.")

    st.markdown("---")
    col_sr, col_fib = st.columns(2)
    with col_sr:
        st.markdown("### 📍 Support & Resistance")
        st.write(f"**Current price:** ${sr_levels['current']:.2f}")
        st.write("**Resistance levels (above):**")
        for r in sr_levels["resistance"]:
            st.write(f"- ${r:.2f} ({(r/sr_levels['current']-1)*100:+.1f}%)")
        st.write("**Support levels (below):**")
        for s in sr_levels["support"]:
            st.write(f"- ${s:.2f} ({(s/sr_levels['current']-1)*100:+.1f}%)")
    with col_fib:
        st.markdown("### 📐 Fibonacci Retracement (60d swing)")
        st.write(f"**Swing high:** ${fib['high']:.2f}")
        st.write(f"**Swing low:** ${fib['low']:.2f}")
        for level in ["23.6%", "38.2%", "50.0%", "61.8%", "78.6%"]:
            st.write(f"- {level}: ${fib[level]:.2f}")

    st.markdown("---")
    st.markdown("### 📊 Indicator Detail Charts")

    # MACD chart
    macd_data = macd(hist_1y["Close"])
    fig_macd = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5],
                             subplot_titles=("MACD", "Stochastic"))
    fig_macd.add_trace(go.Scatter(x=hist_1y.index, y=macd_data["macd"], name="MACD", line=dict(color="blue")), row=1, col=1)
    fig_macd.add_trace(go.Scatter(x=hist_1y.index, y=macd_data["signal"], name="Signal", line=dict(color="orange")), row=1, col=1)
    fig_macd.add_trace(go.Bar(x=hist_1y.index, y=macd_data["histogram"], name="Histogram", marker_color="gray"), row=1, col=1)

    stoch = stochastic(hist_1y)
    fig_macd.add_trace(go.Scatter(x=hist_1y.index, y=stoch["k"], name="%K", line=dict(color="purple")), row=2, col=1)
    fig_macd.add_trace(go.Scatter(x=hist_1y.index, y=stoch["d"], name="%D", line=dict(color="red")), row=2, col=1)
    fig_macd.add_hline(y=80, line_dash="dash", line_color="red", row=2, col=1)
    fig_macd.add_hline(y=20, line_dash="dash", line_color="green", row=2, col=1)
    fig_macd.update_layout(height=500, showlegend=True)
    st.plotly_chart(fig_macd, use_container_width=True)

# ============ QUANT TAB ============
with tab_quant:
    st.markdown("## 🎯 Quantitative Risk Profile")

    if risk.get("insufficient_data"):
        st.warning("Insufficient price history for full risk metrics")
    else:
        rcol1, rcol2, rcol3, rcol4 = st.columns(4)
        rcol1.metric("Annualized Return", fmt_pct(risk.get("annualized_return")))
        rcol2.metric("Annualized Volatility", fmt_pct(risk.get("annualized_volatility")))
        rcol3.metric("Sharpe Ratio", fmt_num(risk.get("sharpe_ratio")))
        rcol4.metric("Sortino Ratio", fmt_num(risk.get("sortino_ratio")))

        rcol5, rcol6, rcol7, rcol8 = st.columns(4)
        rcol5.metric("Max Drawdown", fmt_pct(risk.get("max_drawdown")))
        rcol6.metric("Calmar Ratio", fmt_num(risk.get("calmar_ratio")))
        rcol7.metric("Beta vs SPY", fmt_num(risk.get("beta")))
        rcol8.metric("Alpha (annualized)", fmt_pct(risk.get("alpha")))

        st.markdown("---")
        st.markdown("### 📉 Tail Risk")
        tcol1, tcol2, tcol3, tcol4 = st.columns(4)
        tcol1.metric("VaR 95% (daily)", fmt_pct(risk.get("var_95")))
        tcol2.metric("CVaR 95% (daily)", fmt_pct(risk.get("cvar_95")))
        tcol3.metric("VaR 99% (daily)", fmt_pct(risk.get("var_99")))
        tcol4.metric("CVaR 99% (daily)", fmt_pct(risk.get("cvar_99")))

        st.caption(f"Skewness: {fmt_num(risk.get('skewness'))} · Kurtosis: {fmt_num(risk.get('kurtosis'))} · R² vs market: {fmt_num(risk.get('r_squared'))}")

    st.markdown("---")
    st.markdown("### 🧬 Factor Profile")
    fc1, fc2, fc3 = st.columns(3)
    fc1.metric("Value Tilt", factors["value"])
    fc2.metric("Quality Tilt", factors["quality"])
    fc3.metric("Size", factors["size"])

    if factors.get("momentum_12_1") is not None:
        st.write(f"**12-1 month momentum:** {fmt_pct(factors['momentum_12_1'])} (vs market excess: {fmt_pct(factors.get('momentum_excess_vs_market'))})")
    if factors.get("realized_volatility") is not None:
        st.write(f"**Realized 1Y volatility:** {fmt_pct(factors['realized_volatility'])}")

    # Drawdown chart
    if not risk.get("insufficient_data") and not hist_5y.empty:
        from analysis.quant import max_drawdown
        dd = max_drawdown(hist_5y["Close"])
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=dd["drawdown_series"].index, y=dd["drawdown_series"] * 100,
            fill="tozeroy", name="Drawdown %", line=dict(color="red"),
        ))
        fig_dd.update_layout(title="Historical Drawdowns (5Y)", yaxis_title="Drawdown (%)", height=300)
        st.plotly_chart(fig_dd, use_container_width=True)

# ============ FILINGS TAB ============
with tab_filings:
    st.markdown("## 📑 SEC Filings (EDGAR)")
    if sec_data.get("available"):
        st.write(f"**CIK:** {sec_data['cik']} · **Company:** {sec_data['company_name']}")
        st.markdown("### Recent Key Filings")
        st.dataframe(sec_data["filings"], use_container_width=True, hide_index=True)
        st.markdown("**All recent filings:**")
        st.dataframe(sec_data["all_filings"], use_container_width=True, hide_index=True)
        st.caption("Source: data.sec.gov · Filings linked above can be fetched directly from EDGAR.")
    else:
        st.info("No SEC filings available (likely a non-US listing, ETF, or CIK lookup failed)")

st.markdown("---")
st.caption(f"Analysis generated {datetime.now().strftime('%Y-%m-%d %H:%M')} · Data: yfinance + Stooq cross-check + SEC EDGAR · Risk-free rate: {risk_free*100:.2f}% (^TNX)")
st.caption("⚠️ This is an analytical tool, not investment advice. All forecasts are probabilistic scenarios based on historical data and explicit assumptions.")
