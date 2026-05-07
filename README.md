"""
Data Sources Module
-------------------
Pulls live data from multiple sources and cross-references them.
Primary: yfinance (Yahoo Finance)
Cross-check: Stooq (independent price feed)
Fundamentals reference: SEC EDGAR (source of truth for filings)
Macro: FRED (via pandas-datareader)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from io import StringIO
from datetime import datetime, timedelta
import streamlit as st


@st.cache_data(ttl=900)  # 15-minute cache
def fetch_yfinance_core(ticker: str) -> dict:
    """Fetch core data from yfinance with defensive error handling."""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        hist_1y = t.history(period="1y", auto_adjust=True)
        hist_5y = t.history(period="5y", auto_adjust=True)

        return {
            "info": info,
            "hist_1y": hist_1y,
            "hist_5y": hist_5y,
            "financials": t.financials,
            "quarterly_financials": t.quarterly_financials,
            "balance_sheet": t.balance_sheet,
            "cashflow": t.cashflow,
            "quarterly_cashflow": t.quarterly_cashflow,
            "earnings_dates": t.earnings_dates if hasattr(t, "earnings_dates") else None,
            "recommendations": t.recommendations if hasattr(t, "recommendations") else None,
            "institutional_holders": t.institutional_holders if hasattr(t, "institutional_holders") else None,
            "major_holders": t.major_holders if hasattr(t, "major_holders") else None,
            "success": True,
            "error": None,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "info": {}, "hist_1y": pd.DataFrame(), "hist_5y": pd.DataFrame()}


@st.cache_data(ttl=900)
def fetch_stooq_price(ticker: str) -> pd.DataFrame:
    """Independent price feed from Stooq for cross-checking."""
    try:
        url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"
        r = requests.get(url, timeout=10)
        if r.status_code != 200 or len(r.text) < 100:
            return pd.DataFrame()
        df = pd.read_csv(StringIO(r.text))
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def fetch_sec_filings(ticker: str) -> dict:
    """Pull recent SEC filings via EDGAR. Requires CIK lookup."""
    try:
        # SEC requires User-Agent
        headers = {"User-Agent": "StockAnalyzer research@example.com"}
        # Get CIK from ticker
        r = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=headers,
            timeout=10,
        )
        if r.status_code != 200:
            return {"available": False}
        tickers_data = r.json()
        cik = None
        for entry in tickers_data.values():
            if entry["ticker"].upper() == ticker.upper():
                cik = str(entry["cik_str"]).zfill(10)
                company_name = entry["title"]
                break
        if not cik:
            return {"available": False}

        # Get recent filings
        r = requests.get(
            f"https://data.sec.gov/submissions/CIK{cik}.json",
            headers=headers,
            timeout=10,
        )
        if r.status_code != 200:
            return {"available": False, "cik": cik}
        data = r.json()
        recent = data.get("filings", {}).get("recent", {})
        filings_df = pd.DataFrame({
            "Form": recent.get("form", []),
            "Date": recent.get("filingDate", []),
            "Accession": recent.get("accessionNumber", []),
        })
        # Keep only key forms
        key_forms = filings_df[filings_df["Form"].isin(["10-K", "10-Q", "8-K", "DEF 14A", "S-1"])].head(15)
        return {
            "available": True,
            "cik": cik,
            "company_name": company_name,
            "filings": key_forms,
            "all_filings": filings_df.head(20),
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


@st.cache_data(ttl=3600)
def fetch_market_benchmark() -> pd.DataFrame:
    """Fetch S&P 500 (SPY) for benchmark comparisons."""
    try:
        spy = yf.Ticker("SPY")
        return spy.history(period="5y", auto_adjust=True)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def fetch_risk_free_rate() -> float:
    """Fetch 10-year Treasury yield as risk-free rate proxy. Falls back to 4.5%."""
    try:
        tnx = yf.Ticker("^TNX")
        hist = tnx.history(period="5d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1]) / 100  # TNX is in percent
    except Exception:
        pass
    return 0.045  # sensible fallback


def cross_check_price(yf_price: float, stooq_df: pd.DataFrame) -> dict:
    """Compare yfinance price to Stooq for sanity check."""
    if stooq_df.empty or yf_price is None:
        return {"checked": False, "match": None, "stooq_price": None, "diff_pct": None}
    stooq_price = float(stooq_df["Close"].iloc[-1])
    diff_pct = abs(yf_price - stooq_price) / yf_price * 100
    return {
        "checked": True,
        "match": diff_pct < 2.0,  # within 2% considered matching
        "stooq_price": stooq_price,
        "yf_price": yf_price,
        "diff_pct": diff_pct,
    }
