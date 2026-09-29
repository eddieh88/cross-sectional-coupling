"""
Experiment 6 setup -- pull a real multi-asset daily-close panel from
Coinbase's public (unauthenticated) Exchange API, as the actual escape
hatch from the "classically-tractable generator" caveat surfaced during
Experiment 5: every synthetic generator in this project was built from a
named closed-form stochastic primitive that classical statistics already
has a near-optimal estimator for. Real market data isn't the output of
any such hand-picked process, so it's the one test that isn't subject to
that critique by construction (see results/SUMMARY.md's meta-caveat
section).

Universe: 30 large-cap, Coinbase-listed assets, verified to have >=3 years
of continuous daily history on Coinbase as of this run (checked directly
via spot probes at the 2-year and 3-year marks before committing to this
list). All quoted in USD, all public endpoints, no API
key required.

Endpoint: GET /products/{ID}/candles?granularity=86400&start=..&end=..
Coinbase caps each response at ~300 candles, so a 3-year (~1095 day) pull
needs pagination -- fetched in ~250-day chunks, oldest-to-newest, with a
short sleep between requests to stay well under the public rate limit.
"""

import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
import urllib.request
import json

import numpy as np
import pandas as pd

UNIVERSE = [
    "BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "AVAX", "DOT", "LINK", "LTC",
    "BCH", "ATOM", "UNI", "ETC", "FIL", "NEAR", "ICP", "ALGO", "XLM", "AAVE",
    "XTZ", "COMP", "SNX", "CRV", "GRT", "APT", "ARB", "OP", "INJ", "SUI",
]

GRANULARITY_SECONDS = 86400  # daily
LOOKBACK_DAYS = 1095  # 3 years
CHUNK_DAYS = 250  # comfortably under Coinbase's ~300-candle response cap
REQUEST_SLEEP_SEC = 0.2

BASE_URL = "https://api.exchange.coinbase.com"


def fetch_candles(product_id: str, start: datetime, end: datetime):
    url = (f"{BASE_URL}/products/{product_id}/candles"
           f"?granularity={GRANULARITY_SECONDS}"
           f"&start={start.strftime('%Y-%m-%dT%H:%M:%S')}"
           f"&end={end.strftime('%Y-%m-%dT%H:%M:%S')}")
    req = urllib.request.Request(url, headers={"User-Agent": "research-script"})
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.load(r)
    # Coinbase candle row: [time, low, high, open, close, volume]
    return data


def fetch_full_history(symbol: str, lookback_days: int) -> pd.Series:
    product_id = f"{symbol}-USD"
    end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=lookback_days)

    rows = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS), end)
        candles = fetch_candles(product_id, chunk_start, chunk_end)
        rows.extend(candles)
        time.sleep(REQUEST_SLEEP_SEC)
        chunk_start = chunk_end

    if not rows:
        return pd.Series(dtype=float)

    df = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
    df["date"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.normalize()
    df = df.drop_duplicates(subset="date").sort_values("date")
    return df.set_index("date")["close"]


def main():
    print(f"Fetching {len(UNIVERSE)} assets, {LOOKBACK_DAYS} days of daily closes from Coinbase...")
    series = {}
    for i, symbol in enumerate(UNIVERSE):
        try:
            s = fetch_full_history(symbol, LOOKBACK_DAYS)
            series[symbol] = s
            print(f"  [{i+1}/{len(UNIVERSE)}] {symbol}: {len(s)} rows, "
                  f"{s.index.min().date() if len(s) else 'n/a'} -> {s.index.max().date() if len(s) else 'n/a'}")
        except Exception as e:
            print(f"  [{i+1}/{len(UNIVERSE)}] {symbol}: FAILED ({e})")

    panel = pd.DataFrame(series)
    n_before = len(panel)
    panel_complete = panel.dropna(how="any")
    n_after = len(panel_complete)
    print(f"\nRaw joined panel: {n_before} dates x {panel.shape[1]} assets")
    print(f"After dropping any date with a missing asset: {n_after} dates "
          f"({n_before - n_after} dates dropped)")

    out_path = Path(__file__).resolve().parent / "coinbase_daily_panel_3y.parquet"
    panel_complete.to_parquet(out_path)
    print(f"\nSaved: {out_path}  shape={panel_complete.shape}")
    print(f"Date range: {panel_complete.index.min().date()} -> {panel_complete.index.max().date()}")


if __name__ == "__main__":
    main()
