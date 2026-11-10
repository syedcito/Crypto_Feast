import logging
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = ROOT / "data" / "raw" / "candles.parquet"
OUT_DIR = ROOT / "data" / "features"
OUT_FILE = OUT_DIR / "crypto_features.parquet"

WINDOW = 24
FEATURE_COLS = [
    "return_1h",
    "return_24h",
    "volatility_24h",
    "volume_24h",
    "vol_ratio",
    "range_pct",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    df = pd.read_parquet(RAW_FILE)
    df = df.sort_values(["product_id", "event_timestamp"]).reset_index(drop=True)
    logger.info("Read %d raw candles from %s", len(df), RAW_FILE)

    # Every rolling/shift/pct_change below is computed per coin, using only past rows,
    # so windows never cross from one coin into the next or look ahead.
    df["return_1h"] = df.groupby("product_id")["close"].pct_change(1)
    df["return_24h"] = df.groupby("product_id")["close"].pct_change(WINDOW)
    # Regroup after adding return_1h so the new column is visible to the groupby.
    df["volatility_24h"] = df.groupby("product_id")["return_1h"].transform(
        lambda s: s.rolling(WINDOW).std()
    )
    df["volume_24h"] = df.groupby("product_id")["volume"].transform(
        lambda s: s.rolling(WINDOW).sum()
    )
    avg_volume_24h = df.groupby("product_id")["volume"].transform(
        lambda s: s.rolling(WINDOW).mean()
    )
    df["vol_ratio"] = df["volume"] / avg_volume_24h
    df["range_pct"] = (df["high"] - df["low"]) / df["close"]

    # A candle stamped T covers [T, T+1h), so its close isn't known until T+1h.
    # Stamp each feature row with the time its value became knowable, to avoid leakage.
    df["event_timestamp"] = df["event_timestamp"] + pd.Timedelta(hours=1)
    df["created"] = pd.Timestamp.now(tz="UTC")

    out = df[["product_id", "event_timestamp", "created"] + FEATURE_COLS]
    out = out.dropna(subset=FEATURE_COLS)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_FILE, index=False)
    logger.info("Saved %d feature rows to %s", len(out), OUT_FILE)
    logger.info("Rows per coin:\n%s", out.groupby("product_id").size().to_string())


if __name__ == "__main__":
    main()