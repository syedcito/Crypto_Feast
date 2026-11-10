import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.exchange.coinbase.com"
GRANULARITY = 3600
PRODUCTS = ["BTC-USD", "ETH-USD", "SOL-USD"]
LOOKBACK_DAYS = 60
CHUNK_HOURS = 299

# Anchor output to the project root (the parent of scripts/), so the file lands
# in the same place no matter which folder the script is launched from.
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "raw"
OUT_FILE = OUT_DIR / "candles.parquet"

logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def request_candles(product_id, start, end):
	response = requests.get(
		f"{BASE_URL}/products/{product_id}/candles",
		params={
			"granularity": GRANULARITY,
			"start": start.isoformat(),
			"end": end.isoformat(),
		},
		timeout=10,
	)
	response.raise_for_status()
	return response.json()


def main():
	end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
	start = end - timedelta(days=LOOKBACK_DAYS)
	logger.info("Starting candle fetch for %s", ", ".join(PRODUCTS))
	logger.info("Fetching %d windows per product from %s through %s", (LOOKBACK_DAYS * 24 + CHUNK_HOURS - 1) // CHUNK_HOURS, start.isoformat(), end.isoformat())

	windows = []
	window_start = start
	while window_start < end:
		window_end = min(window_start + timedelta(hours=CHUNK_HOURS), end)
		windows.append((window_start, window_end))
		window_start = window_end

	tasks = [
		(product_id, window_start, window_end)
		for product_id in PRODUCTS
		for window_start, window_end in windows
	]

	coin_chunks = {}
	for request_index, (product_id, window_start, window_end) in enumerate(tasks):
		if request_index:
			time.sleep(0.3)

		logger.info(
			"Request %d/%d: %s from %s to %s",
			request_index + 1,
			len(tasks),
			product_id,
			window_start.isoformat(),
			window_end.isoformat(),
		)
		try:
			rows = request_candles(product_id, window_start, window_end)
		except requests.RequestException as exc:
			raise RuntimeError(
				f"Request failed for {product_id} from {window_start.isoformat()} "
				f"through {window_end.isoformat()}"
			) from exc
		logger.info("Received %d candles for %s", len(rows), product_id)

		chunk = pd.DataFrame(
			rows,
			columns=["time", "low", "high", "open", "close", "volume"],
		)
		chunk["time"] = pd.to_datetime(chunk["time"], unit="s", utc=True)
		chunk = chunk.rename(columns={"time": "event_timestamp"})
		chunk["product_id"] = product_id
		chunk[["low", "high", "open", "close", "volume"]] = chunk[
			["low", "high", "open", "close", "volume"]
		].astype(float)
		coin_chunks.setdefault(product_id, []).append(chunk)

	all_coins = [
		pd.concat(coin_chunks[product_id], ignore_index=True)
		for product_id in PRODUCTS
	]
	df = pd.concat(all_coins, ignore_index=True)
	df = df.drop_duplicates(subset=["product_id", "event_timestamp"])
	df = df.sort_values(["product_id", "event_timestamp"], ascending=True)
	expected_latest = pd.Timestamp(end)
	for product_id in PRODUCTS:
		latest = df.loc[df["product_id"] == product_id, "event_timestamp"].max()
		if pd.isna(latest) or latest != expected_latest:
			logger.warning(
				"Latest candle for %s is %s; expected previous full hour %s",
				product_id,
				latest,
				expected_latest,
			)
		else:
			logger.info("Latest candle verified for %s: %s", product_id, latest)

	OUT_DIR.mkdir(parents=True, exist_ok=True)
	df.to_parquet(OUT_FILE, index=False)
	logger.info("Completed: saved %d candles to %s", len(df), OUT_FILE)


if __name__ == "__main__":
	try:
		main()
	except Exception:
		logger.exception("Candle fetch failed")
		raise