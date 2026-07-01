"""One-time script: build data/processed/prices.parquet and data/processed/universe.csv.

Dataset covers 1980–2020-04-01. We rank tickers by 2019 average volume (last full year).

Usage:
    python -m src.data.prepare
"""

from __future__ import annotations

import csv
import io
import logging
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import duckdb
import pandas as pd

# prepare.py only needs DATASET_RAW_DIR — allow missing Azure keys at import time
os.environ.setdefault("AZURE_OPENAI_KEY", "not-needed-for-prepare")
os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://not-needed-for-prepare")

from src.settings import get_settings  # noqa: E402

logger = logging.getLogger(__name__)

RANK_YEAR = 2019
TOP_N = 50
MIN_ROWS_IN_RANK_YEAR = 200

PROCESSED = Path("data/processed")


def _tail_volume(args: tuple[str, str]) -> tuple[str, float] | None:
    """Worker: read last 350 lines of a CSV, return (ticker, avg_volume) for RANK_YEAR."""
    fpath, ticker = args
    try:
        # tail last 350 lines — enough to cover a full trading year
        result = subprocess.run(
            ["tail", "-n", "350", fpath],
            capture_output=True, text=True, timeout=5
        )
        rows = list(csv.reader(io.StringIO(result.stdout)))
        volumes = []
        for row in rows:
            if len(row) < 7:
                continue
            try:
                if row[0].startswith(str(RANK_YEAR)):
                    volumes.append(float(row[6]))
            except (ValueError, IndexError):
                continue
        if len(volumes) >= MIN_ROWS_IN_RANK_YEAR:
            return ticker, sum(volumes) / len(volumes)
    except Exception:
        pass
    return None


def build_universe(raw_dir: Path) -> list[str]:
    """Rank NASDAQ stock tickers by 2019 avg volume using parallel tail reads."""
    csv_files = list((raw_dir / "stocks").glob("*.csv"))
    logger.info(
        "Ranking %d tickers by %d volume (parallel tail reads) ...",
        len(csv_files), RANK_YEAR
    )

    tasks = [(str(f), f.stem) for f in csv_files]
    ranks: list[tuple[str, float]] = []

    with ProcessPoolExecutor() as pool:
        futures = {pool.submit(_tail_volume, t): t for t in tasks}
        done = 0
        for fut in as_completed(futures):
            done += 1
            if done % 500 == 0:
                logger.info("  ... %d / %d scanned", done, len(tasks))
            result = fut.result()
            if result is not None:
                ranks.append(result)

    ranks.sort(key=lambda x: -x[1])
    top = [t for t, _ in ranks[:TOP_N]]
    logger.info("Top %d tickers selected: %s", TOP_N, top)
    return top


def build_parquet(tickers: list[str], raw_dir: Path) -> None:
    """Read top-50 CSVs and write a partitioned Parquet dataset via DuckDB."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED / "prices.parquet"

    union_parts = []
    for ticker in tickers:
        fpath = raw_dir / "stocks" / f"{ticker}.csv"
        if not fpath.exists():
            logger.warning("File not found for %s, skipping.", ticker)
            continue
        safe = str(fpath).replace("'", "''")
        union_parts.append(f"""
            SELECT
                CAST(Date AS DATE)              AS date,
                CAST("Open" AS DOUBLE)          AS open,
                CAST("High" AS DOUBLE)          AS high,
                CAST("Low" AS DOUBLE)           AS low,
                CAST("Close" AS DOUBLE)         AS close,
                CAST("Adj Close" AS DOUBLE)     AS adj_close,
                CAST("Volume" AS BIGINT)        AS volume,
                '{ticker}'                      AS ticker
            FROM read_csv('{safe}', ignore_errors=true)
        """)

    if not union_parts:
        raise RuntimeError("No ticker CSVs found.")

    con = duckdb.connect()
    full_query = " UNION ALL ".join(union_parts)
    logger.info("Writing parquet for %d tickers ...", len(union_parts))
    con.execute(f"""
        COPY ({full_query})
        TO '{out_path}'
        (FORMAT PARQUET, PARTITION_BY (ticker), OVERWRITE_OR_IGNORE)
    """)
    logger.info("Parquet written to %s", out_path)


def build_universe_csv(tickers: list[str]) -> None:
    pd.DataFrame({"ticker": tickers}).to_csv(PROCESSED / "universe.csv", index=False)
    logger.info("Universe CSV written with %d tickers.", len(tickers))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()
    raw_dir = Path(settings.dataset_raw_dir)

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {raw_dir.resolve()}\n"
            "Set DATASET_RAW_DIR in .env to point at the Kaggle dataset root."
        )

    tickers = build_universe(raw_dir)
    build_parquet(tickers, raw_dir)
    build_universe_csv(tickers)
    print(f"\nDone. Top-{TOP_N} tickers:\n{tickers}")


if __name__ == "__main__":
    main()
