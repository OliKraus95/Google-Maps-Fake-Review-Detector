"""Create a smaller debug raw CSV from the newest raw dataset.

This helper script derives a compact debug dataset from the newest raw CSV in
``config.DATA_RAW_DIR``. The output file is named with the ``reviews_complete``
prefix so that ``scripts/preparation.py`` automatically prefers it on the next
pipeline run.

Sampling strategy:
- Preserve all restaurants via stratified sampling by restaurant column.
- Keep a minimum number of rows per restaurant to retain signal for downstream
  temporal/network/semantic analyses.
- Downsample aggressively to speed up debugging iterations.
"""

import csv
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    from scripts import config
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts import config

logger = logging.getLogger(__name__)


def _find_latest_raw_csv() -> Path:
    """Return the most recent raw CSV file in DATA_RAW_DIR.

    Returns:
        Path to the newest raw CSV.

    Raises:
        FileNotFoundError: If no raw CSV is found.
    """
    csv_files = sorted(config.DATA_RAW_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {config.DATA_RAW_DIR}")
    return csv_files[-1]


def _pick_restaurant_column(df: pd.DataFrame) -> str | None:
    """Infer restaurant column name used for stratified sampling.

    Args:
        df: Input raw DataFrame.

    Returns:
        Name of the restaurant column or None if not found.
    """
    for candidate in ["Restaurant", "place_name", "Google Maps URL", "place_url"]:
        if candidate in df.columns:
            return candidate
    return None


def _compute_target_size(n_rows: int) -> int:
    """Compute a debug-friendly target size.

    Args:
        n_rows: Number of rows in source dataset.

    Returns:
        Target row count for sampled debug dataset.
    """
    target = int(n_rows * 0.15)
    target = max(5000, target)
    target = min(9000, target)
    return min(target, n_rows)


def _stratified_sample(df: pd.DataFrame, group_col: str, target_rows: int) -> pd.DataFrame:
    """Create a stratified sample while preserving all groups.

    Args:
        df: Source DataFrame.
        group_col: Column used for grouping.
        target_rows: Desired total number of sampled rows.

    Returns:
        Sampled DataFrame.
    """
    if target_rows >= len(df):
        return df.copy()

    grouped = df.groupby(group_col, dropna=False)
    min_per_group = 20

    sampled_parts = []
    for _, group in grouped:
        proportional_n = int(round(len(group) / len(df) * target_rows))
        n = max(min_per_group, proportional_n)
        n = min(n, len(group))
        sampled_parts.append(group.sample(n=n, random_state=42))

    sampled = pd.concat(sampled_parts, ignore_index=True)

    # Keep one row per group before trimming to exact target size.
    if len(sampled) > target_rows:
        one_per_group = sampled.groupby(group_col, dropna=False).head(1)
        remaining_pool = sampled.drop(index=one_per_group.index)
        needed = max(0, target_rows - len(one_per_group))
        if needed > 0 and len(remaining_pool) > 0:
            remaining = remaining_pool.sample(n=min(needed, len(remaining_pool)), random_state=42)
            sampled = pd.concat([one_per_group, remaining], ignore_index=True)
        else:
            sampled = one_per_group

    return sampled.sample(frac=1.0, random_state=42).reset_index(drop=True)


def create_debug_sample() -> Path:
    """Create a compact debug raw CSV and return the created file path.

    Returns:
        Path to the created debug CSV.
    """
    source_path = _find_latest_raw_csv()
    logger.info(f"Using latest raw CSV: {source_path.name}")

    df = pd.read_csv(
        source_path,
        sep=";",
        encoding="utf-8-sig",
        quoting=csv.QUOTE_ALL,
        dtype=str,
    )

    source_rows = len(df)
    target_rows = _compute_target_size(source_rows)
    group_col = _pick_restaurant_column(df)

    if group_col is not None:
        sampled = _stratified_sample(df, group_col=group_col, target_rows=target_rows)
        logger.info(f"Stratified sampling by '{group_col}'")
    else:
        sampled = df.sample(n=target_rows, random_state=42).reset_index(drop=True)
        logger.warning("No grouping column found, using random sampling")

    output_name = f"reviews_complete_debug_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    output_path = config.DATA_RAW_DIR / output_name

    sampled.to_csv(
        output_path,
        sep=";",
        encoding="utf-8-sig",
        quoting=csv.QUOTE_ALL,
        index=False,
    )

    logger.info(f"Created debug CSV: {output_path.name}")
    logger.info(f"Rows: source={source_rows:,}, sampled={len(sampled):,}")

    if group_col is not None:
        source_groups = df[group_col].nunique(dropna=False)
        sampled_groups = sampled[group_col].nunique(dropna=False)
        logger.info(f"Group coverage ({group_col}): {sampled_groups}/{source_groups}")

    return output_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    created = create_debug_sample()
    logger.info(f"Debug sample ready for next pipeline run: {created}")
