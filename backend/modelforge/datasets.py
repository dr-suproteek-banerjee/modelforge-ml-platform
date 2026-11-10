import csv
from io import StringIO

import numpy as np
import pandas as pd

from .schemas import ColumnProfile, ProfileView


def read_csv(data: bytes, *, max_rows: int = 25_000) -> pd.DataFrame:
    try:
        text = data.decode("utf-8-sig")
        reader = csv.reader(StringIO(text, newline=""), strict=True)
        columns = next(reader, [])
        if (
            not columns
            or len(columns) > 100
            or len(set(columns)) != len(columns)
            or any(not x.strip() or len(x) > 120 or "\x00" in x for x in columns)
        ):
            raise ValueError(
                "Provide 1–100 unique, nonblank column names of at most 120 characters."
            )
        for count, row in enumerate(reader, 1):
            if count > max_rows:
                raise ValueError(f"CSV exceeds the {max_rows:,}-row limit.")
            if len(row) != len(columns) or any(len(cell) > 4096 or "\x00" in cell for cell in row):
                raise ValueError(
                    "Each row must match the header. Cells may contain at most 4,096 characters."
                )
        frame = pd.read_csv(StringIO(text), keep_default_na=False, na_values=[""])
    except (UnicodeError, csv.Error, pd.errors.ParserError) as exc:
        raise ValueError("Upload a valid UTF-8 CSV with comma-separated fields.") from exc
    if frame.empty:
        raise ValueError("CSV must contain at least one data row.")
    for name in frame.select_dtypes(include="number"):
        values = frame[name].dropna().to_numpy(dtype=float)
        if not np.isfinite(values).all() or (np.abs(values) > 1e30).any():
            raise ValueError("Numeric cells must be finite and at most 1e30 in absolute value.")
    return frame


def profile(frame: pd.DataFrame) -> ProfileView:
    columns = []
    for name, values in frame.items():
        present = values.dropna()
        stats: dict[str, float] = {}
        distribution: list[dict[str, str | int]] = []
        if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
            stats = {str(k): float(v) for k, v in present.describe().dropna().items()}
            if len(present):
                counts, edges = np.histogram(present, bins=10)
                distribution = [
                    {"label": f"{left:.4g} to {right:.4g}", "count": int(count)}
                    for left, right, count in zip(edges[:-1], edges[1:], counts, strict=True)
                ]
        else:
            distribution = [
                {"label": str(label), "count": int(count)}
                for label, count in present.value_counts().head(10).items()
            ]
        columns.append(
            ColumnProfile(
                name=str(name),
                dtype=str(values.dtype),
                missing=int(values.isna().sum()),
                unique=int(values.nunique()),
                statistics=stats,
                distribution=distribution,
            )
        )
    corr = frame.select_dtypes(include="number").corr().round(5)
    return ProfileView(
        rows=len(frame),
        columns=columns,
        preview=frame.head(12)
        .astype(object)
        .where(frame.head(12).notna(), None)
        .to_dict("records"),
        correlations=corr.astype(object).where(corr.notna(), None).to_dict(),
    )
