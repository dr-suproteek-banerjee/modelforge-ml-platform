import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

from .schemas import DriftFeature


def baseline_for(frame: pd.DataFrame, schema: dict[str, str], seed: int) -> dict:
    result = {}
    for feature, kind in schema.items():
        values = frame[feature].dropna()
        missing = float(frame[feature].isna().mean())
        if kind == "numeric":
            sample = values.sample(n=min(2048, len(values)), random_state=seed).tolist()
            result[feature] = {"kind": kind, "missing": missing, "sample": sample}
        else:
            counts = values.value_counts()
            kept = {str(key): int(count) for key, count in counts.head(24).items()}
            result[feature] = {
                "kind": kind,
                "missing": missing,
                "counts": kept,
                "other": int(len(values) - sum(kept.values())),
            }
    return result


def compare(baseline: dict, frame: pd.DataFrame) -> list[DriftFeature]:
    results: list[DriftFeature] = []
    for feature, reference in baseline.items():
        values = frame[feature].dropna()
        missing = float(frame[feature].isna().mean())
        base_missing = reference["missing"]
        missing_shift = abs(missing - base_missing) >= 0.10
        if reference["kind"] == "numeric":
            sample = reference["sample"]
            if len(sample) < 5 or len(values) < 5:
                results.append(
                    DriftFeature(
                        feature=feature,
                        method="insufficient data",
                        statistic=0,
                        flagged=missing_shift,
                        baseline_missing=base_missing,
                        current_missing=missing,
                    )
                )
                continue
            observed = values.sample(n=min(2048, len(values)), random_state=42).to_numpy(
                dtype=float
            )
            test = ks_2samp(sample, observed, method="asymp")
            bounds = np.unique(np.quantile(sample, np.linspace(0, 1, 6)[1:-1]))
            edges = [-np.inf, *bounds, np.inf]
            keys = [f"[{a:.4g}, {b:.4g})" for a, b in zip(edges[:-1], edges[1:], strict=True)]
            left = dict(zip(keys, np.histogram(sample, bins=edges)[0].tolist(), strict=True))
            right = dict(zip(keys, np.histogram(observed, bins=edges)[0].tolist(), strict=True))
            results.append(
                DriftFeature(
                    feature=feature,
                    method="KS",
                    statistic=float(test.statistic),
                    p_value=float(test.pvalue),
                    flagged=missing_shift,
                    baseline_missing=base_missing,
                    current_missing=missing,
                    baseline=left,
                    current=right,
                )
            )
        else:
            counts = reference["counts"]
            current = {key: int((values == key).sum()) for key in counts}
            left = {f"Category: {key}": value for key, value in counts.items()}
            right = {f"Category: {key}": value for key, value in current.items()}
            left["Other"] = reference["other"]
            right["Other"] = len(values) - sum(current.values())
            usable = sum(left.values()) > 0 and sum(right.values()) > 0
            distance = (
                float(jensenshannon(list(left.values()), list(right.values()), base=2))
                if usable
                else 0.0
            )
            results.append(
                DriftFeature(
                    feature=feature,
                    method="Jensen–Shannon distance" if usable else "insufficient data",
                    statistic=distance,
                    flagged=distance >= 0.2 or missing_shift,
                    baseline_missing=base_missing,
                    current_missing=missing,
                    baseline=left,
                    current=right,
                )
            )
    numeric = sorted(
        (r for r in results if r.p_value is not None), key=lambda r: float(r.p_value or 0)
    )
    previous = 0.0
    for index, row in enumerate(numeric):
        adjusted = min(1.0, float(row.p_value or 0) * (len(numeric) - index))
        previous = max(previous, adjusted)
        row.adjusted_p_value = previous
        row.flagged = row.flagged or (row.adjusted_p_value < 0.05 and row.statistic >= 0.10)
    return results
