"""Regenerate the checked-in dataset fixtures with the locked Python environment."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

out = Path(__file__).resolve().parent.parent / "samples"
out.mkdir(exist_ok=True)
dataset = load_iris(as_frame=True)
iris = dataset.frame.rename(
    columns={name: name.removesuffix(" (cm)").replace(" ", "_") for name in dataset.feature_names}
)
iris["species"] = iris.pop("target").map(dict(enumerate(dataset.target_names)))
iris.to_csv(out / "iris.csv", index=False)
shifted = iris.copy()
shifted["sepal_length"] += 4
shifted.to_csv(out / "iris_shifted.csv", index=False)

rng = np.random.default_rng(42)
area = rng.uniform(450, 2100, 240)
age = rng.integers(0, 50, len(area))
district = rng.choice(["central", "east", "west"], len(area))
price = 70 * area - 400 * age + 25_000 * (district == "central") + rng.normal(0, 2000, len(area))
pd.DataFrame({"area": area, "age": age, "district": district, "price": price}).to_csv(
    out / "regression.csv", index=False, float_format="%.5f"
)
