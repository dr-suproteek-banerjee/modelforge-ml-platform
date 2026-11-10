from importlib.util import find_spec
from io import BytesIO

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted, validate_data


def torch_available() -> bool:
    return find_spec("torch") is not None


class TorchMLP(BaseEstimator):
    """A small CPU network with portable, numeric-only inference state."""

    def __init__(self, seed: int = 42, hidden: int = 32, learning_rate: float = 0.01):
        self.seed = seed
        self.hidden = hidden
        self.learning_rate = learning_rate

    def fit(self, X, y):
        import torch
        from torch import nn

        X, y = validate_data(self, X, y, dtype=np.float32)
        self.target_mean_ = 0.0
        self.target_scale_ = 1.0
        loss: nn.Module
        if isinstance(self, TorchClassifier):
            self.classes_, indexes = np.unique(y, return_inverse=True)
            target = torch.as_tensor(indexes, dtype=torch.long)
            loss = nn.CrossEntropyLoss()
            outputs = len(self.classes_)
        else:
            self.target_mean_ = float(np.mean(y))
            self.target_scale_ = float(np.std(y)) or 1.0
            target = torch.as_tensor(
                (y - self.target_mean_) / self.target_scale_, dtype=torch.float32
            )
            loss = nn.MSELoss()
            outputs = 1
        inputs = torch.from_numpy(np.asarray(X, dtype=np.float32))
        generator = torch.Generator().manual_seed(self.seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            network = nn.Sequential(
                nn.Linear(X.shape[1], self.hidden), nn.ReLU(), nn.Linear(self.hidden, outputs)
            )
            optimizer = torch.optim.AdamW(network.parameters(), lr=self.learning_rate)
            for _ in range(75):
                batches = torch.randperm(len(inputs), generator=generator).split(256)
                for batch in batches:
                    optimizer.zero_grad(set_to_none=True)
                    scores = network(inputs[batch])
                    objective = loss(scores if outputs > 1 else scores[:, 0], target[batch])
                    objective.backward()
                    optimizer.step()
            self.weights_ = tuple(
                parameter.detach().numpy().copy() for parameter in network.parameters()
            )
        if not all(np.isfinite(value).all() for value in self.weights_):
            raise ValueError(
                "Neural training diverged. Standardize numeric features or lower the learning rate."
            )
        return self

    def _forward(self, X):
        check_is_fitted(self, "weights_")
        X = validate_data(self, X, reset=False, dtype=np.float32)
        weights, bias, output_weights, output_bias = self.weights_
        result = np.maximum(X @ weights.T + bias, 0) @ output_weights.T + output_bias
        if not np.isfinite(result).all():
            raise ValueError("Neural inference produced non-finite values for this input.")
        return result

    def export_weights(self) -> bytes:
        archive = BytesIO()
        weights, bias, output_weights, output_bias = self.weights_
        np.savez(
            archive,
            weights=weights,
            bias=bias,
            output_weights=output_weights,
            output_bias=output_bias,
            scale=np.array([self.target_mean_, self.target_scale_], dtype=float),
        )
        return archive.getvalue()

    def import_weights(self, data: bytes, labels: int) -> None:
        outputs = labels or 1
        with np.load(BytesIO(data), allow_pickle=False) as archive:
            weights, bias, output_weights, output_bias, scale = (
                archive[key]
                for key in ("weights", "bias", "output_weights", "output_bias", "scale")
            )
        if (
            weights.ndim != 2
            or weights.shape[0] != self.hidden
            or weights.shape[1] == 0
            or bias.shape != (self.hidden,)
            or output_weights.shape != (outputs, self.hidden)
            or output_bias.shape != (outputs,)
            or scale.shape != (2,)
            or scale[1] <= 0
            or not all(
                np.isfinite(x).all() for x in (weights, bias, output_weights, output_bias, scale)
            )
        ):
            raise ValueError("The neural model has incompatible weights.")
        self.weights_ = (weights, bias, output_weights, output_bias)
        self.n_features_in_ = weights.shape[1]
        self.target_mean_, self.target_scale_ = map(float, scale)
        if isinstance(self, TorchClassifier):
            self.classes_ = np.arange(labels)


class TorchClassifier(ClassifierMixin, TorchMLP):
    def predict_proba(self, X):
        logits = self._forward(X)
        shifted = np.exp(logits - logits.max(axis=1, keepdims=True))
        return shifted / shifted.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class TorchRegressor(RegressorMixin, TorchMLP):
    def predict(self, X):
        return self._forward(X)[:, 0] * self.target_scale_ + self.target_mean_
