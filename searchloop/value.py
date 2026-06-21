from __future__ import annotations

import json
from collections.abc import Callable

import numpy as np


def fit_standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = np.maximum(X.std(axis=0), 1e-8)
    return mean, std


def apply_standardizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


class ValueMLP:
    def __init__(self, input_dim: int = 7, hidden_dim: int = 16, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.W1 = rng.normal(0, np.sqrt(2 / input_dim), (input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0, np.sqrt(2 / hidden_dim), (hidden_dim, 1))
        self.b2 = np.zeros(1)
        self._Z1: np.ndarray | None = None
        self._A1: np.ndarray | None = None

    def forward(self, Xs: np.ndarray) -> np.ndarray:
        self._Z1 = Xs @ self.W1 + self.b1
        self._A1 = np.maximum(self._Z1, 0)
        return self._A1 @ self.W2 + self.b2

    def train(
        self,
        Xs: np.ndarray,
        y: np.ndarray,
        epochs: int = 500,
        lr: float = 0.01,
    ) -> list[float]:
        losses = []
        n = Xs.shape[0]
        for _ in range(epochs):
            Z1 = Xs @ self.W1 + self.b1
            A1 = np.maximum(Z1, 0)
            P = A1 @ self.W2 + self.b2
            loss = float(np.mean((P - y) ** 2))
            losses.append(loss)

            dP = 2 * (P - y) / n
            dW2 = A1.T @ dP
            db2 = dP.sum(axis=0)
            dA1 = dP @ self.W2.T
            dZ1 = dA1 * (Z1 > 0)
            dW1 = Xs.T @ dZ1
            db1 = dZ1.sum(axis=0)

            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            self.W2 -= lr * dW2
            self.b2 -= lr * db2

        return losses

    def predict(self, Xs: np.ndarray) -> np.ndarray:
        return self.forward(Xs)


def save_model(path: str, model: ValueMLP, mean: np.ndarray, std: np.ndarray) -> None:
    data = {
        "W1": model.W1.tolist(),
        "b1": model.b1.tolist(),
        "W2": model.W2.tolist(),
        "b2": model.b2.tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "input_dim": model.input_dim,
        "hidden_dim": model.hidden_dim,
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file)


def load_model(path: str) -> tuple[ValueMLP, np.ndarray, np.ndarray]:
    with open(path, encoding="utf-8") as file:
        data = json.load(file)

    model = ValueMLP(input_dim=data["input_dim"], hidden_dim=data["hidden_dim"])
    model.W1 = np.array(data["W1"], dtype=float)
    model.b1 = np.array(data["b1"], dtype=float)
    model.W2 = np.array(data["W2"], dtype=float)
    model.b2 = np.array(data["b2"], dtype=float)
    mean = np.array(data["mean"], dtype=float)
    std = np.array(data["std"], dtype=float)
    return model, mean, std


def make_value_fn(
    model: ValueMLP,
    mean: np.ndarray,
    std: np.ndarray,
) -> Callable[[list[float]], float]:
    def value_fn(features: list[float]) -> float:
        X = np.array([features], dtype=float)
        Xs = apply_standardizer(X, mean, std)
        return float(model.predict(Xs)[0, 0])

    return value_fn
