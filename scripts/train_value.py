#!/usr/bin/env python
from __future__ import annotations

import argparse

import numpy as np

from searchloop.traces import read_traces
from searchloop.value import (
    ValueMLP,
    apply_standardizer,
    fit_standardizer,
    save_model,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a small value head from trace JSONL.")
    parser.add_argument("--traces", default="traces.jsonl")
    parser.add_argument("--out", default="value_model.json")
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    records = read_traces(args.traces)
    if not records:
        raise SystemExit(f"no trace records found in {args.traces}")

    X = np.array([features for features, _ in records], dtype=float)
    y = np.array([[value] for _, value in records], dtype=float)

    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(records))
    n_val = int(len(records) * args.val_frac)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    if len(train_idx) == 0:
        raise SystemExit("training split is empty; reduce --val-frac or provide more traces")

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx] if len(val_idx) else X_train
    y_val = y[val_idx] if len(val_idx) else y_train

    mean, std = fit_standardizer(X_train)
    Xs_train = apply_standardizer(X_train, mean, std)
    Xs_val = apply_standardizer(X_val, mean, std)

    model = ValueMLP(input_dim=X.shape[1], hidden_dim=args.hidden, seed=args.seed)
    losses = model.train(Xs_train, y_train, epochs=args.epochs, lr=args.lr)
    train_mse = losses[-1]
    val_pred = model.predict(Xs_val)
    val_mse = float(np.mean((val_pred - y_val) ** 2))
    train_mean = float(y_train.mean())
    baseline_val_mse = float(np.mean((train_mean - y_val) ** 2))

    save_model(args.out, model, mean, std)
    print(f"train_mse={train_mse:.6f}")
    print(f"val_mse={val_mse:.6f}")
    print(f"baseline_val_mse={baseline_val_mse:.6f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
