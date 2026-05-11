#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Weighted average ensemble for physics submission CSV files.")
    p.add_argument("--inputs", type=Path, nargs="+", required=True, help="Submission CSV files to average.")
    p.add_argument("--weights", type=float, nargs="*", default=None, help="Optional weights matching --inputs.")
    p.add_argument("--output", type=Path, required=True, help="Output ensembled submission CSV.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.weights is None or len(args.weights) == 0:
        weights = np.ones(len(args.inputs), dtype=np.float64)
    else:
        if len(args.weights) != len(args.inputs):
            raise ValueError("--weights length must match --inputs length")
        weights = np.asarray(args.weights, dtype=np.float64)

    base = None
    weighted = None
    weight_sum = float(weights.sum())

    for path, weight in zip(args.inputs, weights):
        df = pd.read_csv(path, encoding="utf-8-sig")
        if base is None:
            base = df[["id"]].copy()
            weighted = np.zeros((len(df), 2), dtype=np.float64)
        else:
            if not np.array_equal(base["id"].values, df["id"].values):
                raise RuntimeError(f"id order mismatch: {path}")
        probs = df[["unstable_prob", "stable_prob"]].to_numpy(dtype=np.float64)
        weighted += weight * probs

    assert base is not None
    assert weighted is not None
    probs = weighted / max(weight_sum, 1e-12)
    probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
    probs = probs / probs.sum(axis=1, keepdims=True)

    out = base.copy()
    out["unstable_prob"] = probs[:, 0]
    out["stable_prob"] = probs[:, 1]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"rows={len(out)} -> {args.output}")


if __name__ == "__main__":
    main()
