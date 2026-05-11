#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from train_block_graph_baseline import (
    apply_top_roi_crop,
    build_tables,
    component_summary,
    estimate_fg_mask,
    extract_top_components,
    global_mask_stats,
    maybe_top_normalizer,
    probability_temperature,
    safe_log_loss,
    safe_auc,
    fit_temperature,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Support-graph structural baseline.")
    sub = p.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train")
    train.add_argument("--data-root", type=Path, default=Path("/data/AskFake/Image/physics"))
    train.add_argument("--out-dir", type=Path, required=True)
    train.add_argument("--n-folds", type=int, default=5)
    train.add_argument("--n-clusters", type=int, default=16)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--use-top-normalize", action="store_true")
    train.add_argument("--top-roi-margin-frac", type=float, default=0.15)
    train.add_argument("--top-color-bin", type=int, default=24)
    train.add_argument("--top-min-comp-frac", type=float, default=0.0012)
    train.add_argument("--front-band-thresh", type=float, default=0.09)

    predict = sub.add_parser("predict")
    predict.add_argument("--data-root", type=Path, default=Path("/data/AskFake/Image/physics"))
    predict.add_argument("--run-dir", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def smooth_1d(x: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return x.astype(np.float32, copy=True)
    kernel = np.ones(2 * radius + 1, dtype=np.float32)
    kernel /= kernel.sum()
    return np.convolve(x.astype(np.float32), kernel, mode="same")


def merge_top_components(components: list[dict[str, float]]) -> list[dict[str, float]]:
    if len(components) <= 1:
        return components
    comps = [dict(c) for c in components]
    changed = True
    while changed:
        changed = False
        merged: list[dict[str, float]] = []
        used = [False] * len(comps)
        for i, a in enumerate(comps):
            if used[i]:
                continue
            area = a["area_frac"]
            cx = a["cx"] * area
            cy = a["cy"] * area
            bw = a["bbox_w_frac"]
            bh = a["bbox_h_frac"]
            total = area
            used[i] = True
            for j in range(i + 1, len(comps)):
                if used[j]:
                    continue
                b = comps[j]
                dx = abs(a["cx"] - b["cx"])
                dy = abs(a["cy"] - b["cy"])
                x_touch = dx <= (0.35 * (a["bbox_w_frac"] + b["bbox_w_frac"]) + 0.04)
                y_touch = dy <= (0.35 * (a["bbox_h_frac"] + b["bbox_h_frac"]) + 0.04)
                if x_touch and y_touch:
                    area_b = b["area_frac"]
                    cx += b["cx"] * area_b
                    cy += b["cy"] * area_b
                    bw = max(bw, b["bbox_w_frac"])
                    bh = max(bh, b["bbox_h_frac"])
                    total += area_b
                    used[j] = True
                    changed = True
            merged.append(
                {
                    "area_frac": float(total),
                    "cx": float(cx / max(total, 1e-6)),
                    "cy": float(cy / max(total, 1e-6)),
                    "bbox_w_frac": float(bw),
                    "bbox_h_frac": float(bh),
                }
            )
        comps = sorted(merged, key=lambda x: x["area_frac"], reverse=True)
    return comps[:12]


def build_front_band(mask: np.ndarray, y1: int, y2: int) -> dict[str, float] | None:
    h, w = mask.shape
    band = mask[max(0, y1): min(h, y2 + 1), :]
    ys, xs = np.where(band > 0)
    if len(xs) == 0:
        return None
    xs = xs.astype(np.int32)
    x1, x2 = int(xs.min()), int(xs.max())
    yy_center = 0.5 * (max(0, y1) + min(h - 1, y2))
    return {
        "area_frac": float(len(xs) / max(h * w, 1)),
        "cx": float((xs.mean() / max(w - 1, 1) - 0.5) * 2.0),
        "cy": float((yy_center / max(h - 1, 1) - 0.5) * 2.0),
        "bbox_w_frac": float((x2 - x1 + 1) / max(w, 1)),
        "bbox_h_frac": float((min(h - 1, y2) - max(0, y1) + 1) / max(h, 1)),
        "x1_frac": float(x1 / max(w - 1, 1)),
        "x2_frac": float(x2 / max(w - 1, 1)),
    }


def extract_front_layers(front_rgb: np.ndarray, band_thresh: float) -> tuple[np.ndarray, list[dict[str, float]]]:
    mask = estimate_fg_mask(front_rgb, view="front")
    h, _w = mask.shape
    row_fill = smooth_1d(mask.mean(axis=1), radius=max(2, h // 80))
    thresh = max(band_thresh, float(np.percentile(row_fill, 60.0)))
    active = row_fill > thresh
    if not np.any(active):
        return mask, []

    peak_rows: list[int] = []
    min_gap = max(6, int(round(h * 0.06)))
    for y in range(1, h - 1):
        if not active[y]:
            continue
        if row_fill[y] >= row_fill[y - 1] and row_fill[y] >= row_fill[y + 1]:
            if row_fill[y] < thresh * 1.08:
                continue
            if peak_rows and y - peak_rows[-1] < min_gap:
                if row_fill[y] > row_fill[peak_rows[-1]]:
                    peak_rows[-1] = y
            else:
                peak_rows.append(y)

    if not peak_rows:
        ys = np.where(active)[0]
        band = build_front_band(mask, int(ys.min()), int(ys.max()))
        return mask, [band] if band is not None else []

    boundaries = [int(np.where(active)[0].min())]
    for a, b in zip(peak_rows[:-1], peak_rows[1:]):
        boundaries.append(int(round(0.5 * (a + b))))
    boundaries.append(int(np.where(active)[0].max()))

    bands: list[dict[str, float]] = []
    for idx, peak in enumerate(peak_rows):
        y1 = boundaries[idx]
        y2 = boundaries[idx + 1]
        if idx > 0:
            y1 += 1
        band = build_front_band(mask, y1, y2)
        if band is not None and band["bbox_h_frac"] > 0.015:
            bands.append(band)

    if len(bands) <= 1:
        ys = np.where(active)[0]
        band = build_front_band(mask, int(ys.min()), int(ys.max()))
        return mask, [band] if band is not None else []
    return mask, bands


def layer_profile_from_mask(mask: np.ndarray, bands: list[dict[str, float]]) -> dict[str, float]:
    h, _w = mask.shape
    row_fill = mask.mean(axis=1)
    if row_fill.max() <= 0:
        return {
            "front_rowfill_max": 0.0,
            "front_rowfill_mean": 0.0,
            "front_active_rows": 0.0,
            "front_profile_top_bias": 0.0,
            "front_profile_bottom_bias": 0.0,
            "front_layer_count": 0.0,
            "front_layer_center_drift": 0.0,
            "front_layer_width_decay": 0.0,
            "front_support_continuity": 0.0,
            "front_monotonic_shift_ratio": 0.0,
            "front_signed_drift_sum": 0.0,
            "front_max_single_layer_shift": 0.0,
        }
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    weights = row_fill / max(row_fill.sum(), 1e-6)
    top_bias = float(np.sum(np.maximum(-yy, 0.0) * weights))
    bottom_bias = float(np.sum(np.maximum(yy, 0.0) * weights))

    layer_count = float(len(bands))
    if len(bands) >= 2:
        order = sorted(bands, key=lambda b: b["cy"])
        centers = np.asarray([b["cx"] for b in order], dtype=np.float32)
        widths = np.asarray([b["bbox_w_frac"] for b in order], dtype=np.float32)
        center_drift = float(np.abs(np.diff(centers)).sum())
        signed_drift = np.diff(centers)
        width_decay = float(np.maximum(widths[:-1] - widths[1:], 0.0).sum())
        support_cont = float(np.mean(np.abs(np.diff(centers)) < (0.5 * (widths[:-1] + widths[1:]))))
        drift_sign = np.sign(signed_drift)
        monotonic_shift = float(np.mean(drift_sign == drift_sign[0])) if len(drift_sign) > 0 and drift_sign[0] != 0 else 0.0
        signed_drift_sum = float(signed_drift.sum())
        max_single_shift = float(np.max(np.abs(signed_drift)))
    else:
        center_drift = 0.0
        width_decay = 0.0
        support_cont = 0.0
        monotonic_shift = 0.0
        signed_drift_sum = 0.0
        max_single_shift = 0.0
    return {
        "front_rowfill_max": float(row_fill.max()),
        "front_rowfill_mean": float(row_fill.mean()),
        "front_active_rows": float((row_fill > np.percentile(row_fill, 70.0)).mean()),
        "front_profile_top_bias": top_bias,
        "front_profile_bottom_bias": bottom_bias,
        "front_layer_count": layer_count,
        "front_layer_center_drift": center_drift,
        "front_layer_width_decay": width_decay,
        "front_support_continuity": support_cont,
        "front_monotonic_shift_ratio": monotonic_shift,
        "front_signed_drift_sum": signed_drift_sum,
        "front_max_single_layer_shift": max_single_shift,
    }


def support_graph_features(top_components: list[dict[str, float]], front_bands: list[dict[str, float]]) -> dict[str, float]:
    feats: dict[str, float] = {}
    top_n = len(top_components)
    front_n = len(front_bands)
    feats["graph_top_nodes"] = float(top_n)
    feats["graph_front_nodes"] = float(front_n)

    if top_n == 0:
        feats["graph_top_mass_eccentricity"] = 0.0
        feats["graph_top_max_overhang"] = 0.0
        feats["graph_top_pair_support_gap"] = 0.0
    else:
        areas = np.asarray([c["area_frac"] for c in top_components], dtype=np.float32)
        cxs = np.asarray([c["cx"] for c in top_components], dtype=np.float32)
        widths = np.asarray([c["bbox_w_frac"] for c in top_components], dtype=np.float32)
        mass_center = float(np.average(cxs, weights=np.maximum(areas, 1e-6)))
        feats["graph_top_mass_eccentricity"] = abs(mass_center)
        feats["graph_top_max_overhang"] = float(np.max(np.abs(cxs) + 0.5 * widths))
        if top_n >= 2:
            order = np.argsort(-areas)
            a = int(order[0])
            b = int(order[1])
            feats["graph_top_pair_support_gap"] = float(abs(cxs[a] - cxs[b]) - 0.5 * (widths[a] + widths[b]))
            feats["graph_top_largest12_dx"] = float(abs(cxs[a] - cxs[b]))
        else:
            feats["graph_top_pair_support_gap"] = 0.0
            feats["graph_top_largest12_dx"] = 0.0

    if front_n == 0:
        feats["graph_front_stack_offset"] = 0.0
        feats["graph_front_top_heavy"] = 0.0
        feats["graph_front_base_deficit"] = 0.0
        feats["graph_front_overlap_mean"] = 0.0
        feats["graph_front_overlap_min"] = 0.0
        feats["graph_front_overhang_mean"] = 0.0
        feats["graph_front_center_outside_mean"] = 0.0
    else:
        order = sorted(front_bands, key=lambda b: b["cy"])
        centers = np.asarray([b["cx"] for b in order], dtype=np.float32)
        widths = np.asarray([b["bbox_w_frac"] for b in order], dtype=np.float32)
        areas = np.asarray([b["area_frac"] for b in order], dtype=np.float32)
        feats["graph_front_stack_offset"] = float(np.abs(centers - centers[0]).sum())
        feats["graph_front_top_heavy"] = float(areas[: max(1, len(areas) // 2)].sum() / max(areas.sum(), 1e-6))
        feats["graph_front_base_deficit"] = float(np.maximum(widths[0] - widths[-1], 0.0) if len(widths) >= 2 else 0.0)
        overlaps = []
        overhangs = []
        center_outsides = []
        for low, high in zip(order[:-1], order[1:]):
            l1, r1 = low["x1_frac"], low["x2_frac"]
            l2, r2 = high["x1_frac"], high["x2_frac"]
            inter = max(0.0, min(r1, r2) - max(l1, l2))
            base_w = max(r1 - l1, 1e-6)
            top_w = max(r2 - l2, 1e-6)
            overlap_ratio = inter / max(min(base_w, top_w), 1e-6)
            left_overhang = max(0.0, l1 - l2)
            right_overhang = max(0.0, r2 - r1)
            overhang = left_overhang + right_overhang
            center01 = 0.5 * (high["cx"] + 1.0)
            outside = 0.0
            if center01 < l1:
                outside = l1 - center01
            elif center01 > r1:
                outside = center01 - r1
            overlaps.append(overlap_ratio)
            overhangs.append(overhang)
            center_outsides.append(outside)
        feats["graph_front_overlap_mean"] = float(np.mean(overlaps)) if overlaps else 0.0
        feats["graph_front_overlap_min"] = float(np.min(overlaps)) if overlaps else 0.0
        feats["graph_front_overhang_mean"] = float(np.mean(overhangs)) if overhangs else 0.0
        feats["graph_front_center_outside_mean"] = float(np.mean(center_outsides)) if center_outsides else 0.0

    feats["graph_support_margin_proxy"] = (
        1.0
        - feats["graph_top_mass_eccentricity"]
        - 0.35 * max(feats["graph_top_pair_support_gap"], 0.0)
        - 0.40 * feats["graph_front_stack_offset"]
        + 0.45 * feats["graph_front_base_deficit"]
    )
    feats["graph_overturning_proxy"] = (
        0.90 * feats["graph_top_mass_eccentricity"]
        + 0.75 * feats["graph_front_overhang_mean"]
        + 0.70 * feats["graph_front_center_outside_mean"]
        + 0.55 * feats["graph_front_stack_offset"]
        - 0.50 * feats["graph_front_overlap_mean"]
    )
    return feats


def extract_feature_row(
    front_img: Image.Image,
    top_img: Image.Image,
    top_color_bin: int,
    top_min_comp_frac: float,
    front_band_thresh: float,
    top_roi_margin_frac: float,
) -> dict[str, float]:
    top_roi = apply_top_roi_crop(top_img, top_roi_margin_frac)
    top_rgb = np.asarray(top_roi.convert("RGB"), dtype=np.uint8)
    front_rgb = np.asarray(front_img.convert("RGB"), dtype=np.uint8)

    top_mask, top_components = extract_top_components(top_rgb, color_bin=top_color_bin, min_comp_frac=top_min_comp_frac)
    top_components = merge_top_components(top_components)
    front_mask, front_bands = extract_front_layers(front_rgb, band_thresh=front_band_thresh)
    top_stats = global_mask_stats(top_mask)
    front_stats = global_mask_stats(front_mask)
    layer_stats = layer_profile_from_mask(front_mask, front_bands)
    graph_stats = support_graph_features(top_components, front_bands)

    feats: dict[str, float] = {}
    feats.update({f"top_global_{k}": v for k, v in top_stats.items()})
    feats.update({f"front_global_{k}": v for k, v in front_stats.items()})
    feats.update(component_summary(top_components, "top_blocks"))
    feats.update(component_summary(front_bands, "front_bands"))
    feats.update(layer_stats)
    feats.update(graph_stats)
    feats["top_front_area_ratio"] = feats["top_global_area_frac"] / max(feats["front_global_area_frac"], 1e-6)
    feats["top_front_cx_gap"] = feats["top_global_cx"] - feats["front_global_cx"]
    feats["support_consistency"] = (
        1.1 * (1.0 - abs(feats["top_global_cx"]))
        + 0.7 * feats["front_support_continuity"]
        + 0.5 * feats["graph_front_base_deficit"]
        - 0.8 * feats["graph_top_mass_eccentricity"]
        - 0.6 * feats["front_layer_center_drift"]
    )
    feats["explicit_stability_score"] = (
        1.20 * feats["graph_front_overlap_mean"]
        + 0.80 * (1.0 - feats["graph_top_mass_eccentricity"])
        + 0.60 * feats["graph_front_base_deficit"]
        - 0.90 * feats["graph_front_overhang_mean"]
        - 0.80 * feats["graph_front_center_outside_mean"]
        - 0.50 * abs(feats["front_signed_drift_sum"])
    )
    return feats


def build_feature_table(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    top_norm = maybe_top_normalizer(args.use_top_normalize)
    rows = []
    total = len(df)
    for idx, row in enumerate(df.itertuples(index=False), start=1):
        front_img = Image.open(row.front_path).convert("RGB")
        top_img = Image.open(row.top_path).convert("RGB")
        if top_norm is not None:
            top_img = top_norm.normalize(row.top_path, top_img)
        feats = extract_feature_row(
            front_img=front_img,
            top_img=top_img,
            top_color_bin=args.top_color_bin,
            top_min_comp_frac=args.top_min_comp_frac,
            front_band_thresh=args.front_band_thresh,
            top_roi_margin_frac=args.top_roi_margin_frac,
        )
        feats["id"] = row.id
        rows.append(feats)
        if idx == 1 or idx % 100 == 0 or idx == total:
            print(json.dumps({"stage": "feature_table", "done": idx, "total": total, "last_id": str(row.id)}), flush=True)
    return pd.DataFrame(rows)


def add_clusters(df: pd.DataFrame, path_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    top_norm = maybe_top_normalizer(args.use_top_normalize)
    vecs = []
    total = len(path_df)
    for idx, row in enumerate(path_df.itertuples(index=False), start=1):
        front_img = Image.open(row.front_path).convert("RGB")
        top_img = Image.open(row.top_path).convert("RGB")
        if top_norm is not None:
            top_img = top_norm.normalize(row.top_path, top_img)
        top_roi = apply_top_roi_crop(top_img, args.top_roi_margin_frac)
        front_mask = estimate_fg_mask(np.asarray(front_img, dtype=np.uint8), view="front")
        top_mask = estimate_fg_mask(np.asarray(top_roi, dtype=np.uint8), view="top")
        front_small = Image.fromarray((front_mask * 255).astype(np.uint8)).resize((24, 24))
        top_small = Image.fromarray((top_mask * 255).astype(np.uint8)).resize((24, 24))
        vecs.append(
            np.concatenate(
                [
                    np.asarray(front_small, dtype=np.float32).reshape(-1) / 255.0,
                    np.asarray(top_small, dtype=np.float32).reshape(-1) / 255.0,
                ]
            )
        )
        if idx == 1 or idx % 100 == 0 or idx == total:
            print(json.dumps({"stage": "cluster_vectors", "done": idx, "total": total, "last_id": str(row.id)}), flush=True)
    X = np.stack(vecs, axis=0)
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=args.n_clusters, random_state=args.seed, n_init=20)
    out = df.copy()
    out["geometry_cluster"] = km.fit_predict(Xs)
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"id", "target", "front_path", "top_path", "geometry_cluster", "label"}
    return [c for c in df.columns if c not in exclude]


def sanitize_feature_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    out[cols] = out[cols].replace([np.inf, -np.inf], np.nan)
    out[cols] = out[cols].fillna(0.0)
    return out


def train_cv(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pooled_paths, test_paths = build_tables(args.data_root)

    pooled_feat = build_feature_table(pooled_paths, args)
    pooled = pooled_paths[["id", "target", "front_path", "top_path"]].merge(pooled_feat, on="id", how="left")
    pooled = add_clusters(pooled, pooled_paths[["id", "front_path", "top_path"]], args)

    test_feat = build_feature_table(test_paths, args)
    test_df = test_paths[["id", "front_path", "top_path"]].merge(test_feat, on="id", how="left")

    pooled.to_csv(args.out_dir / "pooled_features.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(args.out_dir / "test_features.csv", index=False, encoding="utf-8-sig")

    feats = feature_columns(pooled)
    pooled = sanitize_feature_frame(pooled, feats)
    test_df = sanitize_feature_frame(test_df, feats)
    X = pooled[feats].to_numpy(dtype=np.float32)
    y = pooled["target"].to_numpy(dtype=np.int64)
    groups = pooled["geometry_cluster"].to_numpy(dtype=np.int64)

    splitter = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    oof = np.zeros(len(pooled), dtype=np.float64)
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(X, y, groups=groups)):
        fold_dir = args.out_dir / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        scaler = StandardScaler().fit(X_tr)
        X_tr_scaled = scaler.transform(X_tr)
        X_va_scaled = scaler.transform(X_va)

        logreg = LogisticRegression(C=1.5, max_iter=4000, random_state=args.seed + fold)
        rf = RandomForestClassifier(
            n_estimators=600,
            max_depth=9,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=args.seed + fold,
            n_jobs=-1,
        )
        hgb = HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_depth=4,
            max_iter=600,
            l2_regularization=5e-3,
            min_samples_leaf=16,
            random_state=args.seed + fold,
        )

        logreg.fit(X_tr_scaled, y_tr)
        rf.fit(X_tr, y_tr)
        hgb.fit(X_tr, y_tr)

        p_lr = logreg.predict_proba(X_va_scaled)[:, 1]
        p_rf = rf.predict_proba(X_va)[:, 1]
        p_hgb = hgb.predict_proba(X_va)[:, 1]
        p_blend = 0.18 * p_lr + 0.32 * p_rf + 0.50 * p_hgb
        temp = fit_temperature(p_blend, y_va)
        p_cal = probability_temperature(p_blend, temp)
        oof[va_idx] = p_cal

        ll = safe_log_loss(y_va, p_cal)
        auc = safe_auc(y_va, p_cal)
        fold_scores.append({"fold": fold, "val_logloss": float(ll), "val_auc": float(auc), "temperature": float(temp)})
        print(json.dumps(fold_scores[-1]))

        with (fold_dir / "artifacts.pkl").open("wb") as f:
            pickle.dump(
                {
                    "features": feats,
                    "scaler": scaler,
                    "logreg": logreg,
                    "rf": rf,
                    "hgb": hgb,
                    "temperature": temp,
                },
                f,
            )

    overall = {"oof_logloss": safe_log_loss(y, oof), "oof_auc": safe_auc(y, oof)}
    print(json.dumps(overall))
    with (args.out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2, default=str)
    with (args.out_dir / "scores.json").open("w", encoding="utf-8") as f:
        json.dump({"folds": fold_scores, "overall": overall}, f, ensure_ascii=False, indent=2)


def predict(args: argparse.Namespace) -> None:
    _config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    test_df = pd.read_csv(args.run_dir / "test_features.csv", encoding="utf-8-sig")
    fold_dirs = sorted([p for p in args.run_dir.iterdir() if p.is_dir() and p.name.startswith("fold")])
    preds = []
    for fold_dir in fold_dirs:
        with (fold_dir / "artifacts.pkl").open("rb") as f:
            art = pickle.load(f)
        test_df = sanitize_feature_frame(test_df, art["features"])
        X = test_df[art["features"]].to_numpy(dtype=np.float32)
        X_scaled = art["scaler"].transform(X)
        p_lr = art["logreg"].predict_proba(X_scaled)[:, 1]
        p_rf = art["rf"].predict_proba(X)[:, 1]
        p_hgb = art["hgb"].predict_proba(X)[:, 1]
        p_blend = 0.18 * p_lr + 0.32 * p_rf + 0.50 * p_hgb
        preds.append(probability_temperature(p_blend, float(art["temperature"])))

    final = np.mean(np.stack(preds, axis=0), axis=0)
    final = np.clip(final, 1e-7, 1 - 1e-7)
    out = test_df[["id"]].copy()
    out["unstable_prob"] = final
    out["stable_prob"] = 1.0 - final
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"rows={len(out)} -> {args.output}")


def main() -> None:
    args = parse_args()
    if args.command == "train":
        train_cv(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
