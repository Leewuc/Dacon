#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

try:
    from checkerboard_rectification import CheckerboardTopNormConfig, CheckerboardTopNormalizer
except Exception:
    CheckerboardTopNormConfig = None
    CheckerboardTopNormalizer = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Block-graph baseline for physics competition.")
    sub = p.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train")
    train.add_argument("--data-root", type=Path, default=Path("/data/AskFake/Image/physics"))
    train.add_argument("--out-dir", type=Path, required=True)
    train.add_argument("--n-folds", type=int, default=5)
    train.add_argument("--n-clusters", type=int, default=16)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--use-top-normalize", action="store_true")
    train.add_argument("--top-roi-margin-frac", type=float, default=0.15)
    train.add_argument("--top-color-bin", type=int, default=32)
    train.add_argument("--top-min-comp-frac", type=float, default=0.0015)
    train.add_argument("--front-band-thresh", type=float, default=0.10)

    predict = sub.add_parser("predict")
    predict.add_argument("--data-root", type=Path, default=Path("/data/AskFake/Image/physics"))
    predict.add_argument("--run-dir", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def read_csv_sig(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def gray_from_rgb(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def sat_from_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32) / 255.0
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    return np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)


def safe_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    h, w = mask.shape
    if len(xs) == 0:
        return 0, 0, w - 1, h - 1
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def pil_center_crop(img: Image.Image, view: str) -> Image.Image:
    w, h = img.size
    if view == "front":
        box = (int(0.25 * w), int(0.20 * h), int(0.75 * w), int(0.88 * h))
    else:
        box = (int(0.29 * w), int(0.29 * h), int(0.71 * w), int(0.71 * h))
    return img.crop(box)


def iter_components(mask: np.ndarray):
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    ys, xs = np.where(mask > 0)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if visited[y0, x0]:
            continue
        stack = [(y0, x0)]
        comp = []
        visited[y0, x0] = True
        while stack:
            y, x = stack.pop()
            comp.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and mask[ny, nx] > 0:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        yield comp


def remove_small_components(mask: np.ndarray, min_area_frac: float = 0.0025) -> np.ndarray:
    h, w = mask.shape
    min_area = max(8, int(round(h * w * min_area_frac)))
    out = np.zeros_like(mask, dtype=np.uint8)
    for comp in iter_components(mask):
        if len(comp) >= min_area:
            for y, x in comp:
                out[y, x] = 1
    return out


def suppress_bottom_wide_band(mask: np.ndarray, start_thresh: float = 0.35, keep_thresh: float = 0.22) -> np.ndarray:
    h, _w = mask.shape
    row_fill = mask.mean(axis=1)
    if row_fill[-1] < start_thresh:
        return mask
    cut = h
    seen_dense = False
    for y in range(h - 1, -1, -1):
        if row_fill[y] >= start_thresh:
            seen_dense = True
            cut = y
            continue
        if seen_dense and row_fill[y] >= keep_thresh:
            cut = y
            continue
        if seen_dense:
            break
    out = mask.copy()
    if cut < h:
        out[cut:, :] = 0
    return out


def suppress_front_grid_components(mask: np.ndarray, bg_dist: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=np.uint8)
    for comp in iter_components(mask):
        ys = np.array([y for y, _ in comp], dtype=np.int32)
        xs = np.array([x for _, x in comp], dtype=np.int32)
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        comp_w = x2 - x1 + 1
        comp_h = y2 - y1 + 1
        touches_bottom = y2 >= h - 2
        touches_left = x1 <= 1
        touches_right = x2 >= w - 2
        width_frac = comp_w / max(w, 1)
        height_frac = comp_h / max(h, 1)
        bottom_frac = ys.mean() / max(h - 1, 1)
        mean_bg = float(bg_dist[ys, xs].mean())
        is_floor_grid = (
            touches_bottom
            and (
                width_frac > 0.40
                or (touches_left and touches_right)
                or (bottom_frac > 0.78 and width_frac > 0.28)
            )
            and height_frac < 0.42
            and mean_bg < float(np.percentile(bg_dist, 82.0))
        )
        if is_floor_grid:
            continue
        for y, x in comp:
            out[y, x] = 1
    return out


def suppress_blue_grid_components(mask: np.ndarray, rgbf: np.ndarray, sat: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    border_mask = np.zeros((h, w), dtype=bool)
    bw = max(2, int(round(min(h, w) * 0.06)))
    border_mask[:bw, :] = True
    border_mask[-bw:, :] = True
    border_mask[:, :bw] = True
    border_mask[:, -bw:] = True
    blueish = (
        (rgbf[:, :, 2] > rgbf[:, :, 1] + 0.05)
        & (rgbf[:, :, 2] > rgbf[:, :, 0] + 0.05)
        & (sat > np.percentile(sat, 55.0))
    )
    blue_border = rgbf[border_mask & blueish]
    if len(blue_border) < 20:
        return mask
    blue_proto = np.median(blue_border, axis=0)
    out = np.zeros_like(mask, dtype=np.uint8)
    for comp in iter_components(mask):
        ys = np.array([y for y, _ in comp], dtype=np.int32)
        xs = np.array([x for _, x in comp], dtype=np.int32)
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        comp_rgb = rgbf[ys, xs]
        mean_rgb = comp_rgb.mean(axis=0)
        color_dist = float(np.sqrt(((mean_rgb - blue_proto) ** 2).sum()))
        touches_border = x1 <= 1 or y1 <= 1 or x2 >= w - 2 or y2 >= h - 2
        width_frac = (x2 - x1 + 1) / max(w, 1)
        height_frac = (y2 - y1 + 1) / max(h, 1)
        mean_sat = float(sat[ys, xs].mean())
        is_blue_grid = (
            touches_border
            and color_dist < 0.16
            and mean_sat > float(np.percentile(sat, 58.0))
            and width_frac < 0.32
            and height_frac < 0.32
        )
        if is_blue_grid:
            continue
        for y, x in comp:
            out[y, x] = 1
    return out


def estimate_roi_mask(rgb: np.ndarray, view: str) -> np.ndarray:
    gray = gray_from_rgb(rgb)
    sat = sat_from_rgb(rgb)
    val = rgb.max(axis=2).astype(np.float32) / 255.0
    rgbf = rgb.astype(np.float32) / 255.0
    h, w = gray.shape
    border = np.concatenate([rgbf[0], rgbf[-1], rgbf[:, 0], rgbf[:, -1]], axis=0)
    bg_proto = np.median(border, axis=0)
    bg_dist = np.sqrt(((rgbf - bg_proto[None, None, :]) ** 2).sum(axis=2))

    s_thr = float(np.percentile(sat, 60.0))
    g_thr = float(np.percentile(gray, 45.0))
    v_thr = float(np.percentile(val, 35.0))
    d_thr = float(np.percentile(bg_dist, 75.0))
    mask = ((sat > s_thr) | (gray < g_thr) | (val < v_thr) | (bg_dist > d_thr)).astype(np.uint8)
    if mask.mean() < 0.01:
        mask = ((sat > np.percentile(sat, 50.0)) | (bg_dist > np.percentile(bg_dist, 65.0))).astype(np.uint8)

    yy, xx = np.mgrid[0:h, 0:w]
    near_border = (xx < int(0.14 * w)) | (xx > int(0.86 * w)) | (yy < int(0.14 * h)) | (yy > int(0.86 * h))
    bright_blob = (val > np.percentile(val, 97.0)) & (sat < np.percentile(sat, 55.0))
    mask[near_border & bright_blob] = 0
    if view == "front":
        top_band = int(0.12 * h)
        mask[:top_band, :] = 0
        mask = suppress_bottom_wide_band(mask)
        mask = suppress_front_grid_components(mask, bg_dist)
    mask = suppress_blue_grid_components(mask, rgbf, sat)
    return remove_small_components(mask, min_area_frac=0.0025)


def dilate_mask(mask: np.ndarray, radius: int = 5) -> np.ndarray:
    out = mask.copy().astype(np.uint8)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
            if dy > 0:
                shifted[:dy, :] = 0
            elif dy < 0:
                shifted[dy:, :] = 0
            if dx > 0:
                shifted[:, :dx] = 0
            elif dx < 0:
                shifted[:, dx:] = 0
            out = np.maximum(out, shifted)
    return out


def apply_top_roi_crop(img: Image.Image, margin_frac: float) -> Image.Image:
    rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    mask = dilate_mask(estimate_roi_mask(rgb, view="top"), radius=max(3, min(rgb.shape[:2]) // 50))
    x1, y1, x2, y2 = safe_bbox(mask)
    h, w = mask.shape
    bw = max(x2 - x1 + 1, 1)
    bh = max(y2 - y1 + 1, 1)
    mx = max(4, int(round(bw * margin_frac)))
    my = max(4, int(round(bh * margin_frac)))
    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(w - 1, x2 + mx)
    y2 = min(h - 1, y2 + my)
    crop = img.crop((x1, y1, x2 + 1, y2 + 1))
    side = max(crop.size)
    canvas = Image.new("RGB", (side, side), (127, 127, 127))
    ox = (side - crop.size[0]) // 2
    oy = (side - crop.size[1]) // 2
    canvas.paste(crop, (ox, oy))
    return canvas


def estimate_fg_mask(rgb: np.ndarray, view: str) -> np.ndarray:
    gray = gray_from_rgb(rgb)
    sat = sat_from_rgb(rgb)
    val = rgb.max(axis=2).astype(np.float32) / 255.0
    rgbf = rgb.astype(np.float32) / 255.0
    border = np.concatenate([rgbf[0], rgbf[-1], rgbf[:, 0], rgbf[:, -1]], axis=0)
    bg_proto = np.median(border, axis=0)
    bg_dist = np.sqrt(((rgbf - bg_proto[None, None, :]) ** 2).sum(axis=2))
    mask = (
        (sat > np.percentile(sat, 60.0))
        | (gray < np.percentile(gray, 45.0))
        | (val < np.percentile(val, 35.0))
        | (bg_dist > np.percentile(bg_dist, 75.0))
    ).astype(np.uint8)
    if view == "front":
        mask[: int(0.12 * mask.shape[0]), :] = 0
        mask = suppress_bottom_wide_band(mask)
    return remove_small_components(mask, min_area_frac=0.0025)


def maybe_top_normalizer(enabled: bool):
    if enabled and CheckerboardTopNormalizer is not None and CheckerboardTopNormConfig is not None:
        return CheckerboardTopNormalizer(CheckerboardTopNormConfig(enabled=True))
    return None


def component_summary(components: list[dict[str, float]], prefix: str) -> dict[str, float]:
    out: dict[str, float] = {}
    n = len(components)
    out[f"{prefix}_num_components"] = float(n)
    if n == 0:
        for name in [
            "area_mean", "area_std", "area_max", "cx_mean", "cy_mean", "cx_abs_max",
            "cy_abs_max", "bbox_w_mean", "bbox_h_mean", "aspect_mean", "pair_dist_mean",
            "pair_dist_max", "degree_mean",
        ]:
            out[f"{prefix}_{name}"] = 0.0
        return out
    areas = np.asarray([c["area_frac"] for c in components], dtype=np.float32)
    cxs = np.asarray([c["cx"] for c in components], dtype=np.float32)
    cys = np.asarray([c["cy"] for c in components], dtype=np.float32)
    bbox_ws = np.asarray([c["bbox_w_frac"] for c in components], dtype=np.float32)
    bbox_hs = np.asarray([c["bbox_h_frac"] for c in components], dtype=np.float32)
    aspects = bbox_ws / np.maximum(bbox_hs, 1e-6)
    out[f"{prefix}_area_mean"] = float(areas.mean())
    out[f"{prefix}_area_std"] = float(areas.std())
    out[f"{prefix}_area_max"] = float(areas.max())
    out[f"{prefix}_cx_mean"] = float(np.average(cxs, weights=np.maximum(areas, 1e-6)))
    out[f"{prefix}_cy_mean"] = float(np.average(cys, weights=np.maximum(areas, 1e-6)))
    out[f"{prefix}_cx_abs_max"] = float(np.max(np.abs(cxs)))
    out[f"{prefix}_cy_abs_max"] = float(np.max(np.abs(cys)))
    out[f"{prefix}_bbox_w_mean"] = float(bbox_ws.mean())
    out[f"{prefix}_bbox_h_mean"] = float(bbox_hs.mean())
    out[f"{prefix}_aspect_mean"] = float(aspects.mean())

    if n >= 2:
        pts = np.stack([cxs, cys], axis=1)
        d = np.sqrt(np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2))
        tri = d[np.triu_indices(n, k=1)]
        out[f"{prefix}_pair_dist_mean"] = float(tri.mean())
        out[f"{prefix}_pair_dist_max"] = float(tri.max())
        deg = (d < 0.35).sum(axis=1) - 1
        out[f"{prefix}_degree_mean"] = float(deg.mean())
    else:
        out[f"{prefix}_pair_dist_mean"] = 0.0
        out[f"{prefix}_pair_dist_max"] = 0.0
        out[f"{prefix}_degree_mean"] = 0.0
    return out


def extract_top_components(top_rgb: np.ndarray, color_bin: int, min_comp_frac: float) -> tuple[np.ndarray, list[dict[str, float]]]:
    mask = estimate_fg_mask(top_rgb, view="top")
    h, w = mask.shape
    if mask.sum() == 0:
        return mask, []
    quant = (top_rgb // color_bin).astype(np.int32)
    key = quant[:, :, 0] * 10000 + quant[:, :, 1] * 100 + quant[:, :, 2]
    comps: list[dict[str, float]] = []
    min_area = max(6, int(round(h * w * min_comp_frac)))
    for k in np.unique(key[mask > 0]):
        submask = ((key == int(k)) & (mask > 0)).astype(np.uint8)
        for comp in iter_components(submask):
            if len(comp) < min_area:
                continue
            ys = np.asarray([y for y, _ in comp], dtype=np.int32)
            xs = np.asarray([x for _, x in comp], dtype=np.int32)
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            area_frac = float(len(comp) / max(h * w, 1))
            cx = float((xs.mean() / max(w - 1, 1) - 0.5) * 2.0)
            cy = float((ys.mean() / max(h - 1, 1) - 0.5) * 2.0)
            comps.append(
                {
                    "area_frac": area_frac,
                    "cx": cx,
                    "cy": cy,
                    "bbox_w_frac": float((x2 - x1 + 1) / max(w, 1)),
                    "bbox_h_frac": float((y2 - y1 + 1) / max(h, 1)),
                }
            )
    comps.sort(key=lambda x: x["area_frac"], reverse=True)
    return mask, comps[:12]


def extract_front_bands(front_rgb: np.ndarray, band_thresh: float) -> tuple[np.ndarray, list[dict[str, float]]]:
    mask = estimate_fg_mask(front_rgb, view="front")
    h, w = mask.shape
    row_fill = mask.mean(axis=1)
    thresh = max(band_thresh, float(np.percentile(row_fill, 70.0)))
    active = row_fill > thresh
    bands: list[dict[str, float]] = []
    y = 0
    while y < h:
        if not active[y]:
            y += 1
            continue
        y1 = y
        while y < h and active[y]:
            y += 1
        y2 = y - 1
        band = mask[y1 : y2 + 1, :]
        ys, xs = np.where(band > 0)
        if len(xs) == 0:
            continue
        xs = xs.astype(np.int32)
        area_frac = float(len(xs) / max(h * w, 1))
        x1, x2 = int(xs.min()), int(xs.max())
        cx = float((xs.mean() / max(w - 1, 1) - 0.5) * 2.0)
        cy = float((((y1 + y2) * 0.5) / max(h - 1, 1) - 0.5) * 2.0)
        bands.append(
            {
                "area_frac": area_frac,
                "cx": cx,
                "cy": cy,
                "bbox_w_frac": float((x2 - x1 + 1) / max(w, 1)),
                "bbox_h_frac": float((y2 - y1 + 1) / max(h, 1)),
            }
        )
    return mask, bands


def global_mask_stats(mask: np.ndarray) -> dict[str, float]:
    ys, xs = np.where(mask > 0)
    h, w = mask.shape
    if len(xs) == 0:
        return {"area_frac": 0.0, "cx": 0.0, "cy": 0.0, "spread_x": 0.0, "spread_y": 0.0}
    x = xs.astype(np.float32) / max(w - 1, 1)
    y = ys.astype(np.float32) / max(h - 1, 1)
    return {
        "area_frac": float(mask.mean()),
        "cx": float((x.mean() - 0.5) * 2.0),
        "cy": float((y.mean() - 0.5) * 2.0),
        "spread_x": float(x.std()),
        "spread_y": float(y.std()),
    }


def extract_feature_row(
    front_img: Image.Image,
    top_img: Image.Image,
    top_color_bin: int,
    top_min_comp_frac: float,
    front_band_thresh: float,
    top_roi_margin_frac: float,
) -> dict[str, float]:
    front_rgb = np.asarray(front_img.convert("RGB"), dtype=np.uint8)
    top_roi = apply_top_roi_crop(top_img, top_roi_margin_frac)
    top_rgb = np.asarray(top_roi.convert("RGB"), dtype=np.uint8)

    top_mask, top_components = extract_top_components(top_rgb, color_bin=top_color_bin, min_comp_frac=top_min_comp_frac)
    front_mask, front_bands = extract_front_bands(front_rgb, band_thresh=front_band_thresh)
    top_stats = global_mask_stats(top_mask)
    front_stats = global_mask_stats(front_mask)

    feats: dict[str, float] = {}
    feats.update({f"top_global_{k}": v for k, v in top_stats.items()})
    feats.update({f"front_global_{k}": v for k, v in front_stats.items()})
    feats.update(component_summary(top_components, "top_blocks"))
    feats.update(component_summary(front_bands, "front_bands"))
    feats["block_band_count_gap"] = feats["top_blocks_num_components"] - feats["front_bands_num_components"]
    feats["top_front_area_ratio"] = feats["top_global_area_frac"] / max(feats["front_global_area_frac"], 1e-6)
    feats["top_front_cx_gap"] = feats["top_global_cx"] - feats["front_global_cx"]
    feats["top_front_spread_ratio"] = feats["top_global_spread_x"] / max(feats["front_global_spread_x"], 1e-6)
    feats["top_overhang_proxy"] = feats["top_blocks_cx_abs_max"] * feats["top_blocks_area_max"]
    feats["front_tower_proxy"] = feats["front_bands_bbox_h_mean"] / max(feats["front_bands_bbox_w_mean"], 1e-6)
    feats["stability_margin_proxy"] = (
        1.2 * (1.0 - abs(feats["top_global_cx"]))
        + 0.8 * feats["front_bands_bbox_w_mean"]
        - 0.6 * feats["front_tower_proxy"]
        - 0.5 * feats["top_overhang_proxy"]
    )
    return feats


def build_tables(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = read_csv_sig(data_root / "train.csv")
    dev_df = read_csv_sig(data_root / "dev.csv")
    test_df = read_csv_sig(data_root / "sample_submission.csv")

    train_df["front_path"] = train_df["id"].apply(lambda x: str(data_root / "train" / x / "front.png"))
    train_df["top_path"] = train_df["id"].apply(lambda x: str(data_root / "train" / x / "top.png"))
    dev_df["front_path"] = dev_df["id"].apply(lambda x: str(data_root / "dev" / x / "front.png"))
    dev_df["top_path"] = dev_df["id"].apply(lambda x: str(data_root / "dev" / x / "top.png"))
    test_df["front_path"] = test_df["id"].apply(lambda x: str(data_root / "test" / x / "front.png"))
    test_df["top_path"] = test_df["id"].apply(lambda x: str(data_root / "test" / x / "top.png"))

    train_df["target"] = (train_df["label"] == "unstable").astype(int)
    dev_df["target"] = (dev_df["label"] == "unstable").astype(int)
    pooled = pd.concat([train_df, dev_df], ignore_index=True)
    return pooled, test_df


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
        front_small = front_img.convert("L").resize((24, 24))
        top_small = top_roi.convert("L").resize((24, 24))
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


def probability_temperature(prob: np.ndarray, temperature: float) -> np.ndarray:
    prob = np.clip(prob, 1e-7, 1 - 1e-7)
    logit = np.log(prob / (1.0 - prob))
    scaled = logit / max(temperature, 1e-6)
    return 1.0 / (1.0 + np.exp(-np.clip(scaled, -60.0, 60.0)))


def safe_log_loss(y_true: np.ndarray, prob: np.ndarray) -> float:
    return float(log_loss(y_true, np.clip(prob, 1e-7, 1 - 1e-7), labels=[0, 1]))


def safe_auc(y_true: np.ndarray, prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, prob))


def fit_temperature(prob: np.ndarray, y_true: np.ndarray) -> float:
    best_t = 1.0
    best_loss = safe_log_loss(y_true, prob)
    for t in np.linspace(0.6, 2.5, 40):
        cal = probability_temperature(prob, float(t))
        ll = safe_log_loss(y_true, cal)
        if ll < best_loss:
            best_loss = ll
            best_t = float(t)
    return best_t


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

        logreg = LogisticRegression(C=2.0, max_iter=4000, random_state=args.seed + fold)
        rf = RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=args.seed + fold,
            n_jobs=-1,
        )
        hgb = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=4,
            max_iter=500,
            l2_regularization=1e-2,
            min_samples_leaf=18,
            random_state=args.seed + fold,
        )

        logreg.fit(X_tr_scaled, y_tr)
        rf.fit(X_tr, y_tr)
        hgb.fit(X_tr, y_tr)

        p_lr = logreg.predict_proba(X_va_scaled)[:, 1]
        p_rf = rf.predict_proba(X_va)[:, 1]
        p_hgb = hgb.predict_proba(X_va)[:, 1]
        p_blend = 0.15 * p_lr + 0.30 * p_rf + 0.55 * p_hgb
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
        X = test_df[art["features"]].to_numpy(dtype=np.float32)
        X_scaled = art["scaler"].transform(X)
        p_lr = art["logreg"].predict_proba(X_scaled)[:, 1]
        p_rf = art["rf"].predict_proba(X)[:, 1]
        p_hgb = art["hgb"].predict_proba(X)[:, 1]
        p_blend = 0.15 * p_lr + 0.30 * p_rf + 0.55 * p_hgb
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
