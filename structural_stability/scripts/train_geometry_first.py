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
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

try:
    from checkerboard_rectification import CheckerboardTopNormConfig, CheckerboardTopNormalizer
except Exception:
    CheckerboardTopNormConfig = None
    CheckerboardTopNormalizer = None

try:
    from train_block_graph_baseline import extract_top_components
    from train_support_graph_baseline import merge_top_components
except Exception:
    extract_top_components = None
    merge_top_components = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Geometry-first CV pipeline for the physics competition.")
    sub = p.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train")
    train.add_argument("--data-root", type=Path, default=Path("/data/AskFake/Image/physics"))
    train.add_argument("--out-dir", type=Path, default=Path("/data/AskFake/Image/physics/baseline/runs/geometry_first_v1"))
    train.add_argument("--n-folds", type=int, default=5)
    train.add_argument("--n-clusters", type=int, default=16)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--use-top-normalize", action="store_true")
    train.add_argument(
        "--physics-theory",
        choices=["classic", "support_margin", "overturning", "hybrid", "paper_hybrid"],
        default="classic",
    )
    train.add_argument("--top-full-roi", action="store_true")
    train.add_argument("--top-roi-margin-frac", type=float, default=0.15)
    train.add_argument("--box-aware-features", action="store_true")
    train.add_argument("--adaptive-crop", action="store_true")
    train.add_argument("--crop-margin-frac", type=float, default=0.12)
    train.add_argument("--remove-shadow", action="store_true")

    predict = sub.add_parser("predict")
    predict.add_argument("--data-root", type=Path, default=Path("/data/AskFake/Image/physics"))
    predict.add_argument("--run-dir", type=Path, required=True)
    predict.add_argument("--output", type=Path, default=Path("/data/AskFake/Image/physics/baseline/submission_geometry_first_v1.csv"))
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def read_csv_sig(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def pil_center_crop(img: Image.Image, view: str) -> Image.Image:
    w, h = img.size
    if view == "front":
        box = (int(0.25 * w), int(0.20 * h), int(0.75 * w), int(0.88 * h))
    else:
        box = (int(0.29 * w), int(0.29 * h), int(0.71 * w), int(0.71 * h))
    return img.crop(box)


def gray_from_rgb(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def sat_from_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32) / 255.0
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    return np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)


def estimate_foreground_mask(rgb: np.ndarray, remove_shadow: bool = False) -> np.ndarray:
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
    d_thr = float(np.percentile(bg_dist, 70.0))
    mask = ((sat > s_thr) | (gray < g_thr) | (val < v_thr) | (bg_dist > d_thr)).astype(np.uint8)
    if remove_shadow:
        shadow = (val < np.percentile(val, 30.0)) & (sat < np.percentile(sat, 45.0))
        mask = np.where(shadow, 0, mask).astype(np.uint8)
    if mask.mean() < 0.01:
        mask = ((sat > np.percentile(sat, 50.0)) | (gray < np.percentile(gray, 50.0)) | (bg_dist > np.percentile(bg_dist, 60.0))).astype(np.uint8)
    return mask


def adaptive_bbox_from_rgb(rgb: np.ndarray, margin_frac: float, remove_shadow: bool = False) -> tuple[int, int, int, int]:
    mask = estimate_foreground_mask(rgb, remove_shadow=remove_shadow)
    x1, y1, x2, y2 = safe_bbox(mask)
    h, w = mask.shape
    bw = max(x2 - x1 + 1, 1)
    bh = max(y2 - y1 + 1, 1)
    mx = max(2, int(round(bw * margin_frac)))
    my = max(2, int(round(bh * margin_frac)))
    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(w - 1, x2 + mx)
    y2 = min(h - 1, y2 + my)
    return x1, y1, x2, y2


def pil_adaptive_crop(img: Image.Image, view: str, margin_frac: float, remove_shadow: bool = False) -> Image.Image:
    base = pil_center_crop(img, view)
    rgb = np.asarray(base.convert("RGB"), dtype=np.uint8)
    x1, y1, x2, y2 = adaptive_bbox_from_rgb(rgb, margin_frac, remove_shadow=remove_shadow)
    cropped = base.crop((x1, y1, x2 + 1, y2 + 1))
    side = max(cropped.size)
    canvas = Image.new("RGB", (side, side), (127, 127, 127))
    ox = (side - cropped.size[0]) // 2
    oy = (side - cropped.size[1]) // 2
    canvas.paste(cropped, (ox, oy))
    return canvas


def estimate_full_roi_mask_view(rgb: np.ndarray, view: str) -> np.ndarray:
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
    return mask


def dilate_binary_mask(mask: np.ndarray, radius: int = 5) -> np.ndarray:
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


def pil_top_roi_crop(img: Image.Image, margin_frac: float) -> Image.Image:
    rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    mask = estimate_full_roi_mask_view(rgb, view="top")
    mask = dilate_binary_mask(mask, radius=max(3, min(mask.shape) // 50))
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
    cropped = img.crop((x1, y1, x2 + 1, y2 + 1))
    side = max(cropped.size)
    canvas = Image.new("RGB", (side, side), (127, 127, 127))
    ox = (side - cropped.size[0]) // 2
    oy = (side - cropped.size[1]) // 2
    canvas.paste(cropped, (ox, oy))
    return canvas


def safe_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    h, w = mask.shape
    if len(xs) == 0:
        return 0, 0, w - 1, h - 1
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def band_width(mask: np.ndarray, y1: int, y2: int) -> float:
    band = mask[max(0, y1):max(y1 + 1, y2), :]
    ys, xs = np.where(band > 0)
    if len(xs) == 0:
        return 0.0
    return float(xs.max() - xs.min() + 1) / max(mask.shape[1], 1)


def centroid_stats(mask: np.ndarray) -> tuple[float, float, float, float]:
    ys, xs = np.where(mask > 0)
    h, w = mask.shape
    if len(xs) == 0:
        return 0.0, 0.0, 0.0, 0.0
    x = xs.astype(np.float32) / max(w - 1, 1)
    y = ys.astype(np.float32) / max(h - 1, 1)
    cx = float((x.mean() - 0.5) * 2.0)
    cy = float((y.mean() - 0.5) * 2.0)
    sx = float(x.std())
    sy = float(y.std())
    return cx, cy, sx, sy


def quadrant_occupancy(mask: np.ndarray) -> list[float]:
    h, w = mask.shape
    h2, w2 = h // 2, w // 2
    quads = [
        mask[:h2, :w2],
        mask[:h2, w2:],
        mask[h2:, :w2],
        mask[h2:, w2:],
    ]
    return [float(q.mean()) for q in quads]


def bbox_center_margin(cx_px: float, x1: int, x2: int, width: int) -> float:
    bbox_w = max(x2 - x1 + 1, 1)
    bbox_center = 0.5 * (x1 + x2)
    offset = abs(cx_px - bbox_center) / bbox_w
    return float(np.clip(0.5 - offset, -1.0, 0.5))


def bottom_contact_ratio(mask: np.ndarray, x1: int, x2: int, y2: int, band_frac: float = 0.08) -> float:
    h, _w = mask.shape
    band_h = max(1, int(round(h * band_frac)))
    band = mask[max(0, y2 - band_h + 1): y2 + 1, x1:x2 + 1]
    if band.size == 0:
        return 0.0
    return float(band.mean())


def collapse_margin(features: dict[str, float]) -> float:
    raw = (
        1.20 * features["top_support_width_frac"]
        + 0.90 * features["front_base_width_frac"]
        + 0.50 * features["top_fill_ratio"]
        - 0.75 * abs(features["top_centroid_dx"])
        - 0.55 * abs(features["front_tilt"])
        - 0.20 * features["front_slenderness"]
        - 0.25 * features["front_top_heaviness"]
    )
    return float(np.clip(0.5 + 0.35 * raw, 0.0, 1.0))


def physics_surrogate_features(features: dict[str, float], theory: str) -> dict[str, float]:
    support_half = 0.5 * features["top_support_width_frac"]
    eccentricity = abs(features["top_centroid_dx"])
    support_reserve = support_half - eccentricity
    base_reserve = features["front_base_width_frac"] - features["front_top_width_frac"]
    tilt_mag = abs(features["front_tilt"])
    slender = features["front_slenderness"]
    top_heavy = features["front_top_heaviness"]
    overturning = (
        1.05 * eccentricity
        + 0.70 * tilt_mag
        + 0.30 * slender
        + 0.40 * top_heavy
        - 0.45 * features["front_base_width_frac"]
    )
    support_margin = (
        1.20 * support_reserve
        + 0.55 * base_reserve
        + 0.30 * features["top_fill_ratio"]
        - 0.35 * top_heavy
    )
    hybrid = (
        0.80 * support_margin
        - 0.65 * overturning
        + 0.20 * features["collapse_margin"]
    )
    out = {
        "support_reserve": float(support_reserve),
        "base_reserve": float(base_reserve),
        "eccentricity_ratio": float(eccentricity / max(support_half, 1e-6)),
    }
    if theory == "classic":
        out["physics_score"] = float(features["collapse_margin"])
    elif theory == "support_margin":
        out["physics_score"] = float(support_margin)
    elif theory == "overturning":
        out["physics_score"] = float(-overturning)
    elif theory == "paper_hybrid":
        out["physics_score"] = float(0.55 * support_margin - 0.45 * overturning + 0.20 * features["collapse_margin"])
    else:
        out["physics_score"] = float(hybrid)
    return out


def paper_physics_scores(top_rgb: np.ndarray, top_mask: np.ndarray) -> dict[str, float]:
    if extract_top_components is None or merge_top_components is None:
        return {
            "paper_local_support_overlap": 0.0,
            "paper_counterbalance_score": 0.0,
            "paper_triplet_eccentricity": 0.0,
            "paper_stable_parse_energy": 0.0,
        }
    comps = merge_top_components(extract_top_components(top_rgb, color_bin=24, min_comp_frac=0.0012)[1])
    if len(comps) == 0:
        return {
            "paper_local_support_overlap": 0.0,
            "paper_counterbalance_score": 0.0,
            "paper_triplet_eccentricity": 0.0,
            "paper_stable_parse_energy": 0.0,
        }
    comps = sorted(comps, key=lambda x: x["area_frac"], reverse=True)[:3]
    areas = np.asarray([c["area_frac"] for c in comps], dtype=np.float32)
    cxs = np.asarray([c["cx"] for c in comps], dtype=np.float32)
    widths = np.asarray([c["bbox_w_frac"] for c in comps], dtype=np.float32)
    lefts = cxs - 0.5 * widths
    rights = cxs + 0.5 * widths

    overlaps = []
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            inter = max(0.0, min(float(rights[i]), float(rights[j])) - max(float(lefts[i]), float(lefts[j])))
            base = max(min(float(widths[i]), float(widths[j])), 1e-6)
            overlaps.append(inter / base)
    local_support_overlap = float(np.mean(overlaps)) if overlaps else 0.0

    total_area = float(np.maximum(areas.sum(), 1e-6))
    weighted_cx = float(np.sum(areas * cxs) / total_area)
    left_mass = float(np.sum(areas[cxs < 0]))
    right_mass = float(np.sum(areas[cxs >= 0]))
    counterbalance = 1.0 - abs(left_mass - right_mass) / max(total_area, 1e-6)

    if len(comps) >= 3:
        triplet_ecc = float(abs(weighted_cx) + np.std(cxs))
    else:
        triplet_ecc = float(abs(weighted_cx))

    support_width = float(max(rights.max() - lefts.min(), 1e-6))
    fill_ratio = float(top_mask.mean())
    stable_parse_energy = (
        1.15 * local_support_overlap
        + 0.90 * counterbalance
        + 0.55 * fill_ratio
        - 1.10 * triplet_ecc
        - 0.45 * abs(weighted_cx) / support_width
    )
    return {
        "paper_local_support_overlap": local_support_overlap,
        "paper_counterbalance_score": float(counterbalance),
        "paper_triplet_eccentricity": float(triplet_ecc),
        "paper_stable_parse_energy": float(stable_parse_energy),
    }


def extract_feature_row(
    front_img: Image.Image,
    top_img: Image.Image,
    physics_theory: str = "classic",
    box_aware_features: bool = False,
    adaptive_crop: bool = False,
    crop_margin_frac: float = 0.12,
    remove_shadow: bool = False,
    top_full_roi: bool = False,
    top_roi_margin_frac: float = 0.15,
) -> dict[str, float]:
    front_base = pil_adaptive_crop(front_img, "front", crop_margin_frac, remove_shadow=remove_shadow) if adaptive_crop else pil_center_crop(front_img, "front")
    if top_full_roi:
        top_base = pil_top_roi_crop(top_img, top_roi_margin_frac)
    else:
        top_base = pil_adaptive_crop(top_img, "top", crop_margin_frac, remove_shadow=remove_shadow) if adaptive_crop else pil_center_crop(top_img, "top")
    front_rgb = np.asarray(front_base.convert("RGB"), dtype=np.uint8)
    top_rgb = np.asarray(top_base.convert("RGB"), dtype=np.uint8)
    front_mask = estimate_foreground_mask(front_rgb, remove_shadow=remove_shadow)
    top_mask = estimate_foreground_mask(top_rgb, remove_shadow=remove_shadow)

    fx1, fy1, fx2, fy2 = safe_bbox(front_mask)
    tx1, ty1, tx2, ty2 = safe_bbox(top_mask)
    fh, fw = front_mask.shape
    th, tw = top_mask.shape
    bbox_h = max(fy2 - fy1 + 1, 1)

    top_cx, top_cy, top_sx, top_sy = centroid_stats(top_mask)
    front_cx, front_cy, front_sx, front_sy = centroid_stats(front_mask)
    top_quads = quadrant_occupancy(top_mask)
    front_quads = quadrant_occupancy(front_mask)
    top_cx_px = 0.5 * (top_cx + 1.0) * max(tw - 1, 1)
    front_cx_px = 0.5 * (front_cx + 1.0) * max(fw - 1, 1)

    features = {
        "top_area_frac": float(top_mask.mean()),
        "top_support_width_frac": float((tx2 - tx1 + 1) / max(tw, 1)),
        "top_support_height_frac": float((ty2 - ty1 + 1) / max(th, 1)),
        "top_fill_ratio": float(top_mask.sum() / max((tx2 - tx1 + 1) * (ty2 - ty1 + 1), 1)),
        "top_centroid_dx": top_cx,
        "top_centroid_dy": top_cy,
        "top_spread_x": top_sx,
        "top_spread_y": top_sy,
        "front_height_frac": float((fy2 - fy1 + 1) / max(fh, 1)),
        "front_width_frac": float((fx2 - fx1 + 1) / max(fw, 1)),
        "front_slenderness": float((fy2 - fy1 + 1) / max(fx2 - fx1 + 1, 1)),
        "front_base_width_frac": band_width(front_mask, fy2 - max(1, int(round(0.20 * bbox_h))) + 1, fy2 + 1),
        "front_mid_width_frac": band_width(front_mask, fy1 + int(round(0.40 * bbox_h)), fy1 + int(round(0.60 * bbox_h))),
        "front_top_width_frac": band_width(front_mask, fy1, fy1 + max(1, int(round(0.25 * bbox_h)))),
        "front_centroid_dx": front_cx,
        "front_centroid_dy": front_cy,
        "front_spread_x": front_sx,
        "front_spread_y": front_sy,
        "front_tilt": band_width(front_mask, fy1, fy1 + max(1, int(round(0.25 * bbox_h)))) - band_width(front_mask, fy2 - max(1, int(round(0.20 * bbox_h))) + 1, fy2 + 1),
        "front_top_heaviness": float(front_mask[fy1:fy1 + bbox_h // 2, :].sum() / max(front_mask.sum(), 1.0)),
        "top_q1": top_quads[0],
        "top_q2": top_quads[1],
        "top_q3": top_quads[2],
        "top_q4": top_quads[3],
        "front_q1": front_quads[0],
        "front_q2": front_quads[1],
        "front_q3": front_quads[2],
        "front_q4": front_quads[3],
    }
    features["collapse_margin"] = collapse_margin(features)
    features.update(physics_surrogate_features(features, physics_theory))
    features.update(paper_physics_scores(top_rgb, top_mask))
    if physics_theory == "paper_hybrid":
        features["physics_score"] = float(
            0.55 * features["paper_stable_parse_energy"]
            + 0.30 * features["paper_counterbalance_score"]
            + 0.20 * features["paper_local_support_overlap"]
            - 0.60 * features["paper_triplet_eccentricity"]
            + 0.25 * features["collapse_margin"]
        )
    features["top_left_right_bias"] = features["top_q1"] + features["top_q3"] - features["top_q2"] - features["top_q4"]
    features["top_top_bottom_bias"] = features["top_q1"] + features["top_q2"] - features["top_q3"] - features["top_q4"]
    features["front_left_right_bias"] = features["front_q1"] + features["front_q3"] - features["front_q2"] - features["front_q4"]
    features["support_minus_centroid"] = features["top_support_width_frac"] - abs(features["top_centroid_dx"])
    if box_aware_features:
        features["top_bbox_aspect"] = float((tx2 - tx1 + 1) / max(ty2 - ty1 + 1, 1))
        features["top_box_center_margin"] = bbox_center_margin(top_cx_px, tx1, tx2, tw)
        features["front_box_center_margin"] = bbox_center_margin(front_cx_px, fx1, fx2, fw)
        features["front_base_top_ratio"] = float(features["front_base_width_frac"] / max(features["front_top_width_frac"], 1e-6))
        features["front_bottom_contact_ratio"] = bottom_contact_ratio(front_mask, fx1, fx2, fy2)
    return features


def cluster_feature_vector(
    front_img: Image.Image,
    top_img: Image.Image,
    adaptive_crop: bool = False,
    crop_margin_frac: float = 0.12,
    remove_shadow: bool = False,
    top_full_roi: bool = False,
    top_roi_margin_frac: float = 0.15,
) -> np.ndarray:
    front_base = pil_adaptive_crop(front_img, "front", crop_margin_frac, remove_shadow=remove_shadow) if adaptive_crop else pil_center_crop(front_img, "front")
    if top_full_roi:
        top_base = pil_top_roi_crop(top_img, top_roi_margin_frac)
    else:
        top_base = pil_adaptive_crop(top_img, "top", crop_margin_frac, remove_shadow=remove_shadow) if adaptive_crop else pil_center_crop(top_img, "top")
    front = front_base.convert("L").resize((24, 24))
    top = top_base.convert("L").resize((24, 24))
    return np.concatenate([
        np.asarray(front, dtype=np.float32).reshape(-1) / 255.0,
        np.asarray(top, dtype=np.float32).reshape(-1) / 255.0,
    ])


def maybe_top_normalizer(enabled: bool):
    if enabled and CheckerboardTopNormalizer is not None and CheckerboardTopNormConfig is not None:
        return CheckerboardTopNormalizer(CheckerboardTopNormConfig(enabled=True))
    return None


def build_feature_table(
    df: pd.DataFrame,
    use_top_normalize: bool,
    physics_theory: str = "classic",
    box_aware_features: bool = False,
    adaptive_crop: bool = False,
    crop_margin_frac: float = 0.12,
    remove_shadow: bool = False,
    top_full_roi: bool = False,
    top_roi_margin_frac: float = 0.15,
) -> pd.DataFrame:
    top_norm = maybe_top_normalizer(use_top_normalize)
    rows = []
    for row in df.itertuples(index=False):
        front_img = Image.open(row.front_path).convert("RGB")
        top_img = Image.open(row.top_path).convert("RGB")
        if top_norm is not None:
            top_img = top_norm.normalize(row.top_path, top_img)
        feats = extract_feature_row(
            front_img,
            top_img,
            physics_theory=physics_theory,
            box_aware_features=box_aware_features,
            adaptive_crop=adaptive_crop,
            crop_margin_frac=crop_margin_frac,
            remove_shadow=remove_shadow,
            top_full_roi=top_full_roi,
            top_roi_margin_frac=top_roi_margin_frac,
        )
        feats["id"] = row.id
        rows.append(feats)
    return pd.DataFrame(rows)


def add_clusters(
    df: pd.DataFrame,
    path_df: pd.DataFrame,
    use_top_normalize: bool,
    n_clusters: int,
    seed: int,
    adaptive_crop: bool = False,
    crop_margin_frac: float = 0.12,
    remove_shadow: bool = False,
    top_full_roi: bool = False,
    top_roi_margin_frac: float = 0.15,
) -> pd.DataFrame:
    top_norm = maybe_top_normalizer(use_top_normalize)
    vecs = []
    for row in path_df.itertuples(index=False):
        front_img = Image.open(row.front_path).convert("RGB")
        top_img = Image.open(row.top_path).convert("RGB")
        if top_norm is not None:
            top_img = top_norm.normalize(row.top_path, top_img)
        vecs.append(
            cluster_feature_vector(
                front_img,
                top_img,
                adaptive_crop=adaptive_crop,
                crop_margin_frac=crop_margin_frac,
                remove_shadow=remove_shadow,
                top_full_roi=top_full_roi,
                top_roi_margin_frac=top_roi_margin_frac,
            )
        )
    X = np.stack(vecs)
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20)
    out = df.copy()
    out["geometry_cluster"] = km.fit_predict(Xs)
    return out


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


def feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"id", "target", "label", "front_path", "top_path", "source_split", "geometry_cluster"}
    return [c for c in df.columns if c not in exclude]


def train_cv(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pooled_paths, test_paths = build_tables(args.data_root)

    pooled_feat = build_feature_table(
        pooled_paths,
        args.use_top_normalize,
        physics_theory=args.physics_theory,
        box_aware_features=args.box_aware_features,
        adaptive_crop=args.adaptive_crop,
        crop_margin_frac=args.crop_margin_frac,
        remove_shadow=args.remove_shadow,
        top_full_roi=args.top_full_roi,
        top_roi_margin_frac=args.top_roi_margin_frac,
    )
    pooled = pooled_paths[["id", "target", "front_path", "top_path"]].merge(pooled_feat, on="id", how="left")
    pooled = add_clusters(
        pooled,
        pooled_paths[["id", "front_path", "top_path"]],
        args.use_top_normalize,
        args.n_clusters,
        args.seed,
        adaptive_crop=args.adaptive_crop,
        crop_margin_frac=args.crop_margin_frac,
        remove_shadow=args.remove_shadow,
        top_full_roi=args.top_full_roi,
        top_roi_margin_frac=args.top_roi_margin_frac,
    )

    test_feat = build_feature_table(
        test_paths,
        args.use_top_normalize,
        physics_theory=args.physics_theory,
        box_aware_features=args.box_aware_features,
        adaptive_crop=args.adaptive_crop,
        crop_margin_frac=args.crop_margin_frac,
        remove_shadow=args.remove_shadow,
        top_full_roi=args.top_full_roi,
        top_roi_margin_frac=args.top_roi_margin_frac,
    )
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
        mlp = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=400, random_state=args.seed + fold)
        hgb = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=4,
            max_iter=400,
            l2_regularization=1e-2,
            min_samples_leaf=20,
            random_state=args.seed + fold,
        )

        logreg.fit(X_tr_scaled, y_tr)
        mlp.fit(X_tr_scaled, y_tr)
        hgb.fit(X_tr, y_tr)

        p_lr = logreg.predict_proba(X_va_scaled)[:, 1]
        p_mlp = mlp.predict_proba(X_va_scaled)[:, 1]
        p_hgb = hgb.predict_proba(X_va)[:, 1]
        p_blend = 0.20 * p_lr + 0.25 * p_mlp + 0.55 * p_hgb
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
                    "mlp": mlp,
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
    config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    test_df = pd.read_csv(args.run_dir / "test_features.csv", encoding="utf-8-sig")
    feats = [c for c in test_df.columns if c not in {"id", "front_path", "top_path"}]

    fold_dirs = sorted([p for p in args.run_dir.iterdir() if p.is_dir() and p.name.startswith("fold")])
    preds = []
    for fold_dir in fold_dirs:
        with (fold_dir / "artifacts.pkl").open("rb") as f:
            art = pickle.load(f)
        X = test_df[art["features"]].to_numpy(dtype=np.float32)
        X_scaled = art["scaler"].transform(X)
        p_lr = art["logreg"].predict_proba(X_scaled)[:, 1]
        p_mlp = art["mlp"].predict_proba(X_scaled)[:, 1]
        p_hgb = art["hgb"].predict_proba(X)[:, 1]
        p_blend = 0.20 * p_lr + 0.25 * p_mlp + 0.55 * p_hgb
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
