#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

try:
    from checkerboard_rectification import CheckerboardTopNormConfig, CheckerboardTopNormalizer
except Exception:
    CheckerboardTopNormConfig = None
    CheckerboardTopNormalizer = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Debug full-image ROI crop using foreground union bbox.")
    p.add_argument("--data-root", type=Path, default=Path("/data/AskFake/Image/physics"))
    p.add_argument("--split", type=str, default="train", choices=["train", "dev", "test"])
    p.add_argument("--sample-id", type=str, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("/data/AskFake/Image/physics/baseline/debug_full_roi"))
    p.add_argument("--margin-frac", type=float, default=0.15)
    p.add_argument("--use-top-normalize", action="store_true")
    p.add_argument("--front-mode", type=str, default="hybrid", choices=["roi", "hybrid", "original", "center10", "vtrim", "geomv1"])
    return p.parse_args()


def gray_from_rgb(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def sat_from_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32) / 255.0
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    return np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)


def estimate_mask_full(rgb: np.ndarray) -> np.ndarray:
    return estimate_mask_full_view(rgb, view="generic")


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


def remove_small_components(mask: np.ndarray, min_area_frac: float = 0.002) -> np.ndarray:
    h, w = mask.shape
    min_area = max(8, int(round(h * w * min_area_frac)))
    out = np.zeros_like(mask, dtype=np.uint8)
    for comp in iter_components(mask):
        if len(comp) >= min_area:
            for y, x in comp:
                out[y, x] = 1
    return out


def suppress_bottom_wide_band(mask: np.ndarray, start_thresh: float = 0.35, keep_thresh: float = 0.22) -> np.ndarray:
    h, w = mask.shape
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


def estimate_mask_full_view(rgb: np.ndarray, view: str) -> np.ndarray:
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

    # Remove small bright border artifacts such as circular lights.
    yy, xx = np.mgrid[0:h, 0:w]
    near_border = (xx < int(0.14 * w)) | (xx > int(0.86 * w)) | (yy < int(0.14 * h)) | (yy > int(0.86 * h))
    bright_blob = (val > np.percentile(val, 97.0)) & (sat < np.percentile(sat, 55.0))
    mask[near_border & bright_blob] = 0

    # Front view often has a non-structural top background band; suppress it.
    if view == "front":
        top_band = int(0.12 * h)
        mask[:top_band, :] = 0
        mask = suppress_bottom_wide_band(mask)
        mask = suppress_front_grid_components(mask, bg_dist)

    mask = suppress_blue_grid_components(mask, rgbf, sat)

    mask = remove_small_components(mask, min_area_frac=0.0025)
    return mask


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


def safe_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    h, w = mask.shape
    if len(xs) == 0:
        return 0, 0, w - 1, h - 1
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def roi_crop(img: Image.Image, margin_frac: float, view: str) -> tuple[Image.Image, tuple[int, int, int, int], np.ndarray]:
    rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    mask = estimate_mask_full_view(rgb, view=view)
    mask = dilate_mask(mask, radius=max(3, min(mask.shape) // 50))
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
    return canvas, (x1, y1, x2, y2), mask


def overlay_mask(rgb: np.ndarray, mask: np.ndarray) -> Image.Image:
    out = rgb.astype(np.float32).copy()
    out[..., 1] = np.where(mask > 0, np.clip(out[..., 1] * 0.5 + 120, 0, 255), out[..., 1])
    out[..., 0] = np.where(mask > 0, np.clip(out[..., 0] * 0.5, 0, 255), out[..., 0])
    out[..., 2] = np.where(mask > 0, np.clip(out[..., 2] * 0.5, 0, 255), out[..., 2])
    return Image.fromarray(out.astype(np.uint8))


def draw_bbox(img: Image.Image, bbox: tuple[int, int, int, int]) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle(bbox, outline=(255, 0, 0), width=3)
    return out


def front_fallback_crop(img: Image.Image, mode: str) -> Image.Image:
    if mode == "original":
        return img.copy()
    if mode == "geomv1":
        w, h = img.size
        x1, x2 = int(round(w * 0.25)), int(round(w * 0.75))
        y1, y2 = int(round(h * 0.20)), int(round(h * 0.88))
        return img.crop((x1, y1, x2, y2))
    if mode == "vtrim":
        w, h = img.size
        y1, y2 = int(round(h * 0.10)), int(round(h * 0.90))
        return img.crop((0, y1, w, y2))
    if mode == "center10":
        w, h = img.size
        x1, x2 = int(round(w * 0.10)), int(round(w * 0.90))
        y1, y2 = int(round(h * 0.10)), int(round(h * 0.90))
        return img.crop((x1, y1, x2, y2))
    return img.copy()


def main() -> None:
    args = parse_args()
    sample_dir = args.data_root / args.split / args.sample_id
    front = Image.open(sample_dir / "front.png").convert("RGB")
    top = Image.open(sample_dir / "top.png").convert("RGB")

    if args.use_top_normalize and CheckerboardTopNormalizer is not None and CheckerboardTopNormConfig is not None:
        top = CheckerboardTopNormalizer(CheckerboardTopNormConfig(enabled=True)).normalize(sample_dir / "top.png", top)

    front_crop, front_bbox, front_mask = roi_crop(front, args.margin_frac, view="front")
    top_crop, top_bbox, top_mask = roi_crop(top, args.margin_frac, view="top")

    if args.front_mode in {"hybrid", "original", "center10", "vtrim", "geomv1"}:
        front_crop = front_fallback_crop(front, "original" if args.front_mode == "hybrid" else args.front_mode)

    out_dir = args.out_dir / args.split / args.sample_id
    out_dir.mkdir(parents=True, exist_ok=True)

    front.save(out_dir / "front_full_original.png")
    top.save(out_dir / "top_full_original.png")
    draw_bbox(front, front_bbox).save(out_dir / "front_full_bbox.png")
    draw_bbox(top, top_bbox).save(out_dir / "top_full_bbox.png")
    Image.fromarray((front_mask * 255).astype(np.uint8)).save(out_dir / "front_full_mask.png")
    Image.fromarray((top_mask * 255).astype(np.uint8)).save(out_dir / "top_full_mask.png")
    overlay_mask(np.asarray(front), front_mask).save(out_dir / "front_full_overlay.png")
    overlay_mask(np.asarray(top), top_mask).save(out_dir / "top_full_overlay.png")
    front_crop.save(out_dir / "front_full_roi.png")
    top_crop.save(out_dir / "top_full_roi.png")

    print(f"saved={out_dir}")
    print(f"front_bbox={front_bbox}")
    print(f"top_bbox={top_bbox}")


if __name__ == "__main__":
    main()
