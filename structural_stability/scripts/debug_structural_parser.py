#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from train_block_graph_baseline import (
    apply_top_roi_crop,
    build_tables,
    component_summary,
    extract_front_bands,
    extract_top_components,
    global_mask_stats,
    maybe_top_normalizer,
)
from train_support_graph_baseline import extract_front_layers, merge_top_components


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Debug structural parser outputs for top blocks and front layers.")
    p.add_argument("--data-root", type=Path, default=Path("/data/AskFake/Image/physics"))
    p.add_argument("--split", choices=["train", "dev", "test"], default="train")
    p.add_argument("--sample-id", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--use-top-normalize", action="store_true")
    p.add_argument("--top-roi-margin-frac", type=float, default=0.15)
    p.add_argument("--top-color-bin", type=int, default=32)
    p.add_argument("--top-min-comp-frac", type=float, default=0.0015)
    p.add_argument("--front-band-thresh", type=float, default=0.10)
    return p.parse_args()


def lookup_paths(data_root: Path, split: str, sample_id: str) -> tuple[Path, Path]:
    root = data_root / split / sample_id
    return root / "front.png", root / "top.png"


def draw_top_overlay(img: Image.Image, components: list[dict[str, float]]) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    colors = [
        (255, 90, 90),
        (90, 220, 120),
        (90, 160, 255),
        (255, 180, 70),
        (220, 90, 255),
        (80, 230, 230),
    ]
    for idx, comp in enumerate(components):
        cx = int(round((0.5 * (comp["cx"] + 1.0)) * max(w - 1, 1)))
        cy = int(round((0.5 * (comp["cy"] + 1.0)) * max(h - 1, 1)))
        bw = max(4, int(round(comp["bbox_w_frac"] * w)))
        bh = max(4, int(round(comp["bbox_h_frac"] * h)))
        x1 = max(0, cx - bw // 2)
        y1 = max(0, cy - bh // 2)
        x2 = min(w - 1, cx + bw // 2)
        y2 = min(h - 1, cy + bh // 2)
        color = colors[idx % len(colors)]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        draw.text((x1 + 2, max(0, y1 - 14)), f"{idx}", fill=color)
    return out


def draw_front_overlay(img: Image.Image, bands: list[dict[str, float]]) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    colors = [
        (255, 90, 90),
        (90, 220, 120),
        (90, 160, 255),
        (255, 180, 70),
        (220, 90, 255),
        (80, 230, 230),
    ]
    for idx, band in enumerate(bands):
        cy = int(round((0.5 * (band["cy"] + 1.0)) * max(h - 1, 1)))
        bh = max(4, int(round(band["bbox_h_frac"] * h)))
        bw = max(4, int(round(band["bbox_w_frac"] * w)))
        cx = int(round((0.5 * (band["cx"] + 1.0)) * max(w - 1, 1)))
        y1 = max(0, cy - bh // 2)
        y2 = min(h - 1, cy + bh // 2)
        x1 = max(0, cx - bw // 2)
        x2 = min(w - 1, cx + bw // 2)
        color = colors[idx % len(colors)]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        draw.text((x1 + 2, max(0, y1 - 14)), f"{idx}", fill=color)
    return out


def save_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L").save(path)


def main() -> None:
    args = parse_args()
    front_path, top_path = lookup_paths(args.data_root, args.split, args.sample_id)
    front_img = Image.open(front_path).convert("RGB")
    top_img = Image.open(top_path).convert("RGB")

    top_norm = maybe_top_normalizer(args.use_top_normalize)
    if top_norm is not None:
        top_img = top_norm.normalize(str(top_path), top_img)

    top_roi = apply_top_roi_crop(top_img, args.top_roi_margin_frac)
    top_rgb = np.asarray(top_roi.convert("RGB"), dtype=np.uint8)
    front_rgb = np.asarray(front_img.convert("RGB"), dtype=np.uint8)

    top_mask, top_components = extract_top_components(top_rgb, args.top_color_bin, args.top_min_comp_frac)
    top_components = merge_top_components(top_components)
    front_mask, front_bands = extract_front_layers(front_rgb, args.front_band_thresh)

    top_overlay = draw_top_overlay(top_roi, top_components)
    front_overlay = draw_front_overlay(front_img, front_bands)

    out_dir = args.out_dir / args.split / args.sample_id
    out_dir.mkdir(parents=True, exist_ok=True)
    front_img.save(out_dir / "front_original.png")
    top_img.save(out_dir / "top_original.png")
    top_roi.save(out_dir / "top_roi.png")
    top_overlay.save(out_dir / "top_blocks_overlay.png")
    front_overlay.save(out_dir / "front_layers_overlay.png")
    save_mask(top_mask, out_dir / "top_mask.png")
    save_mask(front_mask, out_dir / "front_mask.png")

    summary = {
        "id": args.sample_id,
        "top_global": global_mask_stats(top_mask),
        "front_global": global_mask_stats(front_mask),
        "top_blocks_summary": component_summary(top_components, "top"),
        "front_bands_summary": component_summary(front_bands, "front"),
        "top_blocks": top_components,
        "front_bands": front_bands,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"saved": str(out_dir), "top_blocks": len(top_components), "front_bands": len(front_bands)}))


if __name__ == "__main__":
    main()
