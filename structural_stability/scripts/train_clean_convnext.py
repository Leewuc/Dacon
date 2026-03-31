#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

try:
    from checkerboard_rectification import CheckerboardTopNormConfig, CheckerboardTopNormalizer
except Exception:
    CheckerboardTopNormConfig = None
    CheckerboardTopNormalizer = None


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean reproducible ConvNeXt CV pipeline for physics.")
    sub = p.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train")
    train.add_argument("--data-root", type=Path, default=Path("/data/AskFake/Image/physics"))
    train.add_argument("--out-dir", type=Path, required=True)
    train.add_argument("--model-name", type=str, default="convnext_small.fb_in22k_ft_in1k_384")
    train.add_argument("--image-size", type=int, default=384)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--accum-steps", type=int, default=4)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--num-workers", type=int, default=4)
    train.add_argument("--backbone-lr", type=float, default=2e-5)
    train.add_argument("--head-lr", type=float, default=2e-4)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--n-folds", type=int, default=5)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--use-top-normalize", action="store_true")
    train.add_argument("--front-mode", type=str, default="original", choices=["original", "center", "roi"])
    train.add_argument("--top-mode", type=str, default="original", choices=["original", "center", "roi"])
    train.add_argument("--front-roi-margin-frac", type=float, default=0.15)
    train.add_argument("--top-roi-margin-frac", type=float, default=0.15)
    train.add_argument("--jpeg-aug-prob", type=float, default=0.0)
    train.add_argument("--jpeg-quality-min", type=int, default=60)
    train.add_argument("--jpeg-quality-max", type=int, default=90)
    train.add_argument("--gamma-aug-prob", type=float, default=0.0)
    train.add_argument("--gamma-min", type=float, default=0.90)
    train.add_argument("--gamma-max", type=float, default=1.10)
    train.add_argument("--no-pretrained", action="store_true")

    predict = sub.add_parser("predict")
    predict.add_argument("--data-root", type=Path, default=Path("/data/AskFake/Image/physics"))
    predict.add_argument("--run-dir", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--batch-size", type=int, default=16)
    predict.add_argument("--num-workers", type=int, default=4)
    predict.add_argument("--tta", type=str, default="flip", choices=["none", "flip"])
    predict.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_csv_sig(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def load_tables(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = read_csv_sig(data_root / "train.csv")
    dev_df = read_csv_sig(data_root / "dev.csv")
    test_df = read_csv_sig(data_root / "sample_submission.csv")

    train_df["front_path"] = train_df["id"].apply(lambda x: str(data_root / "train" / x / "front.png"))
    train_df["top_path"] = train_df["id"].apply(lambda x: str(data_root / "train" / x / "top.png"))
    dev_df["front_path"] = dev_df["id"].apply(lambda x: str(data_root / "dev" / x / "front.png"))
    dev_df["top_path"] = dev_df["id"].apply(lambda x: str(data_root / "dev" / x / "top.png"))
    test_df["front_path"] = test_df["id"].apply(lambda x: str(data_root / "test" / x / "front.png"))
    test_df["top_path"] = test_df["id"].apply(lambda x: str(data_root / "test" / x / "top.png"))

    train_df["target"] = (train_df["label"] == "unstable").astype(np.float32)
    dev_df["target"] = (dev_df["label"] == "unstable").astype(np.float32)
    pooled = pd.concat([train_df, dev_df], ignore_index=True)
    return pooled, test_df


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


def safe_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    h, w = mask.shape
    if len(xs) == 0:
        return 0, 0, w - 1, h - 1
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def apply_full_roi_crop(img: Image.Image, margin_frac: float, view: str) -> Image.Image:
    rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    mask = estimate_full_roi_mask_view(rgb, view=view)
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
    crop = img.crop((x1, y1, x2 + 1, y2 + 1))
    side = max(crop.size)
    canvas = Image.new("RGB", (side, side), (127, 127, 127))
    ox = (side - crop.size[0]) // 2
    oy = (side - crop.size[1]) // 2
    canvas.paste(crop, (ox, oy))
    return canvas


def preprocess_view(img: Image.Image, view: str, mode: str, roi_margin_frac: float) -> Image.Image:
    if mode == "original":
        return img
    if mode == "center":
        return pil_center_crop(img, view=view)
    if mode == "roi":
        return apply_full_roi_crop(img, roi_margin_frac, view=view)
    raise ValueError(f"unknown mode: {mode}")


class RandomJpegCompression:
    def __init__(self, prob: float, quality_min: int, quality_max: int):
        self.prob = prob
        self.quality_min = quality_min
        self.quality_max = quality_max

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.prob <= 0.0 or random.random() >= self.prob:
            return img
        quality = random.randint(self.quality_min, self.quality_max)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class RandomGamma:
    def __init__(self, prob: float, gamma_min: float, gamma_max: float):
        self.prob = prob
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.prob <= 0.0 or random.random() >= self.prob:
            return img
        gamma = random.uniform(self.gamma_min, self.gamma_max)
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        arr = np.clip(arr, 0.0, 1.0) ** gamma
        arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(arr)


def build_train_transform(args: argparse.Namespace) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)],
                p=0.8,
            ),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.2),
            transforms.RandomAffine(degrees=15, translate=(0.10, 0.10), scale=(0.85, 1.15)),
            RandomJpegCompression(args.jpeg_aug_prob, args.jpeg_quality_min, args.jpeg_quality_max),
            RandomGamma(args.gamma_aug_prob, args.gamma_min, args.gamma_max),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_valid_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class StructureDataset(Dataset):
    def __init__(self, df: pd.DataFrame, args: argparse.Namespace, is_train: bool, is_test: bool = False):
        self.df = df.reset_index(drop=True).copy()
        self.args = args
        self.is_train = is_train
        self.is_test = is_test
        self.tf = build_train_transform(args) if is_train else build_valid_transform(args.image_size)
        self.top_normalizer = None
        if args.use_top_normalize and CheckerboardTopNormalizer is not None and CheckerboardTopNormConfig is not None:
            self.top_normalizer = CheckerboardTopNormalizer(CheckerboardTopNormConfig(enabled=True))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        front_img = Image.open(row["front_path"]).convert("RGB")
        top_img = Image.open(row["top_path"]).convert("RGB")
        if self.top_normalizer is not None:
            top_img = self.top_normalizer.normalize(row["top_path"], top_img)

        front_img = preprocess_view(front_img, "front", self.args.front_mode, self.args.front_roi_margin_frac)
        top_img = preprocess_view(top_img, "top", self.args.top_mode, self.args.top_roi_margin_frac)

        front = self.tf(front_img)
        top = self.tf(top_img)
        if self.is_test:
            return {"id": str(row["id"]), "front": front, "top": top}
        return {"id": str(row["id"]), "front": front, "top": top, "target": torch.tensor(float(row["target"]), dtype=torch.float32)}


class DualStreamModel(nn.Module):
    def __init__(self, backbone_name: str, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        feat_dim = int(self.backbone.num_features)
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, 1),
        )

    def forward(self, front: torch.Tensor, top: torch.Tensor) -> torch.Tensor:
        f = self.backbone(front)
        t = self.backbone(top)
        return self.head(torch.cat([f, t], dim=1)).squeeze(-1)


class TemperatureScaler(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=1e-3)

    def fit(self, logits: np.ndarray, y_true: np.ndarray, max_iter: int = 200) -> float:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        x = torch.tensor(logits, dtype=torch.float32, device=device)
        y = torch.tensor(y_true, dtype=torch.float32, device=device)
        opt = torch.optim.LBFGS(self.parameters(), lr=0.1, max_iter=max_iter)

        def closure():
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(self.forward(x), y)
            loss.backward()
            return loss

        opt.step(closure)
        return float(self.temperature.detach().cpu().item())


def train_one_fold(args: argparse.Namespace, fold: int, train_df: pd.DataFrame, val_df: pd.DataFrame, device: torch.device) -> pd.DataFrame:
    fold_dir = args.out_dir / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_ds = StructureDataset(train_df, args, is_train=True, is_test=False)
    val_ds = StructureDataset(val_df, args, is_train=False, is_test=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda", drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    model = DualStreamModel(args.model_name, pretrained=not args.no_pretrained).to(device)
    optimizer = AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.backbone_lr},
            {"params": model.head.parameters(), "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_ll = math.inf
    best_state = None
    best_logits = None
    best_ids = None
    best_targets = None

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            front = batch["front"].to(device, non_blocking=True)
            top = batch["top"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(front, top)
                loss = F.binary_cross_entropy_with_logits(logits, target) / args.accum_steps
            scaler.scale(loss).backward()
            if step % args.accum_steps == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            train_loss += loss.item() * args.accum_steps
        scheduler.step()
        train_loss /= max(len(train_loader), 1)

        model.eval()
        preds = []
        logits_all = []
        targets = []
        ids = []
        with torch.no_grad():
            for batch in val_loader:
                front = batch["front"].to(device, non_blocking=True)
                top = batch["top"].to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logits = model(front, top)
                logits_np = logits.float().cpu().numpy()
                prob = 1.0 / (1.0 + np.exp(-logits_np))
                preds.append(prob)
                logits_all.append(logits_np)
                targets.append(batch["target"].numpy())
                ids.extend(batch["id"])

        pred_np = np.concatenate(preds)
        logit_np = np.concatenate(logits_all)
        target_np = np.concatenate(targets)
        ll = log_loss(target_np, np.clip(pred_np, 1e-7, 1 - 1e-7))
        auc = roc_auc_score(target_np, pred_np)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(json.dumps({"fold": fold, "epoch": epoch + 1, "train_loss": float(train_loss), "val_auc": float(auc), "val_logloss": float(ll)}))
        if ll < best_ll:
            best_ll = float(ll)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_logits = logit_np.copy()
            best_ids = ids.copy()
            best_targets = target_np.copy()

    assert best_state is not None
    assert best_logits is not None
    torch.save(best_state, fold_dir / "best.pt")

    scaler_model = TemperatureScaler()
    temperature = scaler_model.fit(best_logits.astype(np.float32), best_targets.astype(np.float32))
    clipped_logits = np.clip(best_logits / max(temperature, 1e-6), -60.0, 60.0)
    cal_prob = 1.0 / (1.0 + np.exp(-clipped_logits))
    cal_logloss = log_loss(best_targets.astype(np.float32), np.clip(cal_prob, 1e-7, 1 - 1e-7))
    with (fold_dir / "temperature.json").open("w", encoding="utf-8") as f:
        json.dump({"temperature": temperature, "val_logloss": best_ll, "val_logloss_calibrated": float(cal_logloss)}, f, ensure_ascii=False, indent=2)

    return pd.DataFrame({"id": best_ids, "target": best_targets, f"fold{fold}_pred": cal_prob})


def train_cv(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pooled_df, _test_df = load_tables(args.data_root)
    config = vars(args).copy()
    with (args.out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2, default=str)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splitter = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    oof_frames = []
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(pooled_df, pooled_df["target"])):
        frame = train_one_fold(
            args,
            fold,
            pooled_df.iloc[tr_idx].reset_index(drop=True),
            pooled_df.iloc[va_idx].reset_index(drop=True),
            device,
        )
        oof_frames.append(frame)

    oof = None
    for frame in oof_frames:
        if oof is None:
            oof = frame
        else:
            oof = oof.merge(frame, on=["id", "target"], how="outer")
    assert oof is not None
    pred_cols = [c for c in oof.columns if c.endswith("_pred")]
    oof["pred"] = oof[pred_cols].sum(axis=1, skipna=True)
    for col in pred_cols:
        oof["pred"] = np.where(oof[col].notna(), oof[col], oof["pred"])
    oof.to_csv(args.out_dir / "oof.csv", index=False, encoding="utf-8-sig")


def predict_fold(model: DualStreamModel, loader: DataLoader, device: torch.device, tta: str, temperature: float) -> tuple[list[str], np.ndarray]:
    model.eval()
    ids_all = []
    preds_all = []
    with torch.no_grad():
        for batch in loader:
            front = batch["front"].to(device, non_blocking=True)
            top = batch["top"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(front, top)
                if tta == "flip":
                    logits_flip = model(torch.flip(front, dims=[3]), torch.flip(top, dims=[3]))
                    logits = 0.5 * (logits + logits_flip)
            logits = logits / max(temperature, 1e-6)
            probs = torch.sigmoid(logits.float()).cpu().numpy()
            ids_all.extend(batch["id"])
            preds_all.append(probs)
    return ids_all, np.concatenate(preds_all)


def predict_cv(args: argparse.Namespace) -> None:
    config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    merged = vars(args).copy()
    merged.update(config)
    merged["command"] = "predict"
    ns = argparse.Namespace(**merged)
    device = torch.device(args.device)
    _pooled_df, test_df = load_tables(args.data_root)
    test_ds = StructureDataset(test_df, ns, is_train=False, is_test=True)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    fold_dirs = sorted([p for p in args.run_dir.iterdir() if p.is_dir() and p.name.startswith("fold")])
    all_preds = []
    ordered_ids = None
    for fold_dir in fold_dirs:
        state = torch.load(fold_dir / "best.pt", map_location="cpu", weights_only=False)
        temp_info = json.loads((fold_dir / "temperature.json").read_text(encoding="utf-8"))
        model = DualStreamModel(ns.model_name, pretrained=False).to(device)
        model.load_state_dict(state)
        ids, preds = predict_fold(model, loader, device, args.tta, float(temp_info["temperature"]))
        if ordered_ids is None:
            ordered_ids = ids
        all_preds.append(preds)

    final_preds = np.mean(np.stack(all_preds, axis=0), axis=0)
    final_preds = np.clip(final_preds, 1e-7, 1 - 1e-7)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "unstable_prob", "stable_prob"])
        for sid, prob in zip(ordered_ids, final_preds):
            writer.writerow([sid, f"{prob:.10f}", f"{1.0 - prob:.10f}"])
    print(f"submission_rows={len(ordered_ids)} -> {args.output}")


def main() -> None:
    args = parse_args()
    if args.command == "train":
        train_cv(args)
    else:
        predict_cv(args)


if __name__ == "__main__":
    main()
