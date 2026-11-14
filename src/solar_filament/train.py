"""Train a binary filament model from competition training annotations only."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .data import find_data_root, group_views, validation_filenames
from .model import build_model
from .sampling import sample_object_point
from .features import FEATURE_SPECS, make_features
from .annotation_views import view_index


class PatchDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path,
        filenames: list[str],
        crop_size: int = 512,
        samples_per_epoch: int = 1132,
        seed: int = 2026,
        sampling: str = 'pixel',
        object_points_dir: Path | None = None,
        feature_recipe: str = 'contrast',
        view_selection: str = 'most_complete',
        annotation_views_dir: Path | None = None,
    ) -> None:
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.filenames = filenames
        self.crop_size = crop_size
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.epoch = 0
        self.order = filenames.copy()
        self.sampling = sampling
        self.object_points_dir = object_points_dir
        self.view_selection = view_selection
        self.annotation_views_dir = annotation_views_dir
        if view_selection not in ('most_complete', 'uniform'):
            raise ValueError('Unknown annotation-view selection')
        self.view_counts = {}
        if view_selection == 'uniform':
            if annotation_views_dir is None:
                raise ValueError('Uniform view sampling requires a view cache')
            manifest = json.loads((annotation_views_dir / 'manifest.json').read_text())
            if manifest['version'] != 1 or not set(filenames).issubset(manifest['filenames']):
                raise ValueError('View cache does not cover training observations')
            self.view_counts = manifest['view_counts']
        if feature_recipe not in FEATURE_SPECS:
            raise ValueError(f'Unknown feature recipe: {feature_recipe}')
        self.feature_recipe = feature_recipe
        if sampling == 'object' and object_points_dir is None and view_selection != 'uniform':
            raise ValueError('Object sampling requires a prepared object-point cache')

    @lru_cache(maxsize=8)
    def load_object_points(self, filename: str, view: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        path = (self.object_points_dir / f'{Path(filename).stem}.npz' if view is None else
                self.annotation_views_dir / Path(filename).stem / f'{view}.npz')
        with np.load(path, allow_pickle=False) as cache:
            return cache['points'], cache['offsets']

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        random.Random(self.seed + epoch).shuffle(self.order)

    @lru_cache(maxsize=8)
    def load_image(self, filename: str, view: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        image = cv2.imread(str(self.image_dir / filename), cv2.IMREAD_GRAYSCALE)
        mask_path = (self.mask_dir / f"{Path(filename).stem}.png" if view is None else
                     self.annotation_views_dir / Path(filename).stem / f'{view}.png')
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None or image.shape != (2048, 2048) or mask.shape != image.shape:
            raise OSError(f"Missing or malformed image/mask for {filename}")
        positive = cv2.findNonZero(mask)
        if positive is None:
            positive = np.empty((0, 1, 2), dtype=np.int32)
        return image, mask, positive.reshape(-1, 2)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_000 + index)
        filename = self.order[index % len(self.order)]
        view = (view_index(self.seed, self.epoch, index, self.view_counts[filename])
                if self.view_selection == 'uniform' else None)
        image, mask, positive = self.load_image(filename, view)
        crop = self.crop_size
        if len(positive) and rng.random() < 0.85:
            if self.sampling == 'object':
                points, offsets = self.load_object_points(filename, view)
                x, y = sample_object_point(points, offsets, rng)
            else:
                x, y = positive[rng.integers(len(positive))]
            x, y = int(x), int(y)
            x += int(rng.integers(-64, 65))
            y += int(rng.integers(-64, 65))
            left = int(np.clip(x - crop // 2, 0, image.shape[1] - crop))
            top = int(np.clip(y - crop // 2, 0, image.shape[0] - crop))
        else:
            left = int(rng.integers(0, image.shape[1] - crop + 1))
            top = int(rng.integers(0, image.shape[0] - crop + 1))
        tile = image[top : top + crop, left : left + crop].copy()
        target = mask[top : top + crop, left : left + crop].copy()
        if rng.random() < 0.5:
            tile, target = np.fliplr(tile), np.fliplr(target)
        if rng.random() < 0.5:
            tile, target = np.flipud(tile), np.flipud(target)
        if rng.random() < 0.5:
            tile, target = tile.T, target.T
        tile = np.ascontiguousarray(tile)
        target = np.ascontiguousarray(target)
        features = make_features(tile, self.feature_recipe)
        if rng.random() < 0.8:
            features[0] = np.clip(
                features[0] * float(rng.uniform(0.85, 1.15)) + float(rng.uniform(-0.04, 0.04)),
                0, 1,
            )
        return torch.from_numpy(features), torch.from_numpy((target > 0).astype(np.float32)[None])


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    positive_weight = torch.tensor(10.0, device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=positive_weight)
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * target).sum(dim=(2, 3))
    denominator = probabilities.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = 1 - ((2 * intersection + 1) / (denominator + 1)).mean()
    return bce + dice


def train(args: argparse.Namespace) -> None:
    data = find_data_root(args.data)
    grouped = group_views(data.load_annotations())
    validation = validation_filenames(grouped)
    if args.split_manifest is not None:
        split = json.loads(args.split_manifest.read_text())
        if split['annotations_sha256'] != hashlib.sha256(data.annotations_path.read_bytes()).hexdigest():
            raise ValueError('Split annotation checksum mismatch')
        validation = set(split['validation_filenames'])
        training = set(split['training_filenames'])
        if validation & training or validation | training != set(grouped):
            raise ValueError('Split must partition all physical observations')
        if {n[:8] for n in validation} & {n[:8] for n in training}:
            raise ValueError('Date-grouped split contains shared dates')
    filenames = sorted(set(grouped) - validation)
    if args.limit_images:
        filenames = filenames[: args.limit_images]
    if not filenames:
        raise ValueError("No training images selected")
    view_cache_hash = None
    if args.view_selection == 'uniform':
        manifest_path = args.annotation_views / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        if (manifest['version'] != 1 or manifest['annotations_sha256'] != hashlib.sha256(data.annotations_path.read_bytes()).hexdigest()
                or not set(filenames).issubset(manifest['filenames'])):
            raise ValueError('Annotation-view cache differs from training annotations')
        for name in filenames:
            if manifest['view_counts'][name] != len(grouped[name]):
                raise ValueError('Annotation-view count mismatch')
            for index in range(len(grouped[name])):
                for suffix in ('png', 'npz'):
                    relative = f'{Path(name).stem}/{index}.{suffix}'
                    if hashlib.sha256((args.annotation_views / relative).read_bytes()).hexdigest() != manifest['files_sha256'][relative]:
                        raise ValueError(f'Annotation-view cache changed: {relative}')
        view_cache_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    elif args.sampling == 'object':
        manifest = json.loads((args.object_points / 'manifest.json').read_text(encoding='utf-8'))
        digest = hashlib.sha256(data.annotations_path.read_bytes()).hexdigest()
        if manifest['annotations_sha256'] != digest or not set(filenames).issubset(manifest['filenames']):
            raise ValueError('Object cache does not match the current annotations and training split')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PatchDataset(
        data.train_dir, args.masks, filenames,
        crop_size=args.crop_size,
        samples_per_epoch=args.samples_per_epoch or 2 * len(filenames),
        seed=args.seed,
        sampling=args.sampling,
        object_points_dir=args.object_points,
        feature_recipe=args.feature_recipe,
        view_selection=args.view_selection,
        annotation_views_dir=args.annotation_views,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == 'cuda')
    start_epoch = 0
    if args.init_checkpoint is not None:
        initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
        if initial.get('feature_recipe', 'contrast') != args.feature_recipe:
            raise ValueError('Initial checkpoint feature recipe differs; train each ablation from initialization')
        if args.split_manifest is not None:
            prior_training = initial.get('training_filenames')
            if prior_training is None or set(prior_training) & validation:
                raise ValueError('Initial checkpoint cannot be proven to exclude this split; retrain from encoder initialization')
        if (initial.get('architecture', 'compact') != args.architecture
                or initial["base_channels"] != args.base_channels or initial["crop_size"] != args.crop_size):
            raise ValueError("Initial checkpoint architecture or crop size does not match arguments")
        start_epoch = int(initial["epoch"])
    model = build_model(args.architecture, args.base_channels,
                        args.encoder_weights if args.init_checkpoint is None else None).to(device)
    if args.init_checkpoint is not None:
        model.load_state_dict(initial["model"])
    if args.architecture == 'resnet34':
        encoder_ids = {id(p) for p in model.encoder.parameters()}
        parameters = [
            {'params': model.encoder.parameters(), 'lr': args.learning_rate * args.encoder_lr_factor},
            {'params': [p for p in model.parameters() if id(p) not in encoder_ids], 'lr': args.learning_rate},
        ]
    else:
        parameters = model.parameters()
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for epoch in range(start_epoch, start_epoch + args.epochs):
        dataset.set_epoch(epoch)
        model.train()
        losses: list[float] = []
        for step, (features, target) in enumerate(loader, 1):
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = segmentation_loss(model(features), target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
            if step % 50 == 0 or step == len(loader):
                print(f"epoch {epoch + 1}/{start_epoch + args.epochs} step {step}/{len(loader)} loss={np.mean(losses[-50:]):.4f}", flush=True)
        scheduler.step()
        checkpoint = {
            "model": model.state_dict(),
            "base_channels": args.base_channels,
            "crop_size": args.crop_size,
            "epoch": epoch + 1,
            "train_images": len(filenames),
            "validation_images": len(validation),
            "seed": args.seed,
            "mean_loss": float(np.mean(losses)),
            "sampling": args.sampling,
            "view_selection": args.view_selection,
            "annotation_views_manifest_sha256": view_cache_hash,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "learning_rate": args.learning_rate,
            "samples_per_epoch": len(dataset),
            "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
            "architecture": args.architecture,
            "encoder_weights": str(args.encoder_weights) if args.encoder_weights else None,
            "encoder_lr_factor": args.encoder_lr_factor,
            "feature_recipe": args.feature_recipe,
            "feature_spec": FEATURE_SPECS[args.feature_recipe],
            "training_filenames": filenames,
            "validation_filenames": sorted(validation),
            "annotations_sha256": hashlib.sha256(data.annotations_path.read_bytes()).hexdigest(),
            "split_manifest": str(args.split_manifest) if args.split_manifest else None,
            "elapsed_seconds": time.monotonic() - started,
        }
        torch.save(checkpoint, args.output)
        if args.save_every and (epoch + 1) % args.save_every == 0:
            torch.save(checkpoint, args.output.with_name(f'{args.output.stem}_epoch{epoch + 1}.pt'))
        print(json.dumps({"epoch": epoch + 1, "loss": checkpoint["mean_loss"],
                          "elapsed_seconds": round(time.monotonic() - started, 1),
                          "checkpoint": str(args.output)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/kagglehub/competitions/filament-segmentation-2026")
    parser.add_argument("--masks", type=Path, default=Path("data/prepared/masks"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/filament_unet.pt"))
    parser.add_argument("--init-checkpoint", type=Path, default=None,
                        help="Continue weights from an earlier checkpoint using a fresh optimizer")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--samples-per-epoch", type=int, default=None)
    parser.add_argument("--limit-images", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument('--sampling', choices=('pixel', 'object'), default='pixel')
    parser.add_argument('--object-points', type=Path, default=Path('data/prepared/object_points'))
    parser.add_argument('--architecture', choices=('compact', 'resnet34'), default='compact')
    parser.add_argument('--encoder-weights', type=Path)
    parser.add_argument('--save-every', type=int, default=0)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--encoder-lr-factor', type=float, default=1.0)
    parser.add_argument('--feature-recipe', choices=FEATURE_SPECS, default='contrast')
    parser.add_argument('--split-manifest', type=Path)
    parser.add_argument('--view-selection', choices=('most_complete', 'uniform'), default='most_complete')
    parser.add_argument('--annotation-views', type=Path, default=Path('data/prepared/annotation_views_v1'))
    train(parser.parse_args())


if __name__ == "__main__":
    main()
