"""Whole-disk U-Net with a ConvNeXt-Tiny encoder and a boundary output.

The current members see 512-pixel crops at native resolution. This model sees the full
solar disk resized to ``--size`` (1024, then 1536), so long filaments and their
surroundings fall inside one receptive field. Ideas follow the public NSO entry
(github.com/pragyaangaur/NSO-Solar-Filament-Challenge); the code is written here.

* Encoder: timm ``convnext_tiny.in12k_ft_in1k`` (ImageNet weights, same category as the
  ResNet34 weights already used), strides 4 to 32; decoder of five upsampling blocks.
* Outputs: filament logit and instance-boundary logit (dilation minus erosion, 3 x 3, of
  every annotated filament at 2048 pixels, then area-resized).
* Targets: one random annotator view per sample per step, area-resized (soft at edges).
* Loss: BCE + soft Dice on the filament output, plus 0.2 x BCE (positive weight 3) on the
  boundary output.
* Augmentation: rot90, flips, brightness, contrast, gamma, blur, noise.
* AdamW (weight decay 0.01, encoder learning rate / 5), warmup then cosine, gradient
  clipping 5, bf16 autocast (``--amp fp16`` with loss scaling on GPUs without bf16, such
  as the T4 on Kaggle and Colab), EMA of weights (decay 0.99 per optimizer step). The EMA
  weights are saved and used for prediction.

Training uses one date-grouped fold file (``results/annotation_round_folds/fold_k.json``):
its training filenames only. ``--predict`` writes out-of-fold probability maps for the
fold's validation filenames (or the test images) as uint16 PNG at 2048 pixels, the format
the candidate scorer reads.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from solar_filament.data import annotation_mask, find_data_root, group_views, read_grayscale

ENCODER = 'convnext_tiny.in12k_ft_in1k'
MEAN, STD = 0.485, 0.229
EDGE = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
AMP = torch.bfloat16  # set from --amp in main()


class Up(nn.Module):
    def __init__(self, inputs: int, skip: int, outputs: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inputs + skip, outputs, 3, padding=1, bias=False), nn.BatchNorm2d(outputs), nn.ReLU(inplace=True),
            nn.Conv2d(outputs, outputs, 3, padding=1, bias=False), nn.BatchNorm2d(outputs), nn.ReLU(inplace=True))

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], 1)
        return self.block(x)


def stride_indices(encoder: str) -> tuple[int, ...]:
    """Feature indices at strides 4, 8, 16 and 32 (ConvNeXt: 0 to 3; EfficientNet: 1 to 4).

    The probe model is built under a forked RNG so the real model's initialization is the
    same as before this helper existed."""
    with torch.random.fork_rng(devices=[]):
        reductions = timm.create_model(encoder, pretrained=False, features_only=True).feature_info.reduction()
    return tuple(reductions.index(r) for r in (4, 8, 16, 32))


class GlobalUNet(nn.Module):
    def __init__(self, encoder: str = ENCODER, pretrained: bool = True):
        super().__init__()
        self.encoder = timm.create_model(encoder, pretrained=pretrained, features_only=True,
                                         out_indices=stride_indices(encoder))
        c = self.encoder.feature_info.channels()
        self.up = nn.ModuleList([Up(c[3], c[2], 256), Up(256, c[1], 128), Up(128, c[0], 64), Up(64, 0, 32), Up(32, 0, 16)])
        self.head = nn.Conv2d(16, 2, 1)

    def forward(self, x):
        f = self.encoder(x)
        x = self.up[0](f[3], f[2])
        x = self.up[1](x, f[1])
        x = self.up[2](x, f[0])
        x = self.up[3](x)
        x = self.up[4](x)
        return self.head(x)


def to_input(image: np.ndarray) -> np.ndarray:
    """uint8 or float image at model size to a normalized (3, H, W) float32 array."""
    x = (image.astype(np.float32) / 255.0 - MEAN) / STD
    return np.repeat(x[None], 3, axis=0)


def targets(views: list[list[dict]], rng: np.random.Generator, size: int) -> tuple[np.ndarray, np.ndarray]:
    view = views[int(rng.integers(len(views)))]
    foreground = np.zeros((2048, 2048), np.uint8)
    boundary = np.zeros((2048, 2048), np.uint8)
    for annotation in view:
        mask = annotation_mask(annotation).astype(np.uint8)
        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            continue
        y0, y1, x0, x1 = max(ys.min() - 2, 0), ys.max() + 3, max(xs.min() - 2, 0), xs.max() + 3
        crop = mask[y0:y1, x0:x1]
        foreground[y0:y1, x0:x1] |= crop
        boundary[y0:y1, x0:x1] |= cv2.dilate(crop, EDGE) - cv2.erode(crop, EDGE)
    resize = lambda m: cv2.resize(m.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    return resize(foreground), resize(boundary)


def augment(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = image.astype(np.float32)
    if rng.random() < 0.8:
        x = x * rng.uniform(0.85, 1.15) + rng.uniform(-20, 20)
    if rng.random() < 0.5:
        x = 255.0 * (np.clip(x, 0, 255) / 255.0) ** rng.uniform(0.8, 1.25)
    if rng.random() < 0.15:
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.5, 1.5))
    if rng.random() < 0.15:
        x = x + rng.normal(0, rng.uniform(2, 6), x.shape).astype(np.float32)
    return np.clip(x, 0, 255)


class DiskDataset(Dataset):
    def __init__(self, data, grouped, names: list[str], size: int):
        self.data, self.grouped, self.names, self.size = data, grouped, names, size

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        rng = np.random.default_rng([torch.initial_seed() % 2**32, index])
        name = self.names[index]
        image = cv2.resize(read_grayscale(self.data.train_dir / name), (self.size,) * 2, interpolation=cv2.INTER_AREA)
        foreground, boundary = targets(self.grouped[name], rng, self.size)
        image = augment(image, rng)
        k, flip_h, flip_v = int(rng.integers(4)), rng.random() < 0.5, rng.random() < 0.5
        stack = [image, foreground, boundary]
        stack = [np.rot90(a, k) for a in stack]
        if flip_h:
            stack = [a[:, ::-1] for a in stack]
        if flip_v:
            stack = [a[::-1] for a in stack]
        image, foreground, boundary = (np.ascontiguousarray(a) for a in stack)
        return torch.from_numpy(to_input(image)), torch.from_numpy(foreground), torch.from_numpy(boundary)


def loss_fn(logits, foreground, boundary):
    logits = logits.float()
    bce = F.binary_cross_entropy_with_logits(logits[:, 0], foreground)
    p = torch.sigmoid(logits[:, 0])
    dice = 1 - (2 * (p * foreground).sum((1, 2)) + 1) / ((p + foreground).sum((1, 2)) + 1)
    edge = F.binary_cross_entropy_with_logits(logits[:, 1], boundary, pos_weight=torch.tensor(3.0, device=logits.device))
    return bce + dice.mean() + 0.2 * edge, {'bce': bce.item(), 'dice': dice.mean().item(), 'edge': edge.item()}


@torch.no_grad()
def update_ema(ema: nn.Module, model: nn.Module, decay: float) -> None:
    for e, m in zip(ema.parameters(), model.parameters()):
        e.mul_(decay).add_(m.detach(), alpha=1 - decay)
    for e, m in zip(ema.buffers(), model.buffers()):
        e.copy_(m)


TTA = {
    'none': [(0, False)],
    'flip4': [(0, False), (0, True), (2, False), (2, True)],  # rot180 + h-flip == v-flip, so this is the flip group
    'd4': [(k, f) for k in range(4) for f in (False, True)],
}


@torch.no_grad()
def predict(model: nn.Module, image: np.ndarray, size: int, tta: str, device) -> tuple[np.ndarray, np.ndarray]:
    """Filament and boundary probabilities at 2048 x 2048 and at model size."""
    small = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    batch = []
    for k, flip in TTA[tta]:
        a = np.rot90(small, k)
        batch.append(to_input(np.ascontiguousarray(a[:, ::-1] if flip else a)))
    probabilities = np.zeros((2, size, size), np.float32)
    for start in range(0, len(batch), 2):
        x = torch.from_numpy(np.stack(batch[start:start + 2])).to(device)
        with torch.autocast('cuda', dtype=AMP):
            out = torch.sigmoid(model(x).float()).cpu().numpy()
        for (k, flip), p in zip(TTA[tta][start:start + 2], out):
            p = p[:, :, ::-1] if flip else p
            probabilities += np.rot90(p, -k, axes=(1, 2))
    probabilities /= len(batch)
    full = cv2.resize(probabilities[0], (2048, 2048), interpolation=cv2.INTER_LINEAR)
    return np.clip(full, 0, 1), probabilities[1]


def quick_dice(model, data, grouped, names, size, device) -> float:
    """Mean Dice of p >= 0.5 against the union of views, at model size (monitoring only)."""
    model.eval()
    scores = []
    for name in names:
        image = read_grayscale(data.train_dir / name)
        small = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(to_input(small)[None]).to(device)
        with torch.no_grad(), torch.autocast('cuda', dtype=AMP):
            p = torch.sigmoid(model(x)[0, 0].float()).cpu().numpy() >= 0.5
        union = np.zeros((2048, 2048), np.uint8)
        for view in grouped[name]:
            for annotation in view:
                union |= annotation_mask(annotation).astype(np.uint8)
        truth = cv2.resize(union.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA) >= 0.5
        denominator = p.sum() + truth.sum()
        scores.append(2 * (p & truth).sum() / denominator if denominator else 1.0)
    model.train()
    return float(np.mean(scores))


def train(args, data, grouped, fold: dict, device) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    names = sorted(fold['training_filenames'])[:args.limit or None]
    monitor = sorted(fold['validation_filenames'])[::6][:args.limit or None]
    model = GlobalUNet(encoder=args.encoder, pretrained=args.init is None).to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device)['ema'])
    ema = copy.deepcopy(model).eval()
    encoder = [p for n, p in model.named_parameters() if n.startswith('encoder.')]
    rest = [p for n, p in model.named_parameters() if not n.startswith('encoder.')]
    optimizer = torch.optim.AdamW([{'params': encoder, 'lr': args.lr / 5}, {'params': rest, 'lr': args.lr}],
                                  weight_decay=0.01)
    base = [g['lr'] for g in optimizer.param_groups]
    loader = DataLoader(DiskDataset(data, grouped, names, args.size), batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, drop_last=True, persistent_workers=False, pin_memory=True)
    steps_per_epoch = len(loader) // args.accumulate
    total = steps_per_epoch * args.epochs
    warmup = max(1, int(steps_per_epoch * args.warmup))
    scaler = torch.amp.GradScaler('cuda', enabled=AMP == torch.float16)  # no-op under bf16
    log = open(args.output.with_suffix('.jsonl'), 'a')
    step, started = 0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        sums, count = {'loss': 0.0, 'bce': 0.0, 'dice': 0.0, 'edge': 0.0}, 0
        optimizer.zero_grad(set_to_none=True)
        for index, (x, foreground, boundary) in enumerate(loader, 1):
            x, foreground, boundary = x.to(device, non_blocking=True), foreground.to(device), boundary.to(device)
            with torch.autocast('cuda', dtype=AMP):
                logits = model(x)
            loss, parts = loss_fn(logits, foreground, boundary)
            scaler.scale(loss / args.accumulate).backward()
            sums['loss'] += loss.item()
            for key, value in parts.items():
                sums[key] += value
            count += 1
            if index % args.accumulate == 0:
                scale = (step + 1) / warmup if step < warmup else \
                    0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))
                for group, lr in zip(optimizer.param_groups, base):
                    group['lr'] = lr * scale
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                update_ema(ema, model, args.ema)
                step += 1
        record = {'epoch': epoch, 'step': step, 'minutes': round((time.time() - started) / 60, 2),
                  **{k: round(v / max(count, 1), 5) for k, v in sums.items()},
                  'peak_gib': round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        if epoch % args.monitor_every == 0 or epoch == args.epochs:
            record['monitor_dice_ema'] = round(quick_dice(ema, data, grouped, monitor, args.size, device), 4)
        print(json.dumps(record), flush=True)
        log.write(json.dumps(record) + '\n')
        log.flush()
        torch.save({'ema': ema.state_dict(), 'args': {k: str(v) for k, v in vars(args).items()}, 'epoch': epoch},
                   args.output)


def predict_split(args, data, fold: dict | None, device) -> None:
    state = torch.load(args.output, map_location=device)
    model = GlobalUNet(encoder=state['args'].get('encoder', ENCODER), pretrained=False).to(device)
    model.load_state_dict(state['ema'])
    model.eval()
    if args.test:
        directory, names = data.test_dir, sorted(p.name for p in data.test_dir.glob('*.jp*g'))
    else:
        directory, names = data.train_dir, sorted(fold['validation_filenames'])
    root = Path(args.maps_dir)
    (root / 'probability').mkdir(parents=True, exist_ok=True)
    (root / 'boundary').mkdir(parents=True, exist_ok=True)
    started = time.time()
    for number, name in enumerate(names, 1):
        image = read_grayscale(directory / name)
        full, boundary = predict(model, image, args.size, args.tta, device)
        cv2.imwrite(str(root / 'probability' / f'{Path(name).stem}.png'), np.round(full * 65535).astype(np.uint16))
        cv2.imwrite(str(root / 'boundary' / f'{Path(name).stem}.png'), np.round(np.clip(boundary, 0, 1) * 255).astype(np.uint8))
        if number % 20 == 0 or number == len(names):
            print(f'{number}/{len(names)} images, {time.time() - started:.0f} s', flush=True)
    manifest = {'filenames': names, 'checkpoint': str(args.output),
                'checkpoint_sha256': hashlib.sha256(Path(args.output).read_bytes()).hexdigest(),
                'size': args.size, 'tta': args.tta, 'split': 'test' if args.test else f'fold {fold["fold"]} validation'}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/kagglehub/competitions/filament-segmentation-2026')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--folds-dir', default='results/annotation_round_folds')
    parser.add_argument('--all-data', action='store_true', help='train on all 707 images (no held-out fold)')
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--warmup', type=float, default=1.0, help='warmup length in epochs')
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--accumulate', type=int, default=2)
    parser.add_argument('--ema', type=float, default=0.99)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--encoder', default=ENCODER, help='timm model name with pretrained tag')
    parser.add_argument('--monitor-every', type=int, default=5)
    parser.add_argument('--limit', type=int, default=0, help='smoke test: use only this many images')
    parser.add_argument('--init', type=Path, help='start from this checkpoint (EMA weights)')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--predict', action='store_true', help='skip training; write maps from --output')
    parser.add_argument('--test', action='store_true', help='with --predict: the 180 test images')
    parser.add_argument('--maps-dir')
    parser.add_argument('--tta', default='flip4', choices=sorted(TTA))
    parser.add_argument('--amp', default='bf16', choices=('bf16', 'fp16'), help='fp16 for GPUs without bf16 (T4)')
    args = parser.parse_args()
    global AMP
    AMP = torch.float16 if args.amp == 'fp16' else torch.bfloat16
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    data = find_data_root(args.data)
    grouped = group_views(data.load_annotations())
    fold = None if args.all_data else json.loads((Path(args.folds_dir) / f'fold_{args.fold}.json').read_text())
    if args.all_data:
        fold = {'fold': 'all', 'training_filenames': sorted(grouped), 'validation_filenames': sorted(grouped)[::20]}
    if not args.predict:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        train(args, data, grouped, fold, device)
    if args.maps_dir:
        predict_split(args, data, fold, device)


if __name__ == '__main__':
    main()
