"""Train a 3D SegResNet on MSD Task01 BrainTumour."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from monai.data import CacheDataset, DataLoader, list_data_collate
from monai.inferers import SlidingWindowInferer
from monai.utils import set_determinism

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from monai_brats import (
    DEFAULT_ROI_SIZE,
    MODALITIES,
    REGIONS,
    build_model,
    load_cases,
    make_transforms,
    patient_split,
    probabilities_to_region_masks,
    read_dataset_channel_order,
    read_dataset_label_region_map,
    save_split,
    select_device,
)


def _case_dice(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    scores = []
    for channel in range(len(REGIONS)):
        pred, truth = prediction[channel], target[channel]
        denominator = int(pred.sum()) + int(truth.sum())
        scores.append(
            1.0 if denominator == 0
            else float(2 * np.logical_and(pred, truth).sum() / denominator)
        )
    return np.asarray(scores, dtype=np.float64)


def validate(model, loader, inferer, device, amp_enabled: bool) -> dict[str, float]:
    """Run full-volume sliding-window validation and return per-region Dice."""
    per_case = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["label"].cpu().numpy()[0] >= 0.5
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = inferer(image, model)
            probabilities = torch.sigmoid(logits)[0].cpu().numpy()
            predicted = probabilities_to_region_masks(probabilities, 0.5)
            per_case.append(_case_dice(predicted, target))
    if not per_case:
        raise RuntimeError("Validation split produced no cases")
    mean_scores = np.mean(per_case, axis=0)
    return {name: float(value) for name, value in zip(REGIONS, mean_scores)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/monai_brats"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-rate", type=float, default=0.0)
    parser.add_argument("--max-cases", type=int, default=0,
                        help="Limit to a deterministic subset for a smoke test")
    parser.add_argument("--roi-size", type=int, nargs=3, default=DEFAULT_ROI_SIZE,
                        metavar=("X", "Y", "Z"))
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable CUDA mixed-precision training/inference")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs < 1 or args.num_workers < 0:
        raise ValueError("--epochs must be positive and --num-workers non-negative")
    if args.max_cases and args.max_cases < 3:
        raise ValueError("--max-cases must be at least 3 to create train/val/test splits")
    if not 0 <= args.cache_rate <= 1:
        raise ValueError("--cache-rate must be between 0 and 1")
    if any(size < 16 or size % 16 for size in args.roi_size):
        raise ValueError("Each --roi-size dimension must be a positive multiple of 16")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_determinism(seed=args.seed)
    cases = load_cases(args.dataset_root)
    if args.max_cases:
        cases = cases[:args.max_cases]
    split = patient_split(cases, seed=args.seed)
    save_split(args.output_dir / "split.json", split)

    device = select_device()
    amp_enabled = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        print(f"Training on {torch.cuda.get_device_name(device)} with AMP={amp_enabled}")
    else:
        print("CUDA unavailable; training on CPU.")

    source_channel_order = read_dataset_channel_order(args.dataset_root)
    label_region_map = read_dataset_label_region_map(args.dataset_root)
    train_transform, evaluation_transform = make_transforms(
        args.roi_size, source_channel_order, label_region_map
    )
    train_dataset = CacheDataset(
        data=split["train"],
        transform=train_transform,
        cache_rate=args.cache_rate,
        num_workers=args.num_workers,
    )
    validation_dataset = CacheDataset(
        data=split["validation"],
        transform=evaluation_transform,
        cache_rate=args.cache_rate,
        num_workers=args.num_workers,
    )
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
        collate_fn=list_data_collate,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )

    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    inferer = SlidingWindowInferer(
        roi_size=tuple(args.roi_size),
        sw_batch_size=1,
        overlap=0.25,
        mode="gaussian",
    )
    from monai.losses import DiceLoss

    dice_loss = DiceLoss(sigmoid=True, smooth_nr=1e-5, smooth_dr=1e-5)
    bce_loss = torch.nn.BCEWithLogitsLoss()

    best_wt_dice = -1.0
    history_path = args.output_dir / "history.csv"
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        started = time.perf_counter()
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["label"].to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(image)
                loss = dice_loss(logits, target) + bce_loss(logits, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach().cpu())

        validation = validate(
            model, validation_loader, inferer, device, amp_enabled
        )
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(len(train_loader), 1),
            "validation_dice_WT": validation["WT"],
            "validation_dice_TC": validation["TC"],
            "validation_dice_ET": validation["ET"],
            "epoch_seconds": time.perf_counter() - started,
        }
        exists = history_path.exists()
        with history_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=row.keys())
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"loss={row['train_loss']:.4f} "
            f"Dice WT/TC/ET={validation['WT']:.4f}/"
            f"{validation['TC']:.4f}/{validation['ET']:.4f} "
            f"({row['epoch_seconds']:.1f}s)"
        )
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "best_validation_dice_wt": max(best_wt_dice, validation["WT"]),
            "config": {
                "architecture": "SegResNet",
                "input_modalities": list(MODALITIES),
                "source_channel_order": list(source_channel_order),
                "label_region_map": {
                    str(label_id): list(regions)
                    for label_id, regions in label_region_map.items()
                },
                "regions": list(REGIONS),
                "roi_size": list(args.roi_size),
                "target_spacing_mm": [1.0, 1.0, 1.0],
                "intensity_normalization": "nonzero, channel-wise",
                "seed": args.seed,
                "split_manifest": "split.json",
            },
        }
        if validation["WT"] > best_wt_dice:
            best_wt_dice = validation["WT"]
            temporary_checkpoint = args.output_dir / "best_model.tmp.pt"
            torch.save(checkpoint, temporary_checkpoint)
            temporary_checkpoint.replace(args.output_dir / "best_model.pt")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metadata = {
        "device": str(device),
        "torch_version": torch.__version__,
        "monai_version": __import__("monai").__version__,
        "epochs": args.epochs,
        "best_validation_dice_wt": best_wt_dice,
        "input_channel_order": list(MODALITIES),
        "source_channel_order": list(source_channel_order),
        "label_region_map": {
            str(label_id): list(regions)
            for label_id, regions in label_region_map.items()
        },
        "case_counts": {name: len(items) for name, items in split.items()},
        "smoke_test_limited_cases": bool(args.max_cases),
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Best checkpoint: {args.output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
