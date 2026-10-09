"""Evaluate a MONAI checkpoint on its patient-level held-out test split."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from monai.data import CacheDataset, DataLoader
from monai.inferers import SlidingWindowInferer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from monai_brats import (  # noqa: E402
    REGIONS,
    load_cases,
    load_checkpoint,
    make_transforms,
    probabilities_to_label_map,
    probabilities_to_region_masks,
    read_dataset_channel_order,
    read_dataset_label_region_map,
    select_device,
)
from profiler import profile_tumor  # noqa: E402
from test.eval_metrics import dice, hd95, iou, summarize_metric  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("runs/monai_brats/test_metrics"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-cases", type=int, default=0,
                        help="Limit evaluation cases for a short smoke test")
    parser.add_argument("--roi-size", type=int, nargs=3, default=(96, 96, 64))
    parser.add_argument("--save-example", action="store_true",
                        help="Save first held-out image, reference mask, prediction, and profile")
    return parser.parse_args()


def _load_test_cases(dataset_root: Path, split_path: Path):
    cases = load_cases(dataset_root)
    try:
        split = json.loads(split_path.read_text(encoding="utf-8"))
        test_ids = split["test"]
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        raise ValueError(f"Could not read test patient IDs from {split_path}") from exc
    case_by_id = {case["patient_id"]: case for case in cases}
    missing = sorted(set(test_ids) - case_by_id.keys())
    if missing:
        raise ValueError(f"Test split contains patients absent from dataset: {missing[:3]}")
    return [case_by_id[patient_id] for patient_id in test_ids]


def main():
    args = parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be strictly between 0 and 1")
    split_path = args.split or args.checkpoint.parent / "split.json"
    test_cases = _load_test_cases(args.dataset_root, split_path)
    if args.max_cases:
        if args.max_cases < 1:
            raise ValueError("--max-cases must be positive")
        test_cases = test_cases[:args.max_cases]
    if not test_cases:
        raise ValueError("The held-out test split is empty")

    device = select_device()
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    transform_roi = checkpoint.get("config", {}).get("roi_size", args.roi_size)
    inferer = SlidingWindowInferer(
        roi_size=tuple(int(size) for size in transform_roi),
        sw_batch_size=1,
        overlap=0.25,
        mode="gaussian",
    )
    source_channel_order = read_dataset_channel_order(args.dataset_root)
    label_region_map = read_dataset_label_region_map(args.dataset_root)
    _, evaluation_transform = make_transforms(
        transform_roi, source_channel_order, label_region_map
    )
    dataset = CacheDataset(test_cases, transform=evaluation_transform, cache_rate=0.0)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    example_dir = args.output / "example_case"
    if args.save_example:
        example_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    with torch.inference_mode():
        for index, batch in enumerate(loader, start=1):
            image = batch["image"].to(device)
            truth = batch["label"].cpu().numpy()[0] >= 0.5
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                probabilities = torch.sigmoid(inferer(image, model))[0].cpu().numpy()
            predicted = probabilities_to_region_masks(probabilities, args.threshold)
            for channel, region in enumerate(REGIONS):
                pred_region, gt_region = predicted[channel], truth[channel]
                rows.append({
                    "patient_id": test_cases[index - 1]["patient_id"],
                    "region": region,
                    "dice": dice(pred_region, gt_region),
                    "iou": iou(pred_region, gt_region),
                    "hd95_mm": hd95(pred_region, gt_region, (1.0, 1.0, 1.0)),
                })

            pred_labels = probabilities_to_label_map(probabilities, args.threshold)
            pred_wt = pred_labels > 0
            gt_wt = truth[0]
            pred_profile = profile_tumor(np.transpose(pred_wt, (2, 0, 1)))
            gt_profile = profile_tumor(np.transpose(gt_wt, (2, 0, 1)))
            gt_volume = gt_profile["total_volume_cm3"]
            volume_error = pred_profile["total_volume_cm3"] - gt_volume
            volume_error_pct = (
                0.0 if gt_volume == 0 and volume_error == 0
                else float("inf") if gt_volume == 0
                else abs(volume_error) / gt_volume * 100
            )
            profile_row = {
                "patient_id": test_cases[index - 1]["patient_id"],
                "region": "WT_profile",
                "predicted_volume_cm3": pred_profile["total_volume_cm3"],
                "ground_truth_volume_cm3": gt_volume,
                "absolute_volume_error_cm3": abs(volume_error),
                "absolute_volume_error_percent": volume_error_pct,
                "predicted_peak_area_cm2": pred_profile["max_cross_section_cm2"],
                "ground_truth_peak_area_cm2": gt_profile["max_cross_section_cm2"],
                "predicted_depth_span_mm": pred_profile["lesion_depth_span_mm"],
                "ground_truth_depth_span_mm": gt_profile["lesion_depth_span_mm"],
            }
            rows.append(profile_row)
            if args.save_example and index == 1:
                import nibabel as nib

                patient_id = test_cases[index - 1]["patient_id"]
                safe_id = Path(patient_id).name.replace(".nii.gz", "").replace(".nii", "")
                affine = batch["image"].affine.detach().cpu().numpy()[0]
                normalized_image = batch["image"].cpu().numpy()[0]
                source_labels = np.zeros(truth.shape[1:], dtype=np.uint8)
                class_by_region = {
                    tuple(bool(value) for value in signature): int(label_id)
                    for label_id, signature in label_region_map.items()
                }
                for signature in (
                    (True, False, False),
                    (True, True, False),
                    (True, True, True),
                ):
                    source_labels[truth[0] & (
                        truth[1] if signature[1] else ~truth[1]
                    ) & (
                        truth[2] if signature[2] else ~truth[2]
                    )] = class_by_region[signature]
                nib.save(
                    nib.Nifti1Image(
                        np.moveaxis(normalized_image, 0, -1).astype(np.float32),
                        affine,
                    ),
                    example_dir / f"{safe_id}_image_4channel.nii.gz",
                )
                nib.save(
                    nib.Nifti1Image(source_labels, affine),
                    example_dir / f"{safe_id}_reference_msd_labels.nii.gz",
                )
                nib.save(
                    nib.Nifti1Image(pred_labels.astype(np.uint8), affine),
                    example_dir / f"{safe_id}_prediction_brats_labels.nii.gz",
                )
                case_metrics = {}
                for row in rows:
                    if row["patient_id"] != patient_id or row["region"] not in REGIONS:
                        continue
                    region_metrics = {}
                    for key in ("dice", "iou", "hd95_mm"):
                        value = float(row[key])
                        region_metrics[key] = value if np.isfinite(value) else None
                    region_metrics["hd95_non_finite"] = not np.isfinite(
                        float(row["hd95_mm"])
                    )
                    case_metrics[row["region"]] = region_metrics
                example_report = {
                    "patient_id": patient_id,
                    "checkpoint": str(args.checkpoint.resolve()),
                    "input_channel_order": ["FLAIR", "T1", "T1Gd", "T2"],
                    "reference_label_encoding": "MSD Task01 (1 edema, 2 non-enhancing, 3 enhancing)",
                    "prediction_label_encoding": "BraTS style (1 core, 2 edema, 4 enhancing)",
                    "region_metrics": case_metrics,
                    "whole_tumor_profile": {
                        "prediction": pred_profile,
                        "reference": gt_profile,
                        "absolute_volume_error_cm3": abs(volume_error),
                        "absolute_volume_error_percent": volume_error_pct,
                    },
                    "disclaimer": "Experimental research output; not a radiologist report or a clinical result.",
                }
                (example_dir / f"{safe_id}_report.json").write_text(
                    json.dumps(example_report, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8",
                )
            print(f"Evaluated held-out patient {index}/{len(test_cases)}")

    metric_fields = ["patient_id", "region", "dice", "iou", "hd95_mm"]
    profile_fields = [
        "patient_id", "region", "predicted_volume_cm3", "ground_truth_volume_cm3",
        "absolute_volume_error_cm3", "absolute_volume_error_percent",
        "predicted_peak_area_cm2", "ground_truth_peak_area_cm2",
        "predicted_depth_span_mm", "ground_truth_depth_span_mm",
    ]
    with (args.output / "per_case_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=metric_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(row for row in rows if row["region"] in REGIONS)
    with (args.output / "profile_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=profile_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(row for row in rows if row["region"] == "WT_profile")

    summary = {}
    for region in REGIONS:
        selected = [row for row in rows if row["region"] == region]
        summary[region] = {
            name: summarize_metric([row[name] for row in selected])
            for name in ("dice", "iou", "hd95_mm")
        }
    summary["hd95_note"] = (
        "Mean and standard deviation use finite HD95 cases only. "
        "Non-finite cases are counted explicitly; HD95 is infinite when exactly "
        "one of prediction and reference is empty."
    )
    summary["evaluation"] = {
        "checkpoint": str(args.checkpoint.resolve()),
        "split_manifest": str(split_path.resolve()),
        "held_out_patient_count": len(test_cases),
        "threshold": args.threshold,
        "device": str(device),
        "is_full_test_split": not bool(args.max_cases),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
