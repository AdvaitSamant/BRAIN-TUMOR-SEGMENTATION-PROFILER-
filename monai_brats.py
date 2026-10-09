"""Shared MONAI components for the MSD Brain Tumour task."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# Network channels use a stable canonical order, also used by the MSD release.
MODALITIES = ("FLAIR", "T1", "T1Gd", "T2")
REGIONS = ("WT", "TC", "ET")
DEFAULT_ROI_SIZE = (96, 96, 64)
DEFAULT_BRATS_LABEL_REGION_MAP = {
    1: (True, True, False),
    2: (True, False, False),
    4: (True, True, True),
}


def load_cases(dataset_root: str | Path) -> list[dict[str, Any]]:
    """Read and validate the MSD training manifest."""
    root = Path(dataset_root).resolve()
    manifest = root / "dataset.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"MSD manifest not found: {manifest}")
    with manifest.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    entries = metadata.get("training")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{manifest} has no non-empty 'training' list")

    cases = []
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict) or "image" not in entry or "label" not in entry:
            raise ValueError("Each training item must have 'image' and 'label' paths")
        image_rel = entry["image"]
        if isinstance(image_rel, list):
            if len(image_rel) != 4:
                raise ValueError("Expected four MRI modality paths per case")
            image_path = [str((root / item).resolve()) for item in image_rel]
        else:
            image_path = str((root / image_rel).resolve())
        label_path = str((root / entry["label"]).resolve())
        patient_id = Path(entry["label"]).name
        if patient_id in seen:
            raise ValueError(f"Duplicate patient in dataset manifest: {patient_id}")
        seen.add(patient_id)
        paths = image_path if isinstance(image_path, list) else [image_path]
        missing = [path for path in [*paths, label_path] if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"Dataset file is missing: {missing[0]}")
        cases.append({
            "image": image_path,
            "label": label_path,
            "patient_id": patient_id,
        })
    return cases


def read_dataset_channel_order(dataset_root: str | Path) -> tuple[str, ...]:
    """Read channel names from the MSD manifest and normalize common spellings."""
    manifest = Path(dataset_root).resolve() / "dataset.json"
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    modality_map = metadata.get("modality")
    if not isinstance(modality_map, dict) or not modality_map:
        raise ValueError(
            f"{manifest} must define a 'modality' mapping to identify the 4 input channels"
        )

    def canonical_name(value: str) -> str:
        name = str(value).strip().lower().replace(" ", "")
        if name in {"flair"}:
            return "FLAIR"
        if name in {"t1", "t1w"}:
            return "T1"
        if name in {"t1gd", "t1ce", "t1gdce", "t1gdenhanced"}:
            return "T1Gd"
        if name in {"t2", "t2w"}:
            return "T2"
        raise ValueError(f"Unsupported input modality in {manifest}: {value!r}")

    order = tuple(
        canonical_name(modality_map[key])
        for key in sorted(modality_map, key=lambda item: int(item))
    )
    if len(order) != 4 or set(order) != set(MODALITIES):
        raise ValueError(
            f"Expected exactly one each of {MODALITIES} in {manifest}; got {order}"
        )
    return order


def read_dataset_label_region_map(dataset_root: str | Path) -> dict[int, tuple[bool, ...]]:
    """Interpret label IDs from the dataset's own label descriptions."""
    manifest = Path(dataset_root).resolve() / "dataset.json"
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    labels = metadata.get("labels")
    if not isinstance(labels, dict) or not labels:
        raise ValueError(f"{manifest} must define a 'labels' mapping")
    region_map = {}
    for raw_id, description in labels.items():
        label_id = int(raw_id)
        if label_id == 0:
            continue
        name = str(description).strip().lower()
        if "edema" in name:
            region_map[label_id] = (True, False, False)
        elif "non-enhanc" in name or "nonenhanc" in name or "necrot" in name:
            region_map[label_id] = (True, True, False)
        elif "enhanc" in name:
            region_map[label_id] = (True, True, True)
        else:
            raise ValueError(
                f"Cannot map tumor label {label_id} ({description!r}) from {manifest}"
            )
    signatures = set(region_map.values())
    if len(region_map) != 3 or signatures != {
        (True, False, False),
        (True, True, False),
        (True, True, True),
    }:
        raise ValueError(
            f"Expected edema, non-enhancing tumor, and enhancing tumor labels in {manifest}"
        )
    return region_map


def patient_split(
    cases: Sequence[dict[str, Any]],
    seed: int = 42,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> dict[str, list[dict[str, Any]]]:
    """Make a deterministic, disjoint split at the case/patient level."""
    if len(cases) < 3:
        raise ValueError("At least three labeled patients are required for train/val/test")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split fractions must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must sum to less than 1")

    shuffled = list(cases)
    random.Random(seed).shuffle(shuffled)
    n_train = max(1, int(len(shuffled) * train_fraction))
    n_validation = max(1, int(len(shuffled) * validation_fraction))
    if n_train + n_validation >= len(shuffled):
        n_train = len(shuffled) - 2
        n_validation = 1
    result = {
        "train": shuffled[:n_train],
        "validation": shuffled[n_train:n_train + n_validation],
        "test": shuffled[n_train + n_validation:],
    }
    patient_ids = [
        case["patient_id"] for subset in result.values() for case in subset
    ]
    if len(patient_ids) != len(set(patient_ids)):
        raise ValueError("Patient leakage detected between data splits")
    return result


def save_split(path: str | Path, split: dict[str, list[dict[str, Any]]]) -> None:
    """Write a portable split manifest using dataset-relative image paths."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        key: [case["patient_id"] for case in values]
        for key, values in split.items()
    }
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


class ReorderModalityChannelsd:
    def __init__(self, channel_order: Sequence[str]):
        self.indices = tuple(channel_order.index(name) for name in MODALITIES)

    def __call__(self, data):
        item = dict(data)
        image = item["image"]
        if image.shape[0] != 4:
            raise ValueError(f"Expected four input channels, got shape {image.shape}")
        item["image"] = image[list(self.indices)]
        return item


class ConvertBraTSRegionsd:
    def __init__(self, label_region_map=DEFAULT_BRATS_LABEL_REGION_MAP):
        self.label_region_map = {
            int(label_id): tuple(bool(value) for value in regions)
            for label_id, regions in label_region_map.items()
        }

    def __call__(self, data):
        item = dict(data)
        label = np.asarray(item["label"])
        if label.ndim == 4 and label.shape[0] == 1:
            label = label[0]
        values = set(np.unique(label).astype(int).tolist()) - {0}
        unknown = values - self.label_region_map.keys()
        if unknown:
            raise ValueError(f"Unexpected tumor label values: {sorted(unknown)}")
        item["label"] = np.stack(
            [
                np.isin(label, [
                    label_id for label_id, regions in self.label_region_map.items()
                    if regions[channel]
                ])
                for channel in range(len(REGIONS))
            ],
            axis=0,
        ).astype(np.float32)
        return item


def make_transforms(
    roi_size: Sequence[int] = DEFAULT_ROI_SIZE,
    source_channel_order: Sequence[str] = MODALITIES,
    label_region_map=DEFAULT_BRATS_LABEL_REGION_MAP,
):
    """Build deterministic preprocessing and random training augmentation."""
    from monai.transforms import (
        Compose,
        EnsureChannelFirstd,
        EnsureTyped,
        LoadImaged,
        NormalizeIntensityd,
        Orientationd,
        RandCropByPosNegLabeld,
        RandFlipd,
        Spacingd,
    )

    deterministic = [
        LoadImaged(keys=("image", "label"), image_only=True),
        EnsureChannelFirstd(keys="image", channel_dim=-1),
        EnsureChannelFirstd(keys="label", channel_dim="no_channel"),
        ReorderModalityChannelsd(source_channel_order),
        Orientationd(keys=("image", "label"), axcodes="RAS", labels=None),
        Spacingd(
            keys=("image", "label"),
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=("image", "label")),
    ]
    validation = Compose(
        [*deterministic, ConvertBraTSRegionsd(label_region_map)]
    )
    training = Compose(
        [
            *deterministic,
            RandCropByPosNegLabeld(
                keys=("image", "label"),
                label_key="label",
                spatial_size=tuple(int(size) for size in roi_size),
                pos=1,
                neg=1,
                num_samples=1,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=("image", "label"), prob=0.5, spatial_axis=0),
            RandFlipd(keys=("image", "label"), prob=0.5, spatial_axis=1),
            RandFlipd(keys=("image", "label"), prob=0.5, spatial_axis=2),
            ConvertBraTSRegionsd(label_region_map),
        ]
    )
    return training, validation


def build_model():
    """Build the three-output-channel 3D SegResNet used by training/inference."""
    from monai.networks.nets import SegResNet

    return SegResNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        init_filters=16,
        dropout_prob=0.2,
    )


def load_checkpoint(path: str | Path, device):
    """Load a project checkpoint and return an initialized model."""
    import torch

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"MONAI checkpoint not found: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location=device)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError("Checkpoint does not contain a MONAI model_state_dict")
    model = build_model().to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, payload


def brats_regions(label_map: np.ndarray) -> dict[str, np.ndarray]:
    """Convert standard BraTS labels (0/1/2/4) to nested region masks."""
    return labels_to_regions(label_map, DEFAULT_BRATS_LABEL_REGION_MAP)


def probabilities_to_region_masks(
    probabilities: np.ndarray, threshold: float = 0.5
) -> np.ndarray:
    """Threshold and nest predictions exactly as the exported label mask does."""
    label_map = probabilities_to_label_map(probabilities, threshold)
    regions = brats_regions(label_map)
    return np.stack([regions[name] for name in REGIONS], axis=0)


def labels_to_regions(
    label_map: np.ndarray, label_region_map
) -> dict[str, np.ndarray]:
    """Convert source label IDs to WT/TC/ET masks using explicit class semantics."""
    labels = np.asarray(label_map)
    mapping = {
        int(label_id): tuple(bool(value) for value in regions)
        for label_id, regions in label_region_map.items()
    }
    values = set(np.unique(labels).astype(int).tolist()) - {0}
    unknown = values - mapping.keys()
    if unknown:
        raise ValueError(f"Unexpected tumor label values: {sorted(unknown)}")
    return {
        region: np.isin(
            labels,
            [
                label_id for label_id, flags in mapping.items()
                if flags[index]
            ],
        )
        for index, region in enumerate(REGIONS)
    }


def probabilities_to_label_map(
    probabilities: np.ndarray, threshold: float = 0.5
) -> np.ndarray:
    """Convert nested [WT, TC, ET] probabilities to 0/1/2/4 labels."""
    probs = np.asarray(probabilities)
    if probs.ndim != 4 or probs.shape[0] != 3:
        raise ValueError(f"Expected probabilities shaped (3, X, Y, Z), got {probs.shape}")
    wt, tc, et = probs >= threshold
    tc |= et
    wt |= tc
    labels = np.zeros(wt.shape, dtype=np.uint8)
    labels[wt] = 2
    labels[tc] = 1
    labels[et] = 4
    return labels


def select_device():
    """Prefer an available CUDA device and otherwise use CPU."""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
