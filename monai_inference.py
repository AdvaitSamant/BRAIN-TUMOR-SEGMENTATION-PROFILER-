"""Four-modality MONAI inference using the training preprocessing pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from monai.data import Dataset
from monai.inferers import SlidingWindowInferer
from monai.transforms import (
    Compose,
    ConcatItemsd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
)

from monai_brats import (
    DEFAULT_BRATS_LABEL_REGION_MAP,
    MODALITIES,
    ReorderModalityChannelsd,
    labels_to_regions,
    load_checkpoint,
    probabilities_to_label_map,
    select_device,
)


def _validate_aligned_niftis(paths: dict[str, str], reference_path: str | None) -> None:
    import nibabel as nib

    all_paths = [paths[name] for name in MODALITIES]
    if reference_path:
        all_paths.append(reference_path)
    reference = nib.load(all_paths[0])
    if len(reference.shape) != 3:
        raise ValueError(f"Expected 3D NIfTI modality, got shape {reference.shape}")
    for path in all_paths[1:]:
        image = nib.load(path)
        if len(image.shape) != 3:
            raise ValueError(f"Expected 3D NIfTI modality, got shape {image.shape}")
        if image.shape[:3] != reference.shape[:3] or not np.allclose(
            image.affine, reference.affine, atol=1e-3
        ):
            raise ValueError(
                "All MRI modalities and the optional label must have matching "
                "spatial shapes and affines (co-registered NIfTI volumes)."
            )


def _inference_transform(include_label: bool):
    keys = [*MODALITIES, *(["label"] if include_label else [])]
    modes = ["bilinear"] * len(MODALITIES)
    if include_label:
        modes.append("nearest")
    transforms = [
        LoadImaged(keys=keys, image_only=True),
        EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
        Orientationd(keys=keys, axcodes="RAS", labels=None),
        Spacingd(keys=keys, pixdim=(1.0, 1.0, 1.0), mode=tuple(modes)),
        NormalizeIntensityd(keys=MODALITIES, nonzero=True, channel_wise=True),
        ConcatItemsd(keys=MODALITIES, name="image", dim=0),
        EnsureTyped(keys=("image",)),
    ]
    if include_label:
        transforms.append(EnsureTyped(keys=("label",)))
    return Compose(transforms)


def _combined_image_transform(include_label: bool, source_channel_order):
    keys = ["image", *(["label"] if include_label else [])]
    modes = ("bilinear", "nearest") if include_label else ("bilinear",)
    transforms = [
        LoadImaged(keys=keys, image_only=True),
        EnsureChannelFirstd(keys="image", channel_dim=-1),
    ]
    if include_label:
        transforms.append(EnsureChannelFirstd(keys="label", channel_dim="no_channel"))
    transforms.extend(
        [
            ReorderModalityChannelsd(source_channel_order),
            Orientationd(keys=keys, axcodes="RAS", labels=None),
            Spacingd(keys=keys, pixdim=(1.0, 1.0, 1.0), mode=modes),
            NormalizeIntensityd(keys=("image",), nonzero=True, channel_wise=True),
            EnsureTyped(keys=keys),
        ]
    )
    return Compose(transforms)


def predict_case(
    modality_paths: dict[str, str] | None,
    checkpoint_path: str | Path,
    threshold: float = 0.5,
    reference_label_path: str | None = None,
    combined_path: str | None = None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Predict WT/TC/ET; output is RAS at 1 mm in a NIfTI affine grid."""
    if (modality_paths is None) == (combined_path is None):
        raise ValueError("Provide either four modality paths or one combined 4D image")
    if modality_paths is not None and set(modality_paths) != set(MODALITIES):
        raise ValueError(f"Expected paths for exactly these modalities: {MODALITIES}")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between 0 and 1")
    if modality_paths is not None:
        _validate_aligned_niftis(modality_paths, reference_label_path)
    else:
        import nibabel as nib

        combined = nib.load(combined_path)
        if len(combined.shape) != 4 or combined.shape[-1] != 4:
            raise ValueError(
                f"Expected a 4D NIfTI with four channels on the last axis, got {combined.shape}"
            )
        if reference_label_path:
            label = nib.load(reference_label_path)
            if len(label.shape) != 3 or label.shape != combined.shape[:3] or not np.allclose(
                label.affine, combined.affine, atol=1e-3
            ):
                raise ValueError("Reference label must be aligned to the combined NIfTI image")

    device = torch.device(device) if device is not None else select_device()
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA inference was requested but is unavailable: {device}")
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    config = checkpoint.get("config", {})
    source_channel_order = config.get("source_channel_order", MODALITIES)
    if modality_paths is not None:
        case = {name: str(modality_paths[name]) for name in MODALITIES}
        transform = _inference_transform(reference_label_path is not None)
    else:
        case = {"image": str(combined_path)}
        transform = _combined_image_transform(
            reference_label_path is not None, source_channel_order
        )
    if reference_label_path:
        case["label"] = str(reference_label_path)
    dataset = Dataset(
        data=[case],
        transform=transform,
    )
    processed = dataset[0]
    roi_size = tuple(config.get("roi_size", (96, 96, 64)))
    inferer = SlidingWindowInferer(
        roi_size=roi_size,
        sw_batch_size=1,
        overlap=0.25,
        mode="gaussian",
    )
    image = processed["image"].unsqueeze(0).to(device)
    model.eval()
    with torch.inference_mode():
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            probabilities = torch.sigmoid(inferer(image, model))[0].cpu().numpy()
    labels = probabilities_to_label_map(probabilities, threshold)
    flair = processed["image"].as_tensor().cpu().numpy()[0]
    affine = processed["image"].affine.detach().cpu().numpy()
    result = {
        "probabilities": probabilities,
        "labels": labels,
        "flair": flair,
        "affine": affine,
        "spacing_mm": (1.0, 1.0, 1.0),
        "device": str(device),
        "checkpoint": str(checkpoint_path),
    }
    if reference_label_path:
        label = processed["label"].as_tensor().cpu().numpy()[0]
        reference_labels = np.rint(label).astype(np.uint8)
        label_map = config.get("label_region_map", DEFAULT_BRATS_LABEL_REGION_MAP)
        result["reference_labels"] = reference_labels
        result["reference_regions"] = labels_to_regions(reference_labels, label_map)
    return result
