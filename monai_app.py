"""Streamlit UI for the trained four-modality MONAI model."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import streamlit as st
import torch

from monai_brats import (
    MODALITIES,
    brats_regions,
    load_cases,
    read_dataset_channel_order,
)
from monai_inference import predict_case
from profiler import profile_tumor
from test.eval_metrics import dice, hd95, iou

PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT = PROJECT_ROOT / "runs" / "monai_brats" / "best_model.pt"
DATASET_ROOT = PROJECT_ROOT / "data" / "Task01_BrainTumour"
SPLIT_MANIFEST = CHECKPOINT.parent / "split.json"
INFERENCE_DEVICE = torch.device("cuda:0")

st.set_page_config(page_title="MONAI Brain Tumour Profiler", layout="wide")
st.title("🧠 3D Brain Tumour Segmentation")
st.subheader("MONAI Deep Learning Inference Profiler", divider=True)
st.info(
    "**Educational research software only.** Masks and measurements are not a "
    "radiologist's report and must not be used for diagnosis or treatment."
)

with st.sidebar:
    st.header("Model and inputs")
    st.caption(f"Checkpoint: {CHECKPOINT}")
    input_format = st.radio(
        "MRI input format",
        ("MSD 4-channel NIfTI", "Four separate modalities"),
        help=(
            "MSD 4-channel: one 4D NIfTI containing all four MRI sequences. "
            "Separate modalities: four 3D NIfTIs, one each for FLAIR, T1, T1Gd, "
            "and T2; they must be from the same scan and co-registered."
        ),
    )
    if input_format == "MSD 4-channel NIfTI":
        st.caption(
            "Choose this for the MSD-style file: one 4D NIfTI with FLAIR, T1, "
            "T1Gd, and T2 stored as channels."
        )
    else:
        st.caption(
            "Choose this when you have four 3D NIfTI files. Upload all four "
            "sequences from the same patient; matching dimensions alone do not "
            "guarantee that scans are aligned."
        )
    uploads = {}
    if input_format == "MSD 4-channel NIfTI":
        uploads["combined"] = st.file_uploader(
            "Four-channel MRI (.nii / .nii.gz)", type=("nii", "gz"), key="combined"
        )
    else:
        for modality in MODALITIES:
            uploads[modality] = st.file_uploader(
                f"{modality} (.nii / .nii.gz)",
                type=("nii", "gz"),
                key=f"modality_{modality}",
            )
    reference_upload = st.file_uploader(
        "Optional aligned ground-truth label (.nii / .nii.gz)",
        type=("nii", "gz"),
        key="reference",
    )
    threshold = st.slider("Region threshold", 0.1, 0.9, 0.5, 0.05)
    if st.button("Reset session and clear results"):
        st.session_state.clear()
        st.rerun()

if not CHECKPOINT.is_file():
    st.error(
        "No trained MONAI checkpoint was found. Train the model with "
        "`python scripts/train_monai.py` to create "
        "`runs/monai_brats/best_model.pt`."
    )
    st.stop()

if not torch.cuda.is_available():
    st.error(
        "GPU inference is required on this page, but CUDA is unavailable. "
        "Start Streamlit with the CUDA-enabled PyTorch environment and verify "
        "`torch.cuda.is_available()` before running inference."
    )
    st.stop()

required_uploads = [uploads.get("combined")] if "combined" in uploads else [
    uploads.get(name) for name in MODALITIES
]
uploads_ready = all(required_uploads)
input_bytes = (
    {key: item.getvalue() for key, item in uploads.items() if item is not None}
    if uploads_ready else {}
)
reference_bytes = reference_upload.getvalue() if reference_upload else None
checkpoint_stat = CHECKPOINT.stat()
checkpoint_key = f"{checkpoint_stat.st_size}:{checkpoint_stat.st_mtime_ns}"

upload_digest = hashlib.sha256()
for key in sorted(input_bytes):
    upload_digest.update(key.encode("utf-8"))
    upload_digest.update(input_bytes[key])
if reference_bytes:
    upload_digest.update(reference_bytes)
upload_digest.update(str(threshold).encode("ascii"))
upload_digest.update(checkpoint_key.encode("ascii"))
upload_case_key = upload_digest.hexdigest()

sample_digest = hashlib.sha256()
sample_digest.update(str(threshold).encode("ascii"))
sample_digest.update(checkpoint_key.encode("ascii"))
sample_digest.update(str(SPLIT_MANIFEST).encode("utf-8"))
sample_case_key = sample_digest.hexdigest()

sample_column, upload_column = st.columns(2)
run_sample = sample_column.button("Run held-out sample case", type="primary")
run_upload = upload_column.button(
    "Run uploaded scan",
    disabled=not uploads_ready,
    help="Upload the selected input format in the sidebar first.",
)

if run_sample:
    try:
        if not SPLIT_MANIFEST.is_file():
            raise FileNotFoundError(
                f"Training split manifest not found: {SPLIT_MANIFEST}. "
                "Train the model first to generate it."
            )
        if not DATASET_ROOT.is_dir():
            raise FileNotFoundError(
                f"MSD dataset not found at {DATASET_ROOT}. "
                "Place/extract Task01_BrainTumour there to run the sample."
            )
        split = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
        test_ids = split.get("test", [])
        if not test_ids:
            raise ValueError(f"No held-out test patients listed in {SPLIT_MANIFEST}")
        sample_id = test_ids[0]
        sample_case = next(
            (
                case for case in load_cases(DATASET_ROOT)
                if case["patient_id"] == sample_id
            ),
            None,
        )
        if sample_case is None:
            raise ValueError(f"Could not find held-out sample patient {sample_id}")
        if isinstance(sample_case["image"], list):
            channel_order = read_dataset_channel_order(DATASET_ROOT)
            modality_paths = dict(zip(channel_order, sample_case["image"]))
            result = predict_case(
                modality_paths,
                CHECKPOINT,
                threshold,
                reference_label_path=sample_case["label"],
                device=INFERENCE_DEVICE,
            )
        else:
            result = predict_case(
                None,
                CHECKPOINT,
                threshold,
                reference_label_path=sample_case["label"],
                combined_path=sample_case["image"],
                device=INFERENCE_DEVICE,
            )
        result["sample_patient_id"] = sample_id
        st.session_state["monai_result"] = result
        st.session_state["monai_case_key"] = sample_case_key
        st.session_state["monai_input_source"] = "sample"
    except Exception as exc:
        st.error(f"Sample inference failed: {exc}")

if run_upload:
    try:
        with tempfile.TemporaryDirectory(prefix="monai_brain_tumor_") as tmp_dir:
            temp_root = Path(tmp_dir)

            def write_upload(upload, name):
                suffix = ".nii.gz" if upload.name.lower().endswith(".nii.gz") else ".nii"
                path = temp_root / f"{name}{suffix}"
                path.write_bytes(upload.getvalue())
                return str(path)

            reference_path = (
                write_upload(reference_upload, "reference")
                if reference_upload is not None else None
            )
            if "combined" in input_bytes:
                combined_path = write_upload(uploads["combined"], "combined")
                result = predict_case(
                    None,
                    CHECKPOINT,
                    threshold,
                    reference_label_path=reference_path,
                    combined_path=combined_path,
                    device=INFERENCE_DEVICE,
                )
            else:
                modality_paths = {
                    name: write_upload(uploads[name], name)
                    for name in MODALITIES
                }
                result = predict_case(
                    modality_paths,
                    CHECKPOINT,
                    threshold,
                    reference_label_path=reference_path,
                    device=INFERENCE_DEVICE,
                )
        st.session_state["monai_result"] = result
        st.session_state["monai_case_key"] = upload_case_key
        st.session_state["monai_input_source"] = "upload"
    except Exception as exc:
        st.error(f"MONAI inference failed: {exc}")

result = st.session_state.get("monai_result")
active_input_source = st.session_state.get("monai_input_source")
active_case_key = (
    sample_case_key if active_input_source == "sample" else upload_case_key
)
if result is None or st.session_state.get("monai_case_key") != active_case_key:
    st.info(
        "Run the held-out sample case, or upload a scan and choose "
        "'Run uploaded scan' to see the predicted mask and profile."
    )
    st.stop()

labels = result["labels"]
flair = result["flair"]
spacing = result["spacing_mm"]
profile = profile_tumor(
    np.transpose(labels > 0, (2, 1, 0)),
    voxel_spacing=(spacing[2], spacing[1], spacing[0]),
)
st.success(f"Model loaded; inference device: {result['device']}")
st.caption(f"GPU: {torch.cuda.get_device_name(INFERENCE_DEVICE)}")
if "sample_patient_id" in result:
    st.caption(
        f"Sample case {result['sample_patient_id']} is from the held-out test "
        "split and includes its reference mask for comparison."
    )
st.caption(
    "Prediction grid: RAS orientation, 1 mm isotropic voxels. "
    "The output NIfTI uses this transformed grid and affine."
)

if "reference_labels" in result:
    predicted_regions = brats_regions(labels)
    reference_regions = result["reference_regions"]
    metric_columns = st.columns(3)
    for column, region in zip(metric_columns, ("WT", "TC", "ET")):
        metrics = (
            dice(predicted_regions[region], reference_regions[region]),
            iou(predicted_regions[region], reference_regions[region]),
            hd95(predicted_regions[region], reference_regions[region], spacing),
        )
        column.metric(f"{region} Dice", f"{metrics[0]:.3f}")
        column.caption(f"IoU {metrics[1]:.3f} · HD95 {metrics[2]:.1f} mm")
    st.caption("Reference comparison is meaningful only for a correctly aligned expert label.")
    reference_profile = profile_tumor(
        np.transpose(reference_regions["WT"], (2, 1, 0)),
        voxel_spacing=(spacing[2], spacing[1], spacing[0]),
    )
    st.metric(
        "WT volume difference vs reference",
        f"{profile['total_volume_cm3'] - reference_profile['total_volume_cm3']:+.2f} cm³",
        f"reference {reference_profile['total_volume_cm3']:.2f} cm³",
    )

st.subheader("📊 Tumor Profile", divider="gray")
summary = st.columns(4)
summary[0].metric("Whole tumor volume", f"{profile['total_volume_cm3']:.2f} cm³")
summary[1].metric("Tumor voxels", f"{profile['total_voxel_count']:,}")
summary[2].metric("Peak area", f"{profile['max_cross_section_cm2']:.2f} cm²")
summary[3].metric("Axial slice span", str(profile["lesion_depth_span_slices"]))
st.divider()

slice_index = st.slider("Axial slice", 0, labels.shape[2] - 1, labels.shape[2] // 2)
base = flair[:, :, slice_index]
mask_slice = labels[:, :, slice_index] > 0
nonzero = base[np.isfinite(base) & (base != 0)]
if nonzero.size:
    lower, upper = np.percentile(nonzero, (1, 99))
    base = np.clip((base - lower) / max(float(upper - lower), 1e-6), 0, 1)
else:
    base = np.zeros_like(base)
gray = np.repeat((base * 255).astype(np.uint8)[..., None], 3, axis=2)
gray[mask_slice] = (
    0.55 * gray[mask_slice].astype(np.float32)
    + 0.45 * np.array([255, 35, 35], dtype=np.float32)
).astype(np.uint8)

reference_overlay = None
if "reference_labels" in result:
    reference_slice = result["reference_regions"]["WT"][:, :, slice_index]
    reference_overlay = np.repeat((base * 255).astype(np.uint8)[..., None], 3, axis=2)
    reference_overlay[reference_slice] = (
        0.55 * reference_overlay[reference_slice].astype(np.float32)
        + 0.45 * np.array([35, 190, 70], dtype=np.float32)
    ).astype(np.uint8)

rotated_flair = np.rot90((base * 255).astype(np.uint8))
rotated_prediction = np.rot90(gray)
rotated_reference = np.rot90(reference_overlay) if reference_overlay is not None else None

with st.expander("🔍 Advanced View Controls"):
    view_controls = st.columns(3)
    zoom = view_controls[0].slider(
        "Image zoom", min_value=1.0, max_value=4.0, value=1.0, step=0.25,
        help="Zoom into the same location in the MRI and mask panels.",
    )
    height, width = rotated_flair.shape[:2]
    center_x = view_controls[1].slider(
        "Pan X", min_value=0, max_value=width - 1, value=width // 2,
        disabled=zoom == 1.0,
    )
    center_y = view_controls[2].slider(
        "Pan Y", min_value=0, max_value=height - 1, value=height // 2,
        disabled=zoom == 1.0,
    )

def zoom_view(image):
    crop_width = max(1, round(width / zoom))
    crop_height = max(1, round(height / zoom))
    left = int(np.clip(center_x - crop_width // 2, 0, width - crop_width))
    top = int(np.clip(center_y - crop_height // 2, 0, height - crop_height))
    return image[top:top + crop_height, left:left + crop_width]

left, right = st.columns(2)
left.image(zoom_view(rotated_flair), caption=f"FLAIR · slice {slice_index}", width="stretch")
right.image(
    zoom_view(rotated_prediction),
    caption="Predicted whole-tumor overlay",
    width="stretch",
)
if rotated_reference is not None:
    st.image(
        zoom_view(rotated_reference),
        caption="Ground-truth whole-tumor overlay",
        width="stretch",
    )

output_image = nib.Nifti1Image(labels.astype(np.uint8), result["affine"])
st.download_button(
    "Download predicted BraTS label mask (.nii)",
    data=output_image.to_bytes(),
    file_name="predicted_tumor_mask.nii",
    mime="application/octet-stream",
)
