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

st.set_page_config(
    page_title="Brain Tumour Profiler",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
    :root {
        --canvas: #f8f6f2;
        --surface: #fffefa;
        --surface-soft: #f3eee6;
        --line: #e7dfd3;
        --ink: #302820;
        --muted: #766d62;
        --brown: #76563f;
        --brown-dark: #5d4433;
        --brown-pale: #eee5d9;
    }
    html, body, [class*="css"] {
        font-family: Inter, ui-sans-serif, system-ui, -apple-system,
            BlinkMacSystemFont, "Segoe UI", sans-serif;
        color: var(--ink);
    }
    .stApp, [data-testid="stAppViewContainer"] {
        background: var(--canvas);
        color: var(--ink);
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMain"] > div {
        max-width: 1440px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    [data-testid="stSidebar"] {
        background: #f1ece4;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] > div:first-child {
        padding: 1.5rem 1.15rem;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    h1, h2, h3, p, label { color: var(--ink); }
    [data-testid="stCaptionContainer"], .stCaption {
        color: var(--muted);
    }
    .hero {
        position: relative;
        overflow: hidden;
        padding: 2.2rem 2.4rem;
        margin: 0 0 1.35rem;
        border: 1px solid #e8dfd3;
        border-radius: 22px;
        background: linear-gradient(120deg, #f1e9de 0%, #fbf8f3 68%, #eee4d7 100%);
    }
    .hero:after {
        content: "";
        position: absolute;
        width: 210px;
        height: 210px;
        right: 7%;
        top: -105px;
        border: 1px solid rgba(118, 86, 63, .14);
        border-radius: 50%;
        box-shadow: 0 0 0 28px rgba(118, 86, 63, .035),
                    0 0 0 58px rgba(118, 86, 63, .025);
        pointer-events: none;
    }
    .eyebrow {
        margin: 0 0 .65rem;
        color: var(--brown);
        font-size: .73rem;
        font-weight: 750;
        letter-spacing: .14em;
        text-transform: uppercase;
    }
    .hero h1 {
        position: relative;
        z-index: 1;
        margin: 0;
        color: #302820;
        font-family: Georgia, "Times New Roman", serif;
        font-size: clamp(2rem, 4vw, 3.2rem);
        font-weight: 500;
        letter-spacing: -.045em;
        line-height: 1.08;
    }
    .hero-copy {
        position: relative;
        z-index: 1;
        max-width: 660px;
        margin: .85rem 0 0;
        color: #6d6256;
        font-size: 1rem;
        line-height: 1.7;
    }
    .hero-link {
        position: relative;
        z-index: 1;
        display: inline-flex;
        align-items: center;
        min-height: 2.75rem;
        padding: .65rem 1rem;
        margin-top: 1.25rem;
        border: 1px solid var(--brown);
        border-radius: 9px;
        background: var(--brown);
        color: #fffefa !important;
        font-size: .88rem;
        font-weight: 650;
        text-decoration: none !important;
        transition: background .15s ease, transform .15s ease;
    }
    .hero-link:hover {
        background: var(--brown-dark);
        transform: translateY(-1px);
    }
    .notice {
        padding: .9rem 1.1rem;
        margin: 0 0 1.35rem;
        border: 1px solid #e8ddce;
        border-left: 3px solid var(--brown);
        border-radius: 10px;
        background: #f4eee5;
        color: #5e5144;
        font-size: .88rem;
        line-height: 1.55;
    }
    .section-heading {
        margin: 2rem 0 .4rem;
        color: var(--ink);
        font-family: Georgia, "Times New Roman", serif;
        font-size: 1.65rem;
        font-weight: 500;
        letter-spacing: -.025em;
    }
    .section-copy {
        margin: 0 0 1.2rem;
        color: var(--muted);
        font-size: .9rem;
        line-height: 1.6;
    }
    .feature-card, .step-card {
        height: 100%;
        min-height: 136px;
        box-sizing: border-box;
        padding: 1.15rem 1.2rem;
        border: 1px solid var(--line);
        border-radius: 14px;
        background: var(--surface);
    }
    .feature-card .card-kicker, .step-card .card-kicker {
        display: block;
        margin-bottom: .6rem;
        color: var(--brown);
        font-size: .72rem;
        font-weight: 750;
        letter-spacing: .1em;
        text-transform: uppercase;
    }
    .feature-card strong, .step-card strong {
        display: block;
        margin-bottom: .4rem;
        color: var(--ink);
        font-family: Georgia, "Times New Roman", serif;
        font-size: 1.12rem;
        font-weight: 500;
    }
    .feature-card p, .step-card p {
        margin: 0;
        color: var(--muted);
        font-size: .86rem;
        line-height: 1.6;
    }
    .landing-band {
        padding: 1.1rem 1.25rem;
        margin: 1.35rem 0;
        border: 1px solid #e8ddce;
        border-radius: 12px;
        background: #f4eee5;
        color: #5e5144;
        font-size: .88rem;
        line-height: 1.65;
    }
    .action-card {
        min-height: 116px;
        box-sizing: border-box;
        padding: 1.1rem 1.2rem .9rem;
        border: 1px solid var(--line);
        border-radius: 14px;
        background: var(--surface);
    }
    .action-card strong {
        display: block;
        margin-bottom: .28rem;
        color: var(--ink);
        font-size: .98rem;
    }
    .action-card span {
        display: block;
        color: var(--muted);
        font-size: .83rem;
        line-height: 1.5;
    }
    div[data-testid="stMetric"] {
        min-height: 112px;
        padding: 1rem 1.1rem;
        border: 1px solid var(--line);
        border-radius: 14px;
        background: var(--surface);
    }
    div[data-testid="stMetricLabel"] p {
        color: var(--muted);
        font-size: .8rem;
    }
    div[data-testid="stMetricValue"] {
        color: var(--brown-dark);
        font-size: 1.6rem;
    }
    div.stButton > button, div.stDownloadButton > button {
        min-height: 2.85rem;
        border: 1px solid #d9cbbb;
        border-radius: 10px;
        background: var(--surface);
        color: var(--brown-dark);
        font-weight: 650;
        transition: background .15s ease, border-color .15s ease,
                    transform .15s ease;
    }
    div.stButton > button:focus-visible,
    div.stDownloadButton > button:focus-visible,
    .hero-link:focus-visible {
        outline: 3px solid rgba(118, 86, 63, .28);
        outline-offset: 2px;
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover {
        border-color: #b89d82;
        background: #f5efe7;
        color: var(--brown-dark);
        transform: translateY(-1px);
    }
    div.stButton > button:disabled {
        border-color: #e7dfd3;
        background: #f1eee9;
        color: #92877b;
        opacity: 1;
        transform: none;
    }
    div.stButton > button[kind="primary"] {
        border-color: var(--brown);
        background: var(--brown);
        color: #fffefa;
    }
    div.stButton > button[kind="primary"]:hover {
        border-color: var(--brown-dark);
        background: var(--brown-dark);
        color: #fffefa;
    }
    [data-testid="stFileUploader"] section {
        border: 1px dashed #cdbba6;
        border-radius: 11px;
        background: rgba(255, 254, 250, .72);
    }
    [data-testid="stExpander"] {
        border: 1px solid var(--line);
        border-radius: 12px;
        background: var(--surface);
    }
    [data-testid="stImage"] img {
        border: 1px solid var(--line);
        border-radius: 12px;
    }
    [data-testid="stHorizontalBlock"] { gap: 1.15rem; align-items: stretch; }
    [data-testid="stVerticalBlock"] { gap: .85rem; }
    [data-testid="stMarkdownContainer"] p { line-height: 1.6; }
    [data-testid="stAlert"] {
        border-radius: 11px;
        border: 1px solid var(--line);
    }
    @media (max-width: 700px) {
        [data-testid="stMain"] > div { padding: 1rem 1rem 2rem; }
        .hero { padding: 1.6rem 1.35rem; border-radius: 16px; }
        .hero:after { right: -120px; }
        .section-heading { margin-top: 1.5rem; }
        .feature-card, .step-card { min-height: 0; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <section class="hero">
      <p class="eyebrow">Brain imaging · Research workspace</p>
      <h1>A clearer view of every scan.</h1>
      <p class="hero-copy">Explore 3D brain tumour segmentation and measured
      scan profiles with the MONAI research model.</p>
      <a class="hero-link" href="#choose-how-to-begin">Get started&nbsp; ↓</a>
    </section>
    <h2 class="section-heading">A research tool for exploring MRI scans</h2>
    <p class="section-copy">Brain Tumour Profiler runs a trained MONAI model on
    four MRI sequences, then presents its predicted mask and quantitative
    measurements in one focused workspace.</p>
    """,
    unsafe_allow_html=True,
)
feature_columns = st.columns(3, gap="large")
feature_content = (
    (
        "01 · SEGMENT",
        "See the predicted region",
        "Review the whole-tumour mask on an axial FLAIR slice with shared zoom and pan controls.",
    ),
    (
        "02 · MEASURE",
        "Understand the profile",
        "Explore estimated volume, voxel count, peak cross-sectional area, and slice span.",
    ),
    (
        "03 · COMPARE",
        "Evaluate against a label",
        "Optionally compare against an aligned reference mask using Dice, IoU, and HD95.",
    ),
)
for column, (kicker, title, description) in zip(feature_columns, feature_content):
    column.markdown(
        f'<div class="feature-card"><span class="card-kicker">{kicker}</span>'
        f"<strong>{title}</strong><p>{description}</p></div>",
        unsafe_allow_html=True,
    )
st.markdown(
    '<div class="landing-band"><strong>What you need:</strong> either one '
    'four-channel MSD-style NIfTI, or four co-registered 3D NIfTI files '
    '(FLAIR, T1, T1Gd, and T2). Inference uses the configured CUDA GPU. '
    'The prediction is exported in the preprocessed RAS / 1 mm grid.</div>'
    '<div class="notice"><strong>Research use only.</strong> This educational '
    'project is not a validated medical device. Masks and measurements are not '
    "a radiologist's report and must not be used for diagnosis or treatment.</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Scan setup")
    st.caption("Model checkpoint")
    st.caption(str(CHECKPOINT.relative_to(PROJECT_ROOT)))
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
    if st.button("Reset session and clear results", use_container_width=True):
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

st.markdown(
    '<h2 class="section-heading" id="analysis-options">Choose how to begin</h2>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="section-copy">Choose a held-out example or analyse your own '
    'co-registered MRI scan.</p>',
    unsafe_allow_html=True,
)
sample_column, upload_column = st.columns(2, gap="large")
sample_column.markdown(
    '<div class="action-card"><strong>Explore a held-out case</strong>'
    '<span>Run a test-split patient with an aligned reference mask.</span></div>',
    unsafe_allow_html=True,
)
upload_column.markdown(
    '<div class="action-card"><strong>Analyse your own scan</strong>'
    '<span>Use one four-channel file or four aligned modality files.</span></div>',
    unsafe_allow_html=True,
)
run_sample = sample_column.button(
    "Run held-out sample case",
    type="primary",
    use_container_width=True,
)
run_upload = upload_column.button(
    "Run uploaded scan",
    disabled=not uploads_ready,
    help="Upload the selected input format in the sidebar first.",
    use_container_width=True,
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
        demo_id = "BRATS_064.nii.gz"
        sample_id = demo_id if demo_id in test_ids else test_ids[0]
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
    st.markdown(
        '<div class="notice">Your workspace is ready. Choose one of the '
        'analysis options above to view a segmentation and scan profile.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

labels = result["labels"]
flair = result["flair"]
spacing = result["spacing_mm"]
profile = profile_tumor(
    np.transpose(labels > 0, (2, 1, 0)),
    voxel_spacing=(spacing[2], spacing[1], spacing[0]),
)
st.markdown('<h2 class="section-heading">Analysis overview</h2>', unsafe_allow_html=True)
device_column, case_column, grid_column = st.columns(3)
device_column.markdown(
    f"**Inference device**  \n{torch.cuda.get_device_name(INFERENCE_DEVICE)}",
)
if "sample_patient_id" in result:
    case_column.markdown(
        f"**Case**  \n{result['sample_patient_id']} · held-out test split",
    )
else:
    case_column.markdown("**Case**  \nUploaded MRI scan")
grid_column.markdown(
    "**Prediction grid**  \nRAS orientation · 1 mm isotropic voxels",
)

if "reference_labels" in result:
    st.markdown(
        '<h2 class="section-heading">Reference comparison</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="section-copy">Metrics compare the prediction against the '
        'provided aligned reference label.</p>',
        unsafe_allow_html=True,
    )
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

st.markdown('<h2 class="section-heading">Tumour profile</h2>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-copy">Computed measurements for the predicted whole-tumour region.</p>',
    unsafe_allow_html=True,
)
summary = st.columns(4)
summary[0].metric("Whole tumor volume", f"{profile['total_volume_cm3']:.2f} cm³")
summary[1].metric("Tumor voxels", f"{profile['total_voxel_count']:,}")
summary[2].metric("Peak area", f"{profile['max_cross_section_cm2']:.2f} cm²")
summary[3].metric("Axial slice span", str(profile["lesion_depth_span_slices"]))
st.markdown('<h2 class="section-heading">Slice viewer</h2>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-copy">Review the FLAIR image and segmentation overlay '
    'on the same axial slice.</p>',
    unsafe_allow_html=True,
)
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

if rotated_reference is not None:
    image_columns = st.columns(3, gap="large")
    image_columns[0].image(
        zoom_view(rotated_flair), caption=f"FLAIR · slice {slice_index}", width="stretch"
    )
    image_columns[1].image(
        zoom_view(rotated_prediction),
        caption="Predicted whole-tumour overlay",
        width="stretch",
    )
    image_columns[2].image(
        zoom_view(rotated_reference),
        caption="Ground-truth whole-tumour overlay",
        width="stretch",
    )
else:
    left, right = st.columns(2, gap="large")
    left.image(
        zoom_view(rotated_flair), caption=f"FLAIR · slice {slice_index}", width="stretch"
    )
    right.image(
        zoom_view(rotated_prediction),
        caption="Predicted whole-tumour overlay",
        width="stretch",
    )

output_image = nib.Nifti1Image(labels.astype(np.uint8), result["affine"])
st.markdown('<h2 class="section-heading">Export</h2>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-copy">Download the prediction in the displayed RAS / '
    '1 mm grid.</p>',
    unsafe_allow_html=True,
)
st.download_button(
    "Download predicted BraTS label mask (.nii)",
    data=output_image.to_bytes(),
    file_name="predicted_tumor_mask.nii",
    mime="application/octet-stream",
    use_container_width=True,
)
