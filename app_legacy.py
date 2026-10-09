"""
app_legacy.py – Heuristic Brain Tumor Segmentation Baseline
=============================================
Legacy baseline Streamlit dashboard. This page uses the original heuristic /
2-D inference path and does not use the trained MONAI model. Features:

  - HIPAA-compliant anonymization (local processing, zero PHI retention)
  - Axial slice viewer with mask overlay
  - Quantitative tumor biomarker cards
  - Per-slice area chart and intensity histogram
  - Dynamic .pth model weight upload
  - Anonymized CSV / text report export

Launch:  streamlit run app_legacy.py
"""

import csv
import io
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from anonymizer import anonymize_numpy_volume, verify_phi_removal
from model_inference import (
    get_inference_backend,
    load_from_bytes,
    load_model_from_path,
    load_spacing_from_bytes,
    segment_volume,
)
from profiler import export_profile_csv_rows, format_profile_for_display, profile_tumor

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor Segmentation & Profiler",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global stylesheet  – light mode, brown (#7B4F2E) accent
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Base ── */
    html, body, .stApp {
        background-color: #F7F3EF;
        color: #2C1A0E;
        font-family: 'Georgia', 'Times New Roman', serif;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background-color: #EDE4DA;
        border-right: 1px solid #C9B49A;
    }
    section[data-testid="stSidebar"] * {
        color: #3B2010 !important;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stFileUploader label {
        font-weight: 600;
        font-size: 0.83rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    /* ── Top-level headings ── */
    h1 { color: #5C3317; letter-spacing: -0.5px; }
    h2 { color: #6B3C1F; }
    h3 { color: #7B4F2E; font-size: 1.05rem; font-weight: 700; }

    /* ── Metric cards ── */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #D4B896;
        border-left: 4px solid #7B4F2E;
        border-radius: 6px;
        padding: 10px 14px 10px 14px;
    }
    div[data-testid="stMetric"] label {
        color: #7B6352 !important;
        font-size: 0.72rem !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] > div {
        color: #3B1E0A !important;
        font-size: 1.0rem !important;
        font-weight: 700;
        line-height: 1.25;
        white-space: normal !important;
        word-break: break-word;
    }

    /* ── Divider ── */
    hr { border-color: #C9B49A; }

    /* ── Buttons ── */
    .stDownloadButton > button, .stButton > button {
        background-color: #7B4F2E;
        color: #FFFFFF;
        border: none;
        border-radius: 5px;
        font-weight: 600;
        letter-spacing: 0.03em;
        padding: 0.4rem 1.1rem;
        transition: background 0.2s;
    }
    .stDownloadButton > button:hover, .stButton > button:hover {
        background-color: #5C3317;
        color: #FFFFFF;
    }

    /* ── Selectbox / sliders ── */
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF;
        border-color: #C9B49A;
    }
    .stSlider > div > div > div[data-baseweb="slider"] {
        background: #D4B896;
    }

    /* ── Info / warning / success alerts ── */
    div[data-testid="stAlert"] {
        border-radius: 6px;
    }

    /* ── HIPAA badge ── */
    .hipaa-badge {
        background: #5C8A4E;
        color: #FFFFFF;
        padding: 7px 14px;
        border-radius: 5px;
        font-size: 0.76rem;
        font-weight: 600;
        text-align: center;
        letter-spacing: 0.04em;
        margin-bottom: 0.8rem;
    }

    /* ── Audit table ── */
    .audit-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.82rem;
        margin-top: 0.4rem;
    }
    .audit-table td {
        padding: 6px 10px;
        border-bottom: 1px solid #E2D5C8;
        vertical-align: top;
    }
    .audit-table td:first-child {
        color: #7B6352;
        font-weight: 600;
        width: 48%;
        white-space: nowrap;
    }
    .audit-table td:last-child {
        color: #2C1A0E;
    }
    .audit-table tr:last-child td {
        border-bottom: none;
    }

    /* ── Case ID pill ── */
    .case-pill {
        display: inline-block;
        background: #EDE4DA;
        border: 1px solid #C9B49A;
        color: #5C3317;
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.78rem;
        font-family: 'Courier New', monospace;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }

    /* ── Block container ── */
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Cached model weight loader  (st.cache_resource – survives re-runs)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model weights…")
def _load_weights_from_bytes(weight_bytes: bytes) -> bool:
    """
    Write user-supplied .pth bytes to a temporary file, load the weights,
    and return True on success.  The resource is cached so re-runs don't
    reload from disk.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp:
        tmp.write(weight_bytes)
        tmp_path = tmp.name
    try:
        model = load_model_from_path(tmp_path)
        return model is not None
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Brain Tumor Profiler")

    st.markdown("---")

    # ── MRI upload ──
    st.markdown("**MRI Scan**")
    modality = st.selectbox(
        "Modality",
        options=["FLAIR", "T1ce"],
        index=0,
        help="Select the MRI modality of the uploaded scan.",
    )
    uploaded_file = st.file_uploader(
        "Upload scan file",
        type=["nii", "gz", "png", "jpg", "jpeg"],
        help="Supported: NIfTI (.nii, .nii.gz), PNG, JPG.",
        label_visibility="collapsed",
    )

    st.markdown("---")

    # ── Model weights upload ──
    st.markdown("**Model Weights (optional)**")
    pth_file = st.file_uploader(
        "Upload .pth weights",
        type=["pth"],
        help="Upload a fine-tuned PyTorch checkpoint to enable deep-learning inference.",
        label_visibility="collapsed",
    )

    weights_loaded = False
    if pth_file is not None:
        weight_bytes = pth_file.read()
        weights_loaded = _load_weights_from_bytes(weight_bytes)
        if weights_loaded:
            st.success("Model weights loaded.", icon="✔")
        else:
            st.error("Could not load weights. Check the .pth file.")

    st.markdown("---")

    # ── Inference controls ──
    st.markdown("**Inference Controls**")
    threshold = st.slider(
        "Segmentation Threshold",
        min_value=0.1,
        max_value=0.95,
        value=0.50,
        step=0.05,
        help="Higher values → more selective tumor detection.",
    )
    overlay_alpha = st.slider(
        "Overlay Opacity",
        min_value=0.1,
        max_value=0.9,
        value=0.45,
        step=0.05,
        help="Transparency of the predicted mask overlay.",
    )

    st.markdown("---")
    st.caption("v1.0 · CPU-Optimised · HIPAA Safe Harbor")


# ─────────────────────────────────────────────────────────────────────────────
# Cached processing helpers
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading scan…")
def cached_load(file_bytes: bytes, filename: str) -> np.ndarray:
    return load_from_bytes(file_bytes, filename)


@st.cache_data(show_spinner=False)
def cached_spacing(file_bytes: bytes, filename: str) -> Tuple[float, float, float]:
    return load_spacing_from_bytes(file_bytes, filename)


@st.cache_data(show_spinner="Segmenting…")
def cached_segment(
    volume_bytes: bytes, shape: Tuple[int, ...], threshold: float
) -> Tuple[np.ndarray, Dict[str, Any]]:
    volume = np.frombuffer(volume_bytes, dtype=np.float32).reshape(shape)
    return segment_volume(volume, threshold)


@st.cache_data(show_spinner=False)
def cached_anonymize(
    volume_bytes: bytes, shape: Tuple[int, ...], source: str
) -> Tuple[np.ndarray, Dict[str, Any]]:
    volume = np.frombuffer(volume_bytes, dtype=np.float32).reshape(shape)
    return anonymize_numpy_volume(volume, source)


@st.cache_data(show_spinner=False)
def cached_profile(
    mask_bytes: bytes, mask_shape: Tuple[int, ...], spacing: Tuple[float, float, float]
) -> Dict[str, Any]:
    mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(mask_shape)
    return profile_tumor(mask, spacing)


# ─────────────────────────────────────────────────────────────────────────────
# Main heading
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# Brain Tumor Segmentation & Profiler")

# ─────────────────────────────────────────────────────────────────────────────
# Landing state (no file uploaded)
# ─────────────────────────────────────────────────────────────────────────────
if uploaded_file is None:
    st.info(
        "Upload a brain MRI scan from the sidebar to begin. "
        "Supported formats: NIfTI (.nii, .nii.gz), PNG, JPG.",
    )
    with st.expander("How it works", expanded=True):
        c1, c2, c3 = st.columns(3)
        c1.markdown(
            "**1. Anonymize**  \nAll uploads are processed locally. "
            "DICOM PHI tags are stripped before any data is analysed."
        )
        c2.markdown(
            "**2. Segment**  \nA lightweight 2-D U-Net runs slice-by-slice on CPU. "
            "When no checkpoint is present, a hyperintensity-based fallback is used."
        )
        c3.markdown(
            "**3. Profile**  \nQuantitative biomarkers — volume, cross-section, "
            "lesion span — are computed and presented for clinical review."
        )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Fallback warning banner
# ─────────────────────────────────────────────────────────────────────────────
active_backend = get_inference_backend()
if "Heuristic" in active_backend:
    st.warning(
        "Clinical Warning: Deep learning weights not found. "
        "Using morphological fallback heuristic. "
        "Upload a .pth model file in the sidebar for accurate tumour profiling."
    )

# ─────────────────────────────────────────────────────────────────────────────
# Processing pipeline
# ─────────────────────────────────────────────────────────────────────────────
file_bytes = uploaded_file.read()
filename   = uploaded_file.name

# ── Load ──
try:
    volume = cached_load(file_bytes, filename)
except Exception as exc:
    # Catch corrupt NIfTI, unreadable image, etc.
    import nibabel
    if isinstance(exc, nibabel.filebasedimages.ImageFileError):
        st.error(
            "Invalid MRI file format detected. "
            "Please upload a valid binary .nii or .nii.gz file."
        )
    else:
        st.error(f"Failed to load file: {exc}")
    st.stop()

# ── Anonymize ──
vol_bytes  = volume.astype(np.float32).tobytes()
vol_shape  = volume.shape
anon_volume, anon_meta = cached_anonymize(vol_bytes, vol_shape, filename)
compliance = verify_phi_removal(anon_meta)

# ── Segment ──
mask_volume, seg_info = cached_segment(vol_bytes, vol_shape, threshold)

# ── Spacing ──
try:
    spacing = cached_spacing(file_bytes, filename)
except Exception:
    spacing = (1.0, 1.0, 1.0)

# ── Profile ──
mask_bytes    = mask_volume.astype(np.uint8).tobytes()
tumor_profile = cached_profile(mask_bytes, mask_volume.shape, spacing)
display_mets  = format_profile_for_display(tumor_profile)
case_id       = anon_meta.get("CaseID", "UNKNOWN")

# ─────────────────────────────────────────────────────────────────────────────
# Two-column layout
# ─────────────────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([3, 2], gap="large")

# ── LEFT: Axial slice viewer ──────────────────────────────────────────────────
with left_col:
    st.markdown(f"### Axial Slice Viewer — {modality}")

    num_slices = volume.shape[0]
    default_idx = min(
        tumor_profile.get("peak_slice_index", num_slices // 2),
        num_slices - 1,
    )
    slice_idx = st.slider(
        "Slice Index",
        min_value=0,
        max_value=max(num_slices - 1, 0),
        value=default_idx,
        key="slice_slider",
    )

    slice_data = volume[slice_idx]
    mask_data  = mask_volume[slice_idx]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor="#F7F3EF")

    # Original
    axes[0].imshow(slice_data, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original Scan", color="#3B1E0A", fontsize=11, pad=8)
    axes[0].axis("off")

    # Overlay: build an explicit RGBA image so the mask is always visible
    axes[1].imshow(slice_data, cmap="gray", vmin=0, vmax=1)
    if mask_data.any():
        # Create a bright red RGBA overlay; alpha only where mask == 1
        overlay_rgba = np.zeros((*mask_data.shape, 4), dtype=np.float32)
        overlay_rgba[mask_data > 0, 0] = 0.9   # R
        overlay_rgba[mask_data > 0, 1] = 0.15  # G
        overlay_rgba[mask_data > 0, 2] = 0.15  # B
        overlay_rgba[mask_data > 0, 3] = overlay_alpha  # A
        axes[1].imshow(overlay_rgba)
    axes[1].set_title("Predicted Mask Overlay", color="#3B1E0A", fontsize=11, pad=8)
    axes[1].axis("off")

    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ── RIGHT: Tumour Profiler Dashboard ─────────────────────────────────────────
with right_col:
    st.markdown("### Tumour Profiler")

    if compliance["is_compliant"]:
        st.success("HIPAA Safe Harbor — Fully De-identified")
    else:
        st.warning(f"PHI fields still present: {', '.join(compliance['residual_phi'])}")

    st.markdown(
        f'<div class="case-pill">Case ID: {case_id}</div>',
        unsafe_allow_html=True,
    )

    # Metric cards  2 × 2
    m1, m2 = st.columns(2)
    m3, m4 = st.columns(2)
    m1.metric("Tumour Volume", display_mets["Volume"])
    m2.metric("Peak Slice", display_mets["Peak Slice"])
    m3.metric("Lesion Coverage", display_mets["Lesion Coverage"])
    m4.metric("Processing Time", f"{seg_info['total_latency_ms']:.0f} ms")

    st.markdown("---")

    # HIPAA audit trail rendered as a tidy table, not raw JSON
    st.markdown("#### HIPAA Audit")
    audit_rows = [
        ("Status",            "Compliant" if compliance["is_compliant"] else "Non-compliant"),
        ("PHI Tags Checked",  str(compliance["total_checked"])),
        ("Residual PHI",      str(len(compliance["residual_phi"])) if compliance["residual_phi"] else "None"),
        ("Timestamp (UTC)",   anon_meta.get("Timestamp_UTC", "—")),
        ("Voxel Count",       display_mets["Voxel Count"]),
    ]
    rows_html = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in audit_rows
    )
    st.markdown(
        f'<table class="audit-table"><tbody>{rows_html}</tbody></table>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Bottom section
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
bot_left, bot_right = st.columns([3, 2], gap="large")

with bot_left:
    st.markdown("### Slice Intensity Distribution")

    fig_hist, ax_hist = plt.subplots(figsize=(8, 3), facecolor="#F7F3EF")
    ax_hist.hist(
        volume[slice_idx].ravel(),
        bins=128,
        color="#7B4F2E",
        alpha=0.75,
        edgecolor="none",
    )
    ax_hist.set_xlabel("Intensity", color="#5C3317", fontsize=9)
    ax_hist.set_ylabel("Count", color="#5C3317", fontsize=9)
    ax_hist.set_title(
        f"Slice {slice_idx} — Pixel Intensity Histogram",
        color="#3B1E0A",
        fontsize=10,
    )
    ax_hist.tick_params(colors="#5C3317", labelsize=8)
    ax_hist.set_facecolor("#FFFFFF")
    for spine in ax_hist.spines.values():
        spine.set_color("#D4B896")
    plt.tight_layout()
    st.pyplot(fig_hist, use_container_width=True)
    plt.close(fig_hist)

    if tumor_profile["has_tumor"] and num_slices > 1:
        st.markdown("### Per-Slice Tumour Area")
        areas = tumor_profile["per_slice_area_cm2"]
        fig_area, ax_area = plt.subplots(figsize=(8, 3), facecolor="#F7F3EF")
        ax_area.fill_between(range(len(areas)), areas, color="#C9956A", alpha=0.4)
        ax_area.plot(areas, color="#7B4F2E", linewidth=1.8)
        ax_area.axvline(
            x=slice_idx, color="#B03A2E", linestyle="--", alpha=0.7, linewidth=1.2,
            label="Current slice",
        )
        ax_area.set_xlabel("Slice Index", color="#5C3317", fontsize=9)
        ax_area.set_ylabel("Area (cm²)", color="#5C3317", fontsize=9)
        ax_area.set_title("Tumour Cross-Sectional Area per Slice", color="#3B1E0A", fontsize=10)
        ax_area.tick_params(colors="#5C3317", labelsize=8)
        ax_area.set_facecolor("#FFFFFF")
        ax_area.legend(fontsize=8, facecolor="#F7F3EF", edgecolor="#D4B896")
        for spine in ax_area.spines.values():
            spine.set_color("#D4B896")
        plt.tight_layout()
        st.pyplot(fig_area, use_container_width=True)
        plt.close(fig_area)

with bot_right:
    st.markdown("### Export Anonymized Summary")

    # CSV
    csv_rows   = export_profile_csv_rows(tumor_profile, case_id)
    csv_buffer = io.StringIO()
    if csv_rows:
        writer = csv.DictWriter(csv_buffer, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    st.download_button(
        label="Download CSV Report",
        data=csv_buffer.getvalue(),
        file_name=f"{case_id}_tumor_profile.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # Plain text
    txt_lines = [
        "Brain Tumor Segmentation & Profiler — Anonymized Report",
        "=" * 55,
        f"Case ID:              {case_id}",
        f"Modality:             {modality}",
        f"Threshold:            {threshold}",
        f"Slices Processed:     {seg_info['num_slices']}",
        f"Total Latency:        {seg_info['total_latency_ms']:.0f} ms",
        "",
        "--- Tumor Profile ---",
        f"Has Tumor:            {tumor_profile['has_tumor']}",
        f"Volume:               {tumor_profile['total_volume_cm3']:.4f} cm3",
        f"Voxel Count:          {tumor_profile['total_voxel_count']:,}",
        f"Max Cross-Section:    {tumor_profile['max_cross_section_cm2']:.4f} cm2",
        f"Peak Slice:           {tumor_profile['peak_slice_index']}",
        f"Lesion Start:         {tumor_profile['lesion_start_slice']}",
        f"Lesion End:           {tumor_profile['lesion_end_slice']}",
        f"Lesion Depth:         {tumor_profile['lesion_depth_span_mm']:.2f} mm "
        f"({tumor_profile['lesion_depth_span_slices']} slices)",
        "",
        "--- Compliance ---",
        f"HIPAA Status:         {'Compliant' if compliance['is_compliant'] else 'Non-compliant'}",
        f"PHI Tags Checked:     {compliance['total_checked']}",
        f"Residual PHI:         {len(compliance['residual_phi'])}",
        "",
        "Generated by Brain Tumor Segmentation & Profiler v1.0",
    ]
    st.download_button(
        label="Download Text Summary",
        data="\n".join(txt_lines),
        file_name=f"{case_id}_summary.txt",
        mime="text/plain",
        use_container_width=True,
    )
