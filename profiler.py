"""
profiler.py – Quantitative Biomarker Profiling Engine
======================================================
Computes clinically relevant tumor metrics from binary segmentation masks:
  • Total tumor volume (cm³)
  • Maximum axial cross-sectional area (cm²) and its slice index
  • Lesion depth span (start → end slice)
  • Necrotic / active tumor voxel count

All outputs are structured dictionaries ready for Streamlit rendering and
CSV / PDF export.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np


def profile_tumor(
    binary_mask: np.ndarray,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Dict[str, Any]:
    """
    Compute quantitative tumor biomarkers from a binary segmentation mask.

    Parameters
    ----------
    binary_mask : np.ndarray
        3-D array of shape (D, H, W) where non-zero values denote tumor.
        For 2-D masks (H, W), they are automatically promoted to (1, H, W).
    voxel_spacing : tuple of float, optional
        Physical spacing in mm along (depth/slice, height, width).
        Default is isotropic 1 mm³ voxels.

    Returns
    -------
    profile : dict
        Structured dictionary with the following keys:

        - ``total_volume_cm3``       : float  – Total tumor volume in cm³.
        - ``total_voxel_count``      : int    – Raw voxel count of tumor mass.
        - ``max_cross_section_cm2``  : float  – Largest axial cross-section area (cm²).
        - ``peak_slice_index``       : int    – Slice index of the largest cross-section.
        - ``lesion_start_slice``     : int    – First slice containing tumor.
        - ``lesion_end_slice``       : int    – Last slice containing tumor.
        - ``lesion_depth_span_mm``   : float  – Physical depth span of the lesion (mm).
        - ``lesion_depth_span_slices``: int   – Number of slices the lesion spans.
        - ``per_slice_area_cm2``     : list   – Cross-sectional area per slice (cm²).
        - ``has_tumor``              : bool   – Whether any tumor voxels were found.

    Raises
    ------
    ValueError
        If ``binary_mask`` has fewer than 2 dimensions.
    """
    # ---- Input validation ----
    if binary_mask.ndim < 2:
        raise ValueError(
            f"Expected a 2-D or 3-D mask, got {binary_mask.ndim}-D array."
        )

    # Promote 2-D to 3-D (single slice)
    if binary_mask.ndim == 2:
        binary_mask = binary_mask[np.newaxis, ...]  # (1, H, W)

    # Binarize (threshold at 0.5 for soft masks, >0 for hard masks)
    mask = (binary_mask > 0).astype(np.uint8)

    depth, height, width = mask.shape
    sz, sy, sx = voxel_spacing  # mm per voxel along each axis

    # ---- Voxel-level metrics ----
    total_voxel_count = int(mask.sum())
    voxel_volume_mm3 = sz * sy * sx               # mm³ per voxel
    total_volume_mm3 = total_voxel_count * voxel_volume_mm3
    total_volume_cm3 = total_volume_mm3 / 1000.0   # 1 cm³ = 1000 mm³

    # ---- Per-slice cross-sectional area ----
    pixel_area_mm2 = sy * sx  # in-plane pixel area (mm²)
    per_slice_voxels = np.array([int(mask[s].sum()) for s in range(depth)])
    per_slice_area_mm2 = per_slice_voxels * pixel_area_mm2
    per_slice_area_cm2 = (per_slice_area_mm2 / 100.0).tolist()  # 1 cm² = 100 mm²

    # ---- Peak cross-section ----
    if total_voxel_count > 0:
        peak_slice_index = int(np.argmax(per_slice_voxels))
        max_cross_section_cm2 = per_slice_area_cm2[peak_slice_index]
    else:
        peak_slice_index = 0
        max_cross_section_cm2 = 0.0

    # ---- Lesion depth span ----
    slices_with_tumor = np.nonzero(per_slice_voxels)[0]
    if len(slices_with_tumor) > 0:
        lesion_start = int(slices_with_tumor[0])
        lesion_end = int(slices_with_tumor[-1])
        lesion_depth_slices = lesion_end - lesion_start + 1
        lesion_depth_mm = lesion_depth_slices * sz
    else:
        lesion_start = 0
        lesion_end = 0
        lesion_depth_slices = 0
        lesion_depth_mm = 0.0

    # ---- Assemble profile ----
    profile: Dict[str, Any] = {
        "has_tumor": total_voxel_count > 0,
        "total_volume_cm3": round(total_volume_cm3, 4),
        "total_voxel_count": total_voxel_count,
        "max_cross_section_cm2": round(max_cross_section_cm2, 4),
        "peak_slice_index": peak_slice_index,
        "lesion_start_slice": lesion_start,
        "lesion_end_slice": lesion_end,
        "lesion_depth_span_mm": round(lesion_depth_mm, 2),
        "lesion_depth_span_slices": lesion_depth_slices,
        "per_slice_area_cm2": per_slice_area_cm2,
    }
    return profile


def format_profile_for_display(profile: Dict[str, Any]) -> Dict[str, str]:
    """
    Convert a raw profile dict into human-readable display strings
    suitable for Streamlit metric cards.

    Parameters
    ----------
    profile : dict
        Output of ``profile_tumor``.

    Returns
    -------
    display : dict
        Formatted strings for each metric.
    """
    if not profile["has_tumor"]:
        return {
            "Volume": "No tumor detected",
            "Peak Slice": "—",
            "Lesion Coverage": "0 slices",
            "Voxel Count": "0",
        }

    return {
        "Volume": f"{profile['total_volume_cm3']:.2f} cm³",
        "Peak Slice": (
            f"Slice {profile['peak_slice_index']} "
            f"({profile['max_cross_section_cm2']:.2f} cm²)"
        ),
        "Lesion Coverage": (
            f"Slices {profile['lesion_start_slice']}–{profile['lesion_end_slice']} "
            f"({profile['lesion_depth_span_slices']} slices, "
            f"{profile['lesion_depth_span_mm']:.1f} mm)"
        ),
        "Voxel Count": f"{profile['total_voxel_count']:,}",
    }


def export_profile_csv_rows(
    profile: Dict[str, Any],
    case_id: str = "UNKNOWN",
) -> list:
    """
    Flatten the profile into a list of (key, value) rows for CSV export.

    Parameters
    ----------
    profile : dict
        Output of ``profile_tumor``.
    case_id : str
        Anonymous case identifier.

    Returns
    -------
    rows : list of dict
        Each dict has keys ``CaseID``, ``Metric``, ``Value``.
    """
    flat_metrics = {
        "Total Volume (cm³)": profile["total_volume_cm3"],
        "Total Voxel Count": profile["total_voxel_count"],
        "Max Cross-Section (cm²)": profile["max_cross_section_cm2"],
        "Peak Slice Index": profile["peak_slice_index"],
        "Lesion Start Slice": profile["lesion_start_slice"],
        "Lesion End Slice": profile["lesion_end_slice"],
        "Lesion Depth Span (mm)": profile["lesion_depth_span_mm"],
        "Lesion Depth Span (slices)": profile["lesion_depth_span_slices"],
    }
    return [
        {"CaseID": case_id, "Metric": k, "Value": v}
        for k, v in flat_metrics.items()
    ]
