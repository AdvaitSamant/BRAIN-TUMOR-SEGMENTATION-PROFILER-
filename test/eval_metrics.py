"""Dice, IoU, and HD95 for binary masks and BraTS tumor regions."""
import argparse

import numpy as np


def dice(pred, gt, eps=1e-7) -> float:
    p, g = np.asarray(pred).astype(bool), np.asarray(gt).astype(bool)
    if not p.any() and not g.any():
        return 1.0                                  # both empty: perfect agreement
    return float(2 * (p & g).sum() / (p.sum() + g.sum()))


def iou(pred, gt, eps=1e-7) -> float:
    p, g = np.asarray(pred).astype(bool), np.asarray(gt).astype(bool)
    union = (p | g).sum()
    return 1.0 if union == 0 else float((p & g).sum() / union)


def hd95(pred, gt, voxel_spacing=None) -> float:
    """Symmetric 95th-percentile Hausdorff distance in millimeters."""
    from scipy import ndimage

    p, g = np.asarray(pred).astype(bool), np.asarray(gt).astype(bool)
    if p.shape != g.shape:
        raise ValueError(f"Mask shapes differ: prediction {p.shape}, ground truth {g.shape}")
    if voxel_spacing is None:
        voxel_spacing = (1.0,) * p.ndim
    elif len(voxel_spacing) != p.ndim:
        raise ValueError(
            f"Expected {p.ndim} voxel spacings for a {p.ndim}-D mask, got {len(voxel_spacing)}"
        )
    if not p.any() and not g.any():
        return 0.0
    if not p.any() or not g.any():
        return float("inf")

    structure = ndimage.generate_binary_structure(p.ndim, 1)
    p_surface = p & ~ndimage.binary_erosion(p, structure=structure, border_value=0)
    g_surface = g & ~ndimage.binary_erosion(g, structure=structure, border_value=0)
    distance_to_g = ndimage.distance_transform_edt(~g_surface, sampling=voxel_spacing)
    distance_to_p = ndimage.distance_transform_edt(~p_surface, sampling=voxel_spacing)
    distances = np.concatenate((distance_to_g[p_surface], distance_to_p[g_surface]))
    return float(np.percentile(distances, 95))


def evaluate(pred, gt) -> dict:
    return {"dice": dice(pred, gt), "iou": iou(pred, gt), "hd95_mm": hd95(pred, gt)}


def summarize_metric(values) -> dict:
    """Summarize finite values while counting non-finite cases explicitly."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return {
        "mean": float(np.mean(finite)) if finite.size else None,
        "std": float(np.std(finite)) if finite.size else None,
        "finite_case_count": int(finite.size),
        "non_finite_case_count": int(values.size - finite.size),
    }


def evaluate_brats_regions(
    pred_labels,
    gt_labels,
    voxel_spacing=(1.0, 1.0, 1.0),
    gt_encoding="brats",
) -> dict:
    """Evaluate standard 0/1/2/4 predictions against BraTS or MSD Task01 labels."""
    pred, gt = np.asarray(pred_labels), np.asarray(gt_labels)
    if pred.shape != gt.shape:
        raise ValueError(f"Mask shapes differ: prediction {pred.shape}, ground truth {gt.shape}")
    if gt_encoding == "brats":
        gt_regions = {
            "WT": gt > 0,
            "TC": np.isin(gt, (1, 4)),
            "ET": gt == 4,
        }
    elif gt_encoding == "msd-task01":
        gt_regions = {
            "WT": gt > 0,
            "TC": np.isin(gt, (2, 3)),
            "ET": gt == 3,
        }
    else:
        raise ValueError("gt_encoding must be 'brats' or 'msd-task01'")
    regions = {
        "WT": (pred > 0, gt_regions["WT"]),
        "TC": (np.isin(pred, (1, 4)), gt_regions["TC"]),
        "ET": (pred == 4, gt_regions["ET"]),
    }
    return {
        name: {
            "dice": dice(pred_region, gt_region),
            "iou": iou(pred_region, gt_region),
            "hd95_mm": hd95(pred_region, gt_region, voxel_spacing),
        }
        for name, (pred_region, gt_region) in regions.items()
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred"); ap.add_argument("--gt")
    ap.add_argument(
        "--gt-encoding", choices=("msd-task01", "brats"), default="msd-task01",
        help="Ground-truth encoding: MSD Task01 labels 1/2/3 or BraTS labels 1/2/4",
    )
    a = ap.parse_args()
    if a.pred and a.gt:
        import nibabel as nib
        pred_img, gt_img = nib.load(a.pred), nib.load(a.gt)
        if pred_img.shape != gt_img.shape or not np.allclose(
            pred_img.affine, gt_img.affine, atol=1e-3
        ):
            raise ValueError(
                "Prediction and ground truth must be in the same voxel grid and affine"
            )
        pred = np.asarray(pred_img.dataobj)
        gt = np.asarray(gt_img.dataobj)
        spacing = tuple(float(value) for value in gt_img.header.get_zooms()[:3])
        print(evaluate_brats_regions(pred, gt, spacing, a.gt_encoding))
    else:  # self-test
        a_, b_ = np.zeros((10, 10)), np.zeros((10, 10)); a_[2:6, 2:6] = 1; b_[4:8, 4:8] = 1
        print(evaluate(a_, b_), "(expected dice 0.25, iou 0.143)")
        assert abs(dice(a_, b_) - 0.25) < 1e-3 and dice(np.zeros(3), np.zeros(3)) == 1.0
