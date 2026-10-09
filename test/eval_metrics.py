"""Dice (DSC) and IoU for binary masks. CLI: python eval_metrics.py --pred p.nii.gz --gt seg.nii.gz"""
import argparse

import numpy as np


def dice(pred, gt, eps=1e-7) -> float:
    p, g = np.asarray(pred).astype(bool), np.asarray(gt).astype(bool)
    if not p.any() and not g.any():
        return 1.0                                  # both empty: perfect agreement
    return float(2 * (p & g).sum() / (p.sum() + g.sum() + eps))


def iou(pred, gt, eps=1e-7) -> float:
    p, g = np.asarray(pred).astype(bool), np.asarray(gt).astype(bool)
    union = (p | g).sum()
    return 1.0 if union == 0 else float((p & g).sum() / (union + eps))


def evaluate(pred, gt) -> dict:
    return {"dice": dice(pred, gt), "iou": iou(pred, gt)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred"); ap.add_argument("--gt")
    a = ap.parse_args()
    if a.pred and a.gt:
        import nibabel as nib
        pred, gt = nib.load(a.pred).get_fdata() > 0, nib.load(a.gt).get_fdata() > 0   # BraTS labels 1,2,4 -> whole tumor
        print(evaluate(pred, gt))
    else:  # self-test
        a_, b_ = np.zeros((10, 10)), np.zeros((10, 10)); a_[2:6, 2:6] = 1; b_[4:8, 4:8] = 1
        print(evaluate(a_, b_), "(expected dice 0.25, iou 0.143)")
        assert abs(dice(a_, b_) - 0.25) < 1e-3 and dice(np.zeros(3), np.zeros(3)) == 1.0
