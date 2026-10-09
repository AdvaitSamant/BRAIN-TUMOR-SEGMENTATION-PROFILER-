"""Brain Tumor Segmentation & Profiler - local, in-memory Streamlit app.
Run:  streamlit run app.py"""
import gc
import glob
import os
import time

import numpy as np
import pandas as pd
import streamlit as st

from deidentify import deidentify_bytes, sanitized_copy, snapshot
from eval_metrics import dice, iou
from profiler import TumorProfiler
from segmenter import Segmenter
from slicestream import CHANNELS, SliceStream, load_nifti_bytes

st.set_page_config(page_title="Brain Tumor Segmentation & Profiler", layout="wide")


@st.cache_resource
def get_segmenter():
    return Segmenter()          # model weights only; never patient data


def to_u8(img):
    nz = img[img > 0]
    if nz.size == 0:
        return np.zeros(img.shape, np.uint8)
    lo, hi = np.percentile(nz, (1, 99))
    return (np.clip((img - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)


def overlay(base_u8, mask, alpha=0.45):
    rgb = np.stack([base_u8] * 3, -1).astype(np.float32)
    rgb[mask] = (1 - alpha) * rgb[mask] + alpha * np.array([255, 40, 40], np.float32)
    return np.rot90(rgb.astype(np.uint8))        # rotate so the slice appears upright


def show_report(rep):
    if not rep["detected"]:
        st.info("No tumor detected in this volume.")
        return
    a, b, c, d = st.columns(4)
    a.metric("Tumor volume", f"{rep['volume_cm3']:.2f} cm³")
    b.metric("Axial slice range", f"{rep['slice_first']} to {rep['slice_last']}")
    c.metric("Peak slice", f"{rep['peak_slice']}", f"{rep['peak_area_mm2']:.0f} mm² cross-section")
    d.metric("Slices involved", rep["n_slices_involved"])
    if "contrast" in rep:
        st.dataframe(pd.DataFrame(rep["contrast"]).T.rename(columns={
            "lesion_mean": "Lesion mean", "contralateral_mean": "Contralateral mean", "ratio": "Lesion / contralateral"}))
    st.caption("Slice indices are 0-based along the axial axis. Contralateral reference = left-right mirrored brain tissue.")


seg = get_segmenter()
with st.sidebar:
    st.header("Privacy & runtime")
    st.markdown("- Files are processed **in memory** (`BytesIO`); nothing is written to disk\n"
                "- App binds to `127.0.0.1`, telemetry off\n- Inference runs **locally on CPU**")
    thr = st.slider("Mask threshold", 0.1, 0.9, 0.5, 0.05)
    if st.button("Reset session (wipe memory)"):
        st.session_state.clear(); gc.collect(); st.rerun()

st.title("Brain Tumor Segmentation & Profiler")
if seg.is_demo:
    st.warning("DEMO MODE: no trained weights found in `weights/`. Masks come from a crude intensity rule, "
               "NOT the fine-tuned model. Do not report these numbers.")
else:
    st.success(f"Model backend: {seg.backend}  ({os.path.basename(seg.path)})")
seg.threshold = thr

tab_scan, tab_demo, tab_deid = st.tabs(["Scan: segment & profile", "Demo cache", "DICOM de-identification"])

# ---------------------------------------------------------------- full volume
with tab_scan:
    c1, c2 = st.columns(2)
    f_t1 = c1.file_uploader("T1ce (.nii / .nii.gz)", type=["nii", "gz"])
    f_fl = c2.file_uploader("FLAIR (.nii / .nii.gz)", type=["nii", "gz"])
    if f_t1 and f_fl and st.button("Run segmentation", type="primary"):
        try:
            stream = SliceStream({"t1ce": load_nifti_bytes(f_t1.getvalue(), f_t1.name),
                                  "flair": load_nifti_bytes(f_fl.getvalue(), f_fl.name)})
        except Exception as e:
            st.error(f"Could not read volumes: {e}")
            st.stop()
        K, (H, W) = stream.n_slices, stream.hw
        masks, base, lat = np.zeros((K, H, W), bool), np.zeros((K, H, W), np.uint8), []
        prof, bar = TumorProfiler(stream.zooms, CHANNELS), st.progress(0.0)
        for k, raw in stream:                       # one 2D slice at a time
            base[k] = to_u8(raw[1])
            t0 = time.perf_counter(); m = seg.predict_mask(raw); lat.append((time.perf_counter() - t0) * 1000)
            masks[k] = m; prof.update(k, m, raw)
            bar.progress((k + 1) / K)
        st.session_state["res"] = dict(masks=masks, base=base, report=prof.report(), lat=float(np.mean(lat)))
        del stream; gc.collect()

    res = st.session_state.get("res")
    if res:
        show_report(res["report"])
        st.caption(f"Mean inference latency: {res['lat']:.1f} ms/slice ({seg.backend})")
        rep = res["report"]
        k = st.slider("Axial slice", 0, res["masks"].shape[0] - 1, rep.get("peak_slice", res["masks"].shape[0] // 2))
        left, right = st.columns(2)
        left.image(np.rot90(res["base"][k]), caption=f"FLAIR slice {k}", clamp=True, use_container_width=True)
        right.image(overlay(res["base"][k], res["masks"][k]), caption="Predicted tumor overlay", use_container_width=True)
        areas = res["masks"].sum(axis=(1, 2))
        st.bar_chart(pd.DataFrame({"tumor pixels": areas}), height=160)

# ---------------------------------------------------------------- demo cache
with tab_demo:
    files = sorted(glob.glob("demo_samples/*.npz"))
    if not files:
        st.info("Run `python make_demo_samples.py` first.")
    else:
        pick = st.selectbox("Pre-loaded sample", files, format_func=os.path.basename)
        z = np.load(pick, allow_pickle=False)
        raw, gt = z["image"], z["mask"].astype(bool)
        if "source" in z.files:
            st.caption(f"Source: {z['source']}")
        t0 = time.perf_counter(); m = seg.predict_mask(raw); ms = (time.perf_counter() - t0) * 1000
        l, r = st.columns(2)
        l.image(overlay(to_u8(raw[1]), gt), caption="Ground truth", use_container_width=True)
        r.image(overlay(to_u8(raw[1]), m), caption=f"Prediction ({ms:.0f} ms)", use_container_width=True)
        st.metric("Dice / IoU", f"{dice(m, gt):.3f} / {iou(m, gt):.3f}")
        if not gt.any():
            st.caption("Negative case: Dice = 1.0 only when the model also predicts nothing.")

# ---------------------------------------------------------------- DICOM de-id
with tab_deid:
    f = st.file_uploader("DICOM file (.dcm)", type=["dcm"])
    if f:
        import io, pydicom
        data = f.getvalue()
        ds = pydicom.dcmread(io.BytesIO(data))
        before, after = snapshot(ds), snapshot(sanitized_copy(ds), include_uids=False)
        clean, warns = deidentify_bytes(data)
        for w in warns:
            st.warning(w)
        st.dataframe(pd.DataFrame({"Before": pd.Series(before), "After": pd.Series(after)}).fillna("<removed>"))
        st.download_button("Download sanitized DICOM", clean, file_name="deidentified.dcm")
