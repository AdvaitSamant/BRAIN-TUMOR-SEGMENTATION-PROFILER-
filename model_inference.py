"""
model_inference.py – CPU-Optimized Segmentation Pipeline
=========================================================
Dual-path inference engine for brain tumor segmentation:

  Path A – PyTorch checkpoint (model_weights.pth or user-supplied path):
    Loads a lightweight 2-D U-Net on CPU with torch.inference_mode().

  Path B – Deterministic hyperintensity fallback:
    Isolates the top 5-10% of in-brain pixel intensities to simulate
    FLAIR/T1ce tumor signal, then keeps only the largest connected
    component after morphological cleanup.  The brain is never
    returned wholesale as a tumor mask.

Supports:
  NIfTI volumes  (.nii, .nii.gz)
  Standard 2-D images (.png, .jpg, .jpeg)
"""

import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CHECKPOINT_PATH = Path("model_weights.pth")
INPUT_SIZE = (256, 256)  # Resize target for U-Net inference (H, W)

# ---------------------------------------------------------------------------
# Lazy torch import
# ---------------------------------------------------------------------------
_torch = None
_torch_nn = None


def _ensure_torch() -> bool:
    """Import torch lazily. Returns True when torch is available."""
    global _torch, _torch_nn
    if _torch is None:
        try:
            import torch
            import torch.nn as nn
            _torch = torch
            _torch_nn = nn
        except ImportError:
            _torch = False
            _torch_nn = False
    return _torch is not False


# ===================================================================
# Lightweight 2-D U-Net  (~1.9 M params, fits in < 100 MB RAM)
# ===================================================================

def _build_unet():
    if not _ensure_torch():
        return None

    nn = _torch_nn

    class _DoubleConv(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        def forward(self, x):
            return self.net(x)

    class LiteUNet(nn.Module):
        def __init__(self, in_channels=1, out_channels=1):
            super().__init__()
            fs = [32, 64, 128, 256]
            self.enc1 = _DoubleConv(in_channels, fs[0])
            self.enc2 = _DoubleConv(fs[0], fs[1])
            self.enc3 = _DoubleConv(fs[1], fs[2])
            self.enc4 = _DoubleConv(fs[2], fs[3])
            self.pool = nn.MaxPool2d(2)
            self.bottleneck = _DoubleConv(fs[3], fs[3])
            self.up4 = nn.ConvTranspose2d(fs[3], fs[3], 2, stride=2)
            self.dec4 = _DoubleConv(fs[3] * 2, fs[2])
            self.up3 = nn.ConvTranspose2d(fs[2], fs[2], 2, stride=2)
            self.dec3 = _DoubleConv(fs[2] * 2, fs[1])
            self.up2 = nn.ConvTranspose2d(fs[1], fs[1], 2, stride=2)
            self.dec2 = _DoubleConv(fs[1] * 2, fs[0])
            self.up1 = nn.ConvTranspose2d(fs[0], fs[0], 2, stride=2)
            self.dec1 = _DoubleConv(fs[0] * 2, fs[0])
            self.out_conv = nn.Conv2d(fs[0], out_channels, 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            e4 = self.enc4(self.pool(e3))
            b  = self.bottleneck(self.pool(e4))
            d4 = self.dec4(_torch.cat([self.up4(b), e4], dim=1))
            d3 = self.dec3(_torch.cat([self.up3(d4), e3], dim=1))
            d2 = self.dec2(_torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(_torch.cat([self.up1(d2), e1], dim=1))
            return _torch.sigmoid(self.out_conv(d1))

    return LiteUNet


# ===================================================================
# Model loading – supports both default path and a caller-supplied path
# ===================================================================
_cached_model = None
_cached_checkpoint_path: Optional[str] = None


def load_model_from_path(checkpoint_path: str) -> Optional[Any]:
    """
    Load a PyTorch U-Net checkpoint from an explicit path.
    Returns the model in eval mode, or None on failure.
    This function resets the internal cache so the new weights
    are picked up immediately.
    """
    global _cached_model, _cached_checkpoint_path

    if not _ensure_torch():
        return None

    UNet = _build_unet()
    if UNet is None:
        return None

    try:
        model = UNet()
        state = _torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        _cached_model = model
        _cached_checkpoint_path = checkpoint_path
        return model
    except Exception as exc:
        print(f"[model_inference] Checkpoint load failed ({checkpoint_path}): {exc}")
        _cached_model = None
        _cached_checkpoint_path = None
        return None


def _load_default_checkpoint() -> Optional[Any]:
    """
    Attempt to load the default checkpoint (model_weights.pth) once.
    Results are cached in-process.
    """
    global _cached_model, _cached_checkpoint_path

    # Already loaded (or explicitly set by load_model_from_path)
    if _cached_model is not None:
        return _cached_model

    # No default file present
    if not DEFAULT_CHECKPOINT_PATH.exists():
        return None

    return load_model_from_path(str(DEFAULT_CHECKPOINT_PATH))


def get_active_model() -> Optional[Any]:
    """Return the currently loaded model, or None if none is available."""
    return _cached_model if _cached_model is not None else _load_default_checkpoint()


def get_inference_backend() -> str:
    """Human-readable label for the active inference backend."""
    return "PyTorch U-Net (CPU)" if get_active_model() is not None else "Heuristic / Otsu (fallback)"


# ===================================================================
# Input loaders
# ===================================================================

def load_nifti(filepath: str) -> np.ndarray:
    """
    Load a NIfTI file and return a float32 (D, H, W) volume in [0, 1].
    Uses get_fdata() for safe, float-converted access.

    Raises
    ------
    nibabel.filebasedimages.ImageFileError
        Propagated so callers can show a clean error message.
    """
    import nibabel as nib
    img = nib.load(filepath)               # raises ImageFileError for corrupt files
    data = img.get_fdata(dtype=np.float32) # float32 conversion is guaranteed
    if data.ndim == 4:
        data = data[..., 0]                # take first modality channel
    vmin, vmax = data.min(), data.max()
    if vmax - vmin > 0:
        data = (data - vmin) / (vmax - vmin)
    return data


def load_nifti_spacing(filepath: str) -> Tuple[float, float, float]:
    """Extract voxel spacing (mm) from a NIfTI header."""
    import nibabel as nib
    img = nib.load(filepath)
    zooms = img.header.get_zooms()[:3]
    return tuple(float(z) if float(z) > 0 else 1.0 for z in zooms)


def load_image(filepath: str) -> np.ndarray:
    """Load a 2-D image file and return a (1, H, W) float32 array in [0, 1]."""
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Unable to read image: {filepath}")
    img = img.astype(np.float32)
    vmin, vmax = img.min(), img.max()
    if vmax - vmin > 0:
        img = (img - vmin) / (vmax - vmin)
    return img[np.newaxis, ...]


def load_from_bytes(file_bytes: bytes, filename: str) -> np.ndarray:
    """
    Load scan data from an in-memory byte buffer (Streamlit upload).

    NIfTI files are written to a temporary file on disk because nibabel
    requires a seekable binary stream.  The temp file is always deleted.

    Raises
    ------
    IOError / nibabel.filebasedimages.ImageFileError
        On corrupt or unsupported file content.
    """
    suffix = Path(filename).suffix.lower()
    is_nifti = suffix in (".nii", ".gz")

    if is_nifti:
        actual_suffix = ".nii.gz" if filename.lower().endswith(".nii.gz") else ".nii"
        with tempfile.NamedTemporaryFile(delete=False, suffix=actual_suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            vol = load_nifti(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        return vol
    else:
        arr = np.frombuffer(file_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(
                f"Could not decode image bytes for '{filename}'. "
                "Make sure the file is a valid PNG or JPEG."
            )
        img = img.astype(np.float32)
        vmin, vmax = img.min(), img.max()
        if vmax - vmin > 0:
            img = (img - vmin) / (vmax - vmin)
        return img[np.newaxis, ...]


def load_spacing_from_bytes(file_bytes: bytes, filename: str) -> Tuple[float, float, float]:
    """Extract voxel spacing from in-memory NIfTI bytes; default (1,1,1) for 2-D images."""
    suffix = Path(filename).suffix.lower()
    if suffix in (".nii", ".gz"):
        actual_suffix = ".nii.gz" if filename.lower().endswith(".nii.gz") else ".nii"
        with tempfile.NamedTemporaryFile(delete=False, suffix=actual_suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            spacing = load_nifti_spacing(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        return spacing
    return (1.0, 1.0, 1.0)


# ===================================================================
# Segmentation engines
# ===================================================================

def _preprocess_slice(slice_2d: np.ndarray) -> np.ndarray:
    """Resize and clip a 2-D slice to INPUT_SIZE for U-Net inference."""
    resized = cv2.resize(slice_2d, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    return np.clip(resized, 0.0, 1.0)


def _postprocess_mask(
    prob_map: np.ndarray,
    original_hw: Tuple[int, int],
    threshold: float = 0.5,
) -> np.ndarray:
    """Resize a probability map back to the original resolution and binarize."""
    resized = cv2.resize(
        prob_map, (original_hw[1], original_hw[0]), interpolation=cv2.INTER_LINEAR
    )
    return (resized >= threshold).astype(np.uint8)


def _segment_slice_pytorch(
    slice_2d: np.ndarray,
    model: Any,
    threshold: float = 0.5,
) -> np.ndarray:
    """Run one slice through the PyTorch U-Net on CPU."""
    original_hw = slice_2d.shape[:2]
    tensor = _torch.from_numpy(_preprocess_slice(slice_2d)).unsqueeze(0).unsqueeze(0)
    with _torch.inference_mode():
        prob = model(tensor).squeeze().cpu().numpy()
    return _postprocess_mask(prob, original_hw, threshold)


def _largest_connected_component(binary: np.ndarray) -> np.ndarray:
    """
    Return a boolean mask containing only the single largest connected
    component of the input binary image.  Returns an all-zero mask when
    no components are found.
    """
    from scipy import ndimage as ndi
    labeled, n_labels = ndi.label(binary)
    if n_labels == 0:
        return np.zeros_like(binary, dtype=bool)
    sizes = ndi.sum(binary, labeled, range(1, n_labels + 1))
    largest_label = int(np.argmax(sizes)) + 1
    return labeled == largest_label


def _segment_slice_heuristic(
    slice_2d: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Clinically-constrained heuristic segmentation for FLAIR / T1ce slices.

    Pipeline
    --------
    1. Build a rough brain mask by excluding near-zero background.
    2. Within the brain mask, isolate the top (5 – 10)% of pixel
       intensities — the hyperintense region that corresponds to
       oedema or enhancing tumor on FLAIR / T1ce.
    3. Apply morphological opening (scipy.ndimage) to remove speckle.
    4. Keep only the single largest connected component, so scattered
       noise blobs are never reported as tumour volume.

    This ensures the entire head is never returned as the tumor mask,
    which was the previous clinical error.
    """
    from scipy import ndimage as ndi

    img = np.clip(slice_2d, 0.0, 1.0).astype(np.float32)

    # Skip empty or near-empty slices
    if img.max() < 0.05:
        return np.zeros(img.shape, dtype=np.uint8)

    # --- Step 1: coarse brain mask (exclude air / skull background) ---
    # Background voxels are typically very close to 0 after normalisation.
    brain_mask = img > 0.05

    # Remove isolated background dots
    brain_mask = ndi.binary_fill_holes(brain_mask)
    brain_mask = ndi.binary_opening(brain_mask, iterations=2)

    if brain_mask.sum() == 0:
        return np.zeros(img.shape, dtype=np.uint8)

    # --- Step 2: hyperintensity threshold within the brain ---
    # Confidence slider maps to a percentile band [90th, 95th].
    # threshold=0.5 → 92.5th percentile  (midpoint)
    # threshold=0.1 → 90th percentile    (more inclusive)
    # threshold=0.95→ 95th percentile    (most selective)
    percentile = 90.0 + threshold * 5.0
    brain_pixels = img[brain_mask]
    cutoff = float(np.percentile(brain_pixels, percentile))

    tumor_candidate = brain_mask & (img >= cutoff)

    if tumor_candidate.sum() == 0:
        return np.zeros(img.shape, dtype=np.uint8)

    # --- Step 3: morphological opening to remove speckle ---
    struct = ndi.generate_binary_structure(2, 1)
    cleaned = ndi.binary_opening(tumor_candidate, structure=struct, iterations=2)

    # --- Step 4: keep only the largest connected component ---
    result = _largest_connected_component(cleaned)

    return result.astype(np.uint8)


# ===================================================================
# Public API
# ===================================================================

def segment_volume(
    volume: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Segment an entire 3-D volume slice-by-slice.

    Parameters
    ----------
    volume : np.ndarray
        Float32 array of shape (D, H, W), normalised to [0, 1].
    threshold : float
        Confidence threshold for binarising predictions (0.1 – 0.95).

    Returns
    -------
    mask_volume : np.ndarray
        Binary uint8 mask of shape (D, H, W).
    info : dict
        Keys: backend, num_slices, total_latency_ms, per_slice_latency_ms.
    """
    if volume.ndim == 2:
        volume = volume[np.newaxis, ...]

    depth = volume.shape[0]
    mask_volume = np.zeros_like(volume, dtype=np.uint8)
    model = get_active_model()
    backend = "PyTorch U-Net (CPU)" if model is not None else "Heuristic / Otsu"

    t_start = time.perf_counter()
    for s in range(depth):
        sl = volume[s]
        if model is not None:
            mask_volume[s] = _segment_slice_pytorch(sl, model, threshold)
        else:
            mask_volume[s] = _segment_slice_heuristic(sl, threshold)
    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    return mask_volume, {
        "backend": backend,
        "num_slices": depth,
        "total_latency_ms": round(elapsed_ms, 2),
        "per_slice_latency_ms": round(elapsed_ms / max(depth, 1), 2),
    }
