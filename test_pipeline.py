import sys
sys.stdout.reconfigure(encoding="utf-8")
import warnings
warnings.filterwarnings("ignore")

import model_inference, profiler, anonymizer

# ---- 2D PNG slice ----
with open("sample_data/brain_mri_axial_slice_45.png", "rb") as f:
    raw = f.read()

vol = model_inference.load_from_bytes(raw, "brain_mri_axial_slice_45.png")
anon_vol, meta = anonymizer.anonymize_numpy_volume(vol, "test.png")
mask, info = model_inference.segment_volume(anon_vol, threshold=0.5)
p = profiler.profile_tumor(mask)

print("Backend     :", info["backend"])
print("Latency ms  :", info["total_latency_ms"])
print("Voxel count :", p["total_voxel_count"])
print("Volume cm3  :", p["total_volume_cm3"])
print("Max area cm2:", p["max_cross_section_cm2"])

vol_cm3 = p["total_volume_cm3"]
if vol_cm3 >= 415:
    print("FAIL: old whole-brain bug still present, volume =", vol_cm3)
    sys.exit(1)
elif vol_cm3 == 0:
    print("FAIL: no tumor detected at all")
    sys.exit(1)
else:
    print("PASS: heuristic volume is clinically plausible.")

# ---- NIfTI 3D ----
with open("sample_data/sample_brain_mri.nii", "rb") as f:
    nii_raw = f.read()

vol3d = model_inference.load_from_bytes(nii_raw, "sample_brain_mri.nii")
spacing = model_inference.load_spacing_from_bytes(nii_raw, "sample_brain_mri.nii")
print("\nNIfTI volume shape:", vol3d.shape, "spacing:", spacing)

subset = vol3d[40:48]
mask3d, info3d = model_inference.segment_volume(subset, threshold=0.5)
p3d = profiler.profile_tumor(mask3d, voxel_spacing=spacing)
print("3D volume cm3 :", p3d["total_volume_cm3"])
print("Lesion slices :", p3d["lesion_start_slice"], "->", p3d["lesion_end_slice"])

if p3d["total_volume_cm3"] >= 415:
    print("FAIL: 3D whole-brain volume bug")
    sys.exit(1)
print("PASS: NIfTI 3D pipeline correct.")
print("\nAll tests passed.")
