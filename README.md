# Brain Tumor Segmentation Profiler

This project includes an experimental 3D MONAI training and inference path for
the Medical Segmentation Decathlon (MSD) Brain Tumour task. The original
single-image heuristic remains a demo only; it is not a tumor segmentation
model and should not be used for clinical decisions.

## Environment

Use Python 3.10–3.12 for the broadest PyTorch/MONAI compatibility. On Windows,
create an isolated environment and install a CUDA-enabled PyTorch build that
matches the local driver (the example below matches this RTX 4050 setup):

```powershell
py -3.11 -m venv .venv-monai
.\.venv-monai\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

For another CUDA version or platform, choose the matching PyTorch install
command from the official PyTorch selector instead of copying the example.
Check GPU access:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

The scripts automatically select CUDA when PyTorch can use it. They fall back
to CPU otherwise. A 6 GB GPU may require reducing the default training patch
size or running fewer workers/cases.

## Dataset and training

Download the MSD Brain Tumour archive from the official project mirror
([dataset page](http://medicaldecathlon.com/)). The archive is approximately
7.1 GB; extracting it requires additional disk space. Follow the dataset
license/terms and do not redistribute the archive without permission. The
downloaded release's `dataset.json` does not declare a license field; verify
the current terms with the dataset provider before reuse or redistribution.
The release includes segmentation labels, not a paired radiologist narrative
report. The app's profile is a computed measurement summary, not a clinical
report.

```powershell
New-Item -ItemType Directory -Force data | Out-Null
curl.exe -L --fail --retry 3 `
  -o data/Task01_BrainTumour.tar `
  https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar
tar -xf data/Task01_BrainTumour.tar -C data
```

Train with deterministic **patient-level** train/validation/test partitions.
The test partition is kept out of model selection and is evaluated after
training:

```powershell
python scripts/train_monai.py `
  --dataset-root data/Task01_BrainTumour `
  --output-dir runs/monai_brats `
  --epochs 50
python test/evaluate_monai.py `
  --dataset-root data/Task01_BrainTumour `
  --checkpoint runs/monai_brats/best_model.pt `
  --output runs/monai_brats/test_metrics `
  --save-example
```

Training saves the split manifest, per-epoch CSV history, best validation
checkpoint, and final test metrics. The checkpoint is ignored by Git. Use
`--max-cases` only for a short pipeline smoke test; such a run is not an
accuracy evaluation. Set `--roi-size` (three integers) lower if the GPU runs
out of memory. With `--save-example`, evaluation also writes a held-out
four-channel input NIfTI, its aligned reference label, the predicted BraTS
label map, and a JSON profile/metric comparison under
`test_metrics/example_case/`.

MSD Task01 label descriptions are read from its `dataset.json`: label 1 is
edema, 2 is non-enhancing tumor, and 3 is enhancing tumor. These are converted
to nested learning targets (WT, TC, ET); exported predictions use standard
BraTS IDs (edema 2, non-enhancing core 1, enhancing 4). The CLI evaluator
defaults to MSD Task01 ground-truth encoding and can compare BraTS masks with
`--gt-encoding brats`. Input channels are reordered from the manifest into
FLAIR, T1, T1Gd, T2. State the exact dataset release and region definitions in
any project report.

## Inference app

After training, run the MONAI app:

```powershell
streamlit run app.py
```

Choose **MONAI 3D** from Streamlit's page navigation.
Use **Run held-out sample case** to run a labeled example from the test split
created during training. This requires the extracted dataset at
`data/Task01_BrainTumour/`; the sample includes its reference mask and metrics.
Alternatively, upload a scan and choose **Run uploaded scan**.

**MSD 4-channel NIfTI** means one 4D NIfTI file containing the four MRI
sequences (FLAIR, T1, T1Gd, and T2) as channels. Choose it for an MSD-style
combined file. **Four separate modalities** means four 3D NIfTI files—one per
sequence—from the same patient and already co-registered to the same voxel
grid. Choose it when your scanner/export gives each sequence as its own file;
they must not be four arbitrary slices or scans from different studies.

The app uses the checkpoint's preprocessing (RAS orientation, 1 mm resampling,
channel-wise nonzero intensity normalization), shows the predicted WT mask,
and reports a volume profile. It can optionally compare predictions with an
aligned ground-truth label map using the dataset encoding saved in the
checkpoint. The displayed mask is in the preprocessed RAS/1 mm grid and its
NIfTI download carries the corresponding affine.

The MONAI Streamlit page requires CUDA and explicitly runs inference on
`cuda:0`; it displays the detected GPU model and stops with an error rather than
silently switching to CPU when CUDA is unavailable. The slice viewer fills the
available panel width and provides zoom and X/Y pan controls shared by the
MRI, prediction, and reference overlays.

## Evaluation and limitations

The test evaluator reports per-case and aggregate Dice, IoU, and HD95 for WT,
TC, and ET; empty-vs-empty regions score Dice/IoU 1 and HD95 0, while a
one-sided empty region has infinite HD95. Aggregate HD95 mean/std use only
finite cases and include explicit finite/non-finite counts; a non-finite case
indicates a complete miss or false positive for that region. Evaluate only on
patients excluded from training and validation. Patient-level results—not
slice-level splits—are required to avoid leakage.

This is an educational project, not a validated medical device. A model's
output can be wrong; its report is not a radiologist's report and must not be
used for diagnosis or treatment. The existing heuristic path is especially
unsuitable for accuracy claims. Report performance on an untouched test set,
include failure cases and uncertainty/limitations, and do not infer clinical
accuracy from a plausible-looking overlay.
