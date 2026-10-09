"""Focused tests for the MONAI data split, labels, and spatial metrics."""

import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from monai_brats import (
    ConvertBraTSRegionsd,
    MODALITIES,
    read_dataset_label_region_map,
    patient_split,
    probabilities_to_label_map,
    probabilities_to_region_masks,
    read_dataset_channel_order,
    make_transforms,
)
from monai_inference import predict_case
from test.eval_metrics import dice, evaluate_brats_regions, hd95, iou, summarize_metric


class _EmptySegmentationModel(torch.nn.Module):
    def forward(self, image):
        return image.new_full((image.shape[0], 3, *image.shape[2:]), -10.0)


class MonaiPipelineTests(unittest.TestCase):
    def test_patient_split_is_reproducible_and_disjoint(self):
        cases = [{"patient_id": f"case-{index}"} for index in range(20)]
        first = patient_split(cases, seed=12)
        second = patient_split(cases, seed=12)
        self.assertEqual(
            [[case["patient_id"] for case in first[key]] for key in first],
            [[case["patient_id"] for case in second[key]] for key in second],
        )
        groups = [{case["patient_id"] for case in first[key]} for key in first]
        self.assertFalse(groups[0] & groups[1])
        self.assertFalse(groups[0] & groups[2])
        self.assertFalse(groups[1] & groups[2])

    def test_reads_msd_modality_order(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = {
                "modality": {
                    "0": "FLAIR",
                    "1": "T1w",
                    "2": "T1gd",
                    "3": "T2w",
                },
                "labels": {
                    "0": "background",
                    "1": "edema",
                    "2": "non-enhancing tumor",
                    "3": "enhancing tumour",
                },
            }
            Path(directory, "dataset.json").write_text(json.dumps(manifest))
            self.assertEqual(read_dataset_channel_order(directory), MODALITIES)
            self.assertEqual(
                read_dataset_label_region_map(directory),
                {1: (True, False, False), 2: (True, True, False), 3: (True, True, True)},
            )

    def test_brat_labels_convert_to_nested_regions(self):
        label = np.array([[[0, 1, 2, 4]]])
        converted = ConvertBraTSRegionsd()({"label": label})["label"]
        np.testing.assert_array_equal(
            converted[:, 0, 0, :],
            np.array([[False, True, True, True],
                      [False, True, False, True],
                      [False, False, False, True]]),
        )

    def test_probability_regions_produce_valid_brat_labels(self):
        probabilities = np.zeros((3, 1, 1, 3), dtype=np.float32)
        probabilities[2, 0, 0, 0] = 0.8
        probabilities[1, 0, 0, 1] = 0.8
        probabilities[0, 0, 0, 2] = 0.8
        np.testing.assert_array_equal(
            probabilities_to_label_map(probabilities),
            np.array([[[4, 1, 2]]], dtype=np.uint8),
        )
        regions = probabilities_to_region_masks(probabilities)
        self.assertTrue(np.all(regions[2] <= regions[1]))
        self.assertTrue(np.all(regions[1] <= regions[0]))

    def test_msd_ground_truth_encoding_matches_brat_prediction_regions(self):
        prediction = np.array([[[2, 1, 4]]], dtype=np.uint8)
        reference = np.array([[[1, 2, 3]]], dtype=np.uint8)
        metrics = evaluate_brats_regions(
            prediction, reference, voxel_spacing=(1.0, 1.0, 1.0),
            gt_encoding="msd-task01",
        )
        for region in ("WT", "TC", "ET"):
            self.assertEqual(metrics[region]["dice"], 1.0)
            self.assertEqual(metrics[region]["iou"], 1.0)

    def test_training_transform_crops_and_converts_multimodal_case(self):
        random = np.random.default_rng(5)
        image = random.normal(size=(32, 32, 32, 4)).astype(np.float32)
        image[:2] = 0
        label = np.zeros((32, 32, 32), dtype=np.uint8)
        label[10:19, 10:19, 10:19] = 2
        label[13:16, 13:16, 13:16] = 4
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory, "image.nii.gz")
            label_path = Path(directory, "label.nii.gz")
            nib.save(nib.Nifti1Image(image, np.eye(4)), image_path)
            nib.save(nib.Nifti1Image(label, np.eye(4)), label_path)
            training_transform, _ = make_transforms((16, 16, 16), MODALITIES)
            sample = training_transform(
                {"image": str(image_path), "label": str(label_path)}
            )[0]
            self.assertEqual(tuple(sample["image"].shape), (4, 16, 16, 16))
            self.assertEqual(tuple(sample["label"].shape), (3, 16, 16, 16))

    def test_metrics_include_empty_and_physical_distance_cases(self):
        empty = np.zeros((4, 4, 4), dtype=bool)
        self.assertEqual(dice(empty, empty), 1.0)
        self.assertEqual(iou(empty, empty), 1.0)
        self.assertEqual(hd95(empty, empty), 0.0)
        shifted_a = empty.copy()
        shifted_b = empty.copy()
        shifted_a[1, 1, 1] = True
        shifted_b[2, 1, 1] = True
        self.assertAlmostEqual(hd95(shifted_a, shifted_b, (2.0, 1.0, 1.0)), 2.0)

    def test_metric_summary_counts_non_finite_cases(self):
        summary = summarize_metric([1.0, 0.5, float("inf")])
        self.assertEqual(summary["mean"], 0.75)
        self.assertAlmostEqual(summary["std"], 0.25)
        self.assertEqual(summary["finite_case_count"], 2)
        self.assertEqual(summary["non_finite_case_count"], 1)

    def test_combined_nifti_inference_returns_flair_and_reference(self):
        random = np.random.default_rng(6)
        image = random.normal(size=(16, 16, 16, 4)).astype(np.float32)
        label = np.zeros((16, 16, 16), dtype=np.uint8)
        label[6:10, 6:10, 6:10] = 2
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory, "image.nii.gz")
            label_path = Path(directory, "label.nii.gz")
            nib.save(nib.Nifti1Image(image, np.eye(4)), image_path)
            nib.save(nib.Nifti1Image(label, np.eye(4)), label_path)
            checkpoint = {
                "config": {
                    "roi_size": (8, 8, 8),
                    "source_channel_order": MODALITIES,
                    "label_region_map": {
                        1: (True, False, False),
                        2: (True, True, False),
                        3: (True, True, True),
                    },
                }
            }
            with (
                patch("monai_inference.select_device", return_value=torch.device("cpu")),
                patch(
                    "monai_inference.load_checkpoint",
                    return_value=(_EmptySegmentationModel(), checkpoint),
                ),
            ):
                result = predict_case(
                    None,
                    "mock-checkpoint.pt",
                    reference_label_path=str(label_path),
                    combined_path=str(image_path),
                    device="cpu",
                )
        self.assertEqual(result["labels"].shape, (16, 16, 16))
        self.assertEqual(result["flair"].shape, (16, 16, 16))
        np.testing.assert_array_equal(result["reference_labels"], label)

    def test_requested_cuda_does_not_silently_fall_back_to_cpu(self):
        image = np.zeros((16, 16, 16, 4), dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory, "image.nii.gz")
            nib.save(nib.Nifti1Image(image, np.eye(4)), image_path)
            with patch("monai_inference.torch.cuda.is_available", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "CUDA inference was requested"):
                    predict_case(
                        None,
                        "mock-checkpoint.pt",
                        combined_path=str(image_path),
                        device="cuda:0",
                    )


if __name__ == "__main__":
    unittest.main()
