"""Integration Tests verifying seamless Person 1 and Person 2 interoperability."""

import numpy as np
import pytest
import torch
from pathlib import Path

from src.data.datasets import MockMultimodalDataset
from src.evaluation.metrics import compute_segmentation_metrics, MetricAccumulator
from src.models.predict import predict_chip
from src.data.split import audit_splits


def test_split_integrity():
    """Verify that official split files have zero spatial leakage."""
    splits_dir = Path("data/splits")
    if splits_dir.exists() and (splits_dir / "flood_train_data.csv").exists():
        audit = audit_splits(splits_dir)
        assert audit["leakage"] is False, "Critical spatial leakage detected between splits!"
        assert audit["bolivia_isolated"] is True, "Bolivia held-out split is not isolated!"


def test_evaluation_metric_accumulator():
    """Verify Person 2's MetricAccumulator with ignore_index=-1."""
    pred = torch.tensor([[0, 1], [1, 0]])
    target = torch.tensor([[0, 1], [1, -1]])

    acc = MetricAccumulator(num_classes=2, ignore_index=-1)
    acc.update(pred, target)
    res = acc.compute()

    assert res["water_iou"] == 1.0
    assert res["total_ignored_pixels"] == 1
    assert res["accuracy"] == 1.0


def test_predict_chip_handoff():
    """Verify that Person 1's predict_chip conforms to Person 2's expected interface."""
    dummy_s1 = np.random.rand(2, 512, 512).astype(np.float32)
    dummy_s2 = np.random.rand(13, 512, 512).astype(np.float32)

    pred_mask = predict_chip(dummy_s1, dummy_s2)
    assert pred_mask.shape == (512, 512), f"Expected (512, 512), got {pred_mask.shape}"
    assert pred_mask.dtype == np.int64


if __name__ == "__main__":
    test_split_integrity()
    test_evaluation_metric_accumulator()
    test_predict_chip_handoff()
    print("All integration tests passed!")
