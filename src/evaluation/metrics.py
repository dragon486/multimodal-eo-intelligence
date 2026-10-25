"""
Evaluation metrics for Flood Segmentation and Environmental Intelligence.
Strictly ignores nodata/invalid pixels (-1 or specified ignore_index).
Computes IoU, Mean IoU, Precision, Recall, Macro-F1, and Confusion Matrix.
Pure NumPy core with universal tensor compatibility.
"""
from typing import Dict, Any, Union
import numpy as np

def to_numpy(arr: Any) -> np.ndarray:
    """Converts PyTorch tensor or array-like to NumPy array cleanly."""
    if hasattr(arr, "detach"):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)

def compute_segmentation_metrics(
    pred: Any,
    target: Any,
    ignore_index: int = -1,
    num_classes: int = 2
) -> Dict[str, float]:
    """
    Computes binary/multiclass segmentation metrics while ignoring masked/nodata pixels.
    Args:
        pred: Predicted class indices (H, W) or (B, H, W)
        target: Ground truth class indices (H, W) or (B, H, W)
        ignore_index: Label value to exclude from evaluation (default -1)
        num_classes: Number of active classes (e.g. 2 for non-water=0, water=1)
    """
    pred_arr = to_numpy(pred)
    target_arr = to_numpy(target)

    # Flatten arrays
    pred_flat = pred_arr.flatten()
    target_flat = target_arr.flatten()

    # Mask valid pixels only
    valid_mask = target_flat != ignore_index
    valid_pred = pred_flat[valid_mask]
    valid_target = target_flat[valid_mask]

    total_valid = int(valid_target.size)
    if total_valid == 0:
        return {
            "valid_pixels": 0,
            "accuracy": 0.0,
            "mean_iou": 0.0,
            "water_iou": 0.0,
            "land_iou": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    # Accuracy
    accuracy = float((valid_pred == valid_target).sum() / total_valid)

    # Per-class IoU and F1
    ious = []
    class_metrics = {}

    for cls in range(num_classes):
        tp = int(((valid_pred == cls) & (valid_target == cls)).sum())
        fp = int(((valid_pred == cls) & (valid_target != cls)).sum())
        fn = int(((valid_pred != cls) & (valid_target == cls)).sum())

        union = tp + fp + fn
        iou = float(tp / union) if union > 0 else (1.0 if (tp == 0 and fp == 0 and fn == 0) else 0.0)
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        ious.append(iou)
        cls_name = "water" if cls == 1 else "land"
        class_metrics[f"{cls_name}_iou"] = iou
        class_metrics[f"{cls_name}_precision"] = precision
        class_metrics[f"{cls_name}_recall"] = recall
        class_metrics[f"{cls_name}_f1"] = f1

    mean_iou = float(np.mean(ious))
    
    metrics = {
        "valid_pixels": total_valid,
        "ignored_pixels": int((~valid_mask).sum()),
        "accuracy": accuracy,
        "mean_iou": mean_iou,
        "macro_f1": float(np.mean([class_metrics["land_f1"], class_metrics["water_f1"]])),
        **class_metrics
    }

    return metrics

class MetricAccumulator:
    """Accumulates confusion matrix across multiple batches/chips for dataset-level metrics."""
    def __init__(self, num_classes: int = 2, ignore_index: int = -1):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.total_ignored = 0

    def update(self, pred: Any, target: Any):
        p = to_numpy(pred).flatten()
        t = to_numpy(target).flatten()

        valid = t != self.ignore_index
        self.total_ignored += int((~valid).sum())

        p_valid = p[valid]
        t_valid = t[valid]

        for i in range(len(p_valid)):
            if 0 <= t_valid[i] < self.num_classes and 0 <= p_valid[i] < self.num_classes:
                self.confusion_matrix[t_valid[i], p_valid[i]] += 1

    def compute(self) -> Dict[str, Any]:
        cm = self.confusion_matrix
        tp = np.diag(cm)
        sum_p = cm.sum(axis=0)  # predicted per class
        sum_t = cm.sum(axis=1)  # ground truth per class

        ious = []
        precisions = []
        recalls = []
        f1s = []

        for i in range(self.num_classes):
            union = sum_t[i] + sum_p[i] - tp[i]
            iou = float(tp[i] / union) if union > 0 else 0.0
            prec = float(tp[i] / sum_p[i]) if sum_p[i] > 0 else 0.0
            rec = float(tp[i] / sum_t[i]) if sum_t[i] > 0 else 0.0
            f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

            ious.append(iou)
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

        total_correct = int(tp.sum())
        total_pixels = int(cm.sum())

        return {
            "total_pixels_evaluated": total_pixels,
            "total_ignored_pixels": self.total_ignored,
            "accuracy": float(total_correct / total_pixels) if total_pixels > 0 else 0.0,
            "mean_iou": float(np.mean(ious)),
            "macro_f1": float(np.mean(f1s)),
            "water_iou": ious[1] if self.num_classes > 1 else 0.0,
            "water_precision": precisions[1] if self.num_classes > 1 else 0.0,
            "water_recall": recalls[1] if self.num_classes > 1 else 0.0,
            "water_f1": f1s[1] if self.num_classes > 1 else 0.0,
            "confusion_matrix": cm.tolist()
        }
