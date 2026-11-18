import math

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def compute_binary_metrics(logits, labels):
    probs = torch.softmax(logits.float(), dim=1)[:, 1].detach().cpu().numpy()
    preds = (probs >= 0.5).astype(np.int64)
    labels = labels.detach().cpu().long().numpy() if torch.is_tensor(labels) else np.asarray(labels, dtype=np.int64)
    tn = int(np.sum((labels == 0) & (preds == 0)))
    fp = int(np.sum((labels == 0) & (preds == 1)))
    fn = int(np.sum((labels == 1) & (preds == 0)))
    tp = int(np.sum((labels == 1) & (preds == 1)))
    has_positive = np.any(labels == 1)
    has_negative = np.any(labels == 0)
    acc = float(accuracy_score(labels, preds)) if labels.size else float("nan")
    sensitivity = float(recall_score(labels, preds, pos_label=1, zero_division=0)) if has_positive else float("nan")
    specificity = float(recall_score(labels, preds, pos_label=0, zero_division=0)) if has_negative else float("nan")
    precision = float(precision_score(labels, preds, pos_label=1, zero_division=0)) if labels.size else float("nan")
    f1 = float(f1_score(labels, preds, pos_label=1, zero_division=0)) if has_positive else float("nan")
    auc = float(roc_auc_score(labels, probs)) if has_positive and has_negative else float("nan")
    balanced_acc = (sensitivity + specificity) / 2.0 if not (math.isnan(sensitivity) or math.isnan(specificity)) else float("nan")
    return {
        "acc": acc,
        "auc": auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "balanced_acc": balanced_acc,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def select_best_score(val_stats, best_metric):
    metric_map = {
        "val_loss": "total_loss",
        "val_acc": "acc",
        "val_balanced_acc": "balanced_acc",
        "val_auc": "auc",
        "val_f1": "f1",
    }
    return val_stats[metric_map[best_metric]]


def is_better_score(score, best_score, best_metric):
    if math.isnan(score):
        return False
    if best_score is None:
        return True
    if best_metric == "val_loss":
        return score < best_score
    return score > best_score


def nanmean_std(values):
    array = np.array(values, dtype=np.float64)
    if np.all(np.isnan(array)):
        return float("nan"), float("nan")
    return float(np.nanmean(array)), float(np.nanstd(array))


def format_metric(value):
    return "nan" if math.isnan(value) else f"{value:.4f}"


def format_stats(stats):
    text = (
        f"loss={stats['total_loss']:.6f} "
        f"acc={format_metric(stats['acc'])} "
        f"auc={format_metric(stats['auc'])} "
        f"sen={format_metric(stats['sensitivity'])} "
        f"spe={format_metric(stats['specificity'])} "
        f"f1={format_metric(stats['f1'])}"
    )
    return text
