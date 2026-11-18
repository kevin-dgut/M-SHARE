import csv
import os

import numpy as np
import scipy.io as sio
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Subset


def normalize_label(label):
    return str(label).strip().lower()


def build_binary_labels(records, class0_labels, class1_labels):
    class0_set = {normalize_label(label) for label in class0_labels}
    class1_set = {normalize_label(label) for label in class1_labels}
    overlap = class0_set & class1_set
    if overlap:
        raise ValueError(f"class0_labels and class1_labels contain overlapping labels: {sorted(overlap)}")

    label_map = {label: 0 for label in class0_set}
    label_map.update({label: 1 for label in class1_set})
    labels = []
    for record in records:
        group = normalize_label(record["group"])
        if group not in label_map:
            raise ValueError(f"Unknown group {record['group']}; assign it using --class0_labels or --class1_labels")
        labels.append(label_map[group])
    return torch.tensor(labels, dtype=torch.long)


def load_subject_records(
    data_path,
    label_path,
    class0_labels=None,
    class1_labels=None,
):
    if label_path is None:
        raise ValueError("label_path is required and should point to subject_labels.csv")
    if not os.path.isfile(label_path):
        raise FileNotFoundError(f"label_path does not exist: {label_path}")

    labels = list(class0_labels or []) + list(class1_labels or [])
    selected_labels = {normalize_label(label) for label in labels}
    records = []
    with open(label_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"new_subject_id", "label_name"}
        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(f"{label_path} must contain new_subject_id,label_name")

        for row in reader:
            group = row["label_name"].strip()
            if normalize_label(group) not in selected_labels:
                continue

            subject_id = row["new_subject_id"].strip()
            filename = f"ROISignals_ROISignal_{subject_id}.mat"
            path = os.path.join(data_path, filename)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"{path} is missing for subject {subject_id}")

            records.append(
                {
                    "group": group,
                    "subject_id": subject_id,
                    "filename": filename,
                    "path": path,
                }
            )

    if not records:
        raise FileNotFoundError(f"{label_path} has no samples matching class0/class1 labels")
    return records


def load_roisignals(path):
    mat = sio.loadmat(path)
    if "ROISignals" not in mat:
        raise KeyError(f"{path} does not contain ROISignals")
    return mat["ROISignals"].astype(np.float32)


def load_dataset(
    data_path,
    seq_len=130,
    class0_labels=None,
    class1_labels=None,
    label_path=None,
):
    records = load_subject_records(
        data_path,
        label_path,
        class0_labels,
        class1_labels,
    )
    arrays = []
    for record in records:
        path = record["path"]
        array = load_roisignals(path)
        if array.ndim != 2:
            raise ValueError(f"{path} should be 2-D, got shape {array.shape}")
        if array.shape[0] == seq_len:
            array = array.T
        elif array.shape[1] != seq_len:
            raise ValueError(
                f"{path} should have one dimension equal to seq_len={seq_len}, got {array.shape}"
            )
        arrays.append(array)
    data = np.stack(arrays, axis=0).astype(np.float32)
    return torch.from_numpy(data).float(), records


class SingleAtlasTensorDataset(Dataset):
    def __init__(self, x, records, labels):
        if x.ndim != 3:
            raise ValueError(f"x should be B x ROI x T, got {tuple(x.shape)}")
        if len(records) != x.shape[0] or labels.shape[0] != x.shape[0]:
            raise ValueError("records, labels, and x must have the same sample count")
        self.x = x.float()
        self.records = records
        self.labels = labels.long()

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        return {"x": self.x[index], "label": self.labels[index], "record": self.records[index]}


def collate_single_atlas_batch(batch):
    x = torch.stack([item["x"] for item in batch], dim=0)
    labels = torch.stack([item["label"] for item in batch], dim=0).long()
    records = [item["record"] for item in batch]
    return x, labels, records


def build_single_atlas_fold_loaders(
    dataset,
    train_indices,
    val_indices,
    batch_size,
    seed,
    num_workers=0,
):
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        collate_fn=collate_single_atlas_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_single_atlas_batch,
    )
    return train_loader, val_loader


def stratified_kfold_indices(
    labels,
    n_folds,
    seed,
):
    labels_np = (
        labels.cpu().long().numpy()
        if torch.is_tensor(labels)
        else np.asarray(labels, dtype=np.int64)
    )
    splitter = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=seed,
    )
    dummy_x = np.zeros(labels_np.shape[0], dtype=np.float32)
    return [
        (train_idx.tolist(), val_idx.tolist())
        for train_idx, val_idx in splitter.split(dummy_x, labels_np)
    ]
