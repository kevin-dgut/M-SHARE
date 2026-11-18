import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch


def parse_int_list(text):
    if isinstance(text, (list, tuple)):
        return [int(item) for item in text]
    return [int(item.strip()) for item in str(text).split(",") if item.strip()]


def parse_float_list(text):
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def parse_label_list(text):
    return [item.strip() for item in str(text).split(",") if item.strip()]


def normalize_path(path):
    path_text = str(path)
    if len(path_text) >= 3 and path_text[1] == ":" and path_text[2] in ("\\", "/") and os.name != "nt":
        drive = path_text[0].lower()
        rest = path_text[3:].replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    if path_text.startswith("/mnt/"):
        return path_text
    return str(Path(path_text).expanduser())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Single-atlas multi-scale ROI hypergraph runner")

    parser.add_argument("--data_path", default=None, type=str)
    parser.add_argument("--label_path", default=None, type=str)
    parser.add_argument("--log_path", default="./logs/train.log", type=str)
    parser.add_argument("--fold_model_dir", default="./models", type=str)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)

    parser.add_argument("--model_mode", default="train", choices=["train", "eval"], type=str)
    parser.add_argument("--seq_len", default=130, type=int)#ADNI
    parser.add_argument("--roi_num", default=116, type=int)
    parser.add_argument("--class0_labels", default="NC", type=str)#ADNI
    parser.add_argument("--class1_labels", default="EMCI", type=str)#ADNI

    parser.add_argument("--scale_candidates", default="2,4,8,16,32", type=str)
    parser.add_argument("--conv_channels", default=4, type=int)
    parser.add_argument("--conv_stride", default=1, type=int)
    parser.add_argument("--hgcn_hidden", default=8, type=int)

    parser.add_argument("--roi_attention_heads", default=4, type=int)
    parser.add_argument("--roi_attention_hidden_dim", default=128, type=int)
    parser.add_argument("--cls_hidden", default=32, type=int)
    
    parser.add_argument("--scale_evolution_hidden_dim", default=128, type=int)
    parser.add_argument("--scale_evolution_attention_hidden_dim", default=64, type=int, help="ROI attention hidden size shared by all scales.")
    parser.add_argument("--scale_evolution_attention_heads", default=4, type=int, help="ROI attention head count shared by all scales.")
    parser.add_argument("--scale_evolution_dropout", default=0, type=float, help="Dropout used inside the scale-evolution branch.")
    parser.add_argument("--scale_evolution_aux_loss_weight", default=0.0, type=float, help="Weight for the SGEM/GRU auxiliary classification loss.")

    parser.add_argument("--conv_dropout", default=0, type=float)
    parser.add_argument("--hgcn_dropout", default=0, type=float)

    parser.add_argument("--use_class_weight", action="store_true", default=True)
    parser.add_argument("--class_weights", default="", type=str)
    parser.add_argument("--label_smoothing", default=0, type=float)

    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--n_folds", default=5, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--lr", default=0.00005, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--ema_decay", default=1, type=float)

    parser.add_argument(
        "--best_metric",
        default="val_acc",
        choices=["val_loss", "val_acc", "val_balanced_acc", "val_auc", "val_f1"],
        type=str,
    )

    args = parser.parse_args(argv)
    if not 0.0 <= args.ema_decay <= 1.0:
        parser.error("--ema_decay must be between 0 and 1; use 1 to disable EMA")
    args.data_path = normalize_path(args.data_path)
    args.label_path = normalize_path(args.label_path)
    args.log_path = normalize_path(args.log_path)
    args.fold_model_dir = normalize_path(args.fold_model_dir)
    args.scale_candidates = sorted(parse_int_list(args.scale_candidates))
    args.class_weights = parse_float_list(args.class_weights) if str(args.class_weights).strip() else None
    if args.class_weights is not None:
        if len(args.class_weights) != 2:
            raise ValueError("--class_weights requires two comma-separated values, e.g. 1,2.5")
        if any(weight <= 0.0 for weight in args.class_weights):
            raise ValueError("Both --class_weights values must be greater than 0")
    args.class0_labels = parse_label_list(args.class0_labels)
    args.class1_labels = parse_label_list(args.class1_labels)
    args.device = resolve_device(args.device)
    set_seed(args.seed)
    return args


def resolve_device(device_text):
    if str(device_text).startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_text)
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
