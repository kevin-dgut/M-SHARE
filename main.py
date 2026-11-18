import os
from copy import deepcopy

import torch
from tqdm import tqdm

from dataload import (
    SingleAtlasTensorDataset,
    build_binary_labels,
    build_single_atlas_fold_loaders,
    load_dataset,
    stratified_kfold_indices,
)
from loss import ClassificationLoss
from metrics import compute_binary_metrics, is_better_score, select_best_score
from model import ScaleHypergraphROINet
from opt import parse_args, set_seed
from print_ import (
    Logger,
    print_best_fold_summary,
    print_best_model_saved,
    print_class_weight,
    print_epoch_progress,
    print_experiment_config,
    print_fold_header,
    print_fold_val_result,
    print_full_run_args,
    print_summary,
)


def main():
    args = parse_args()
    with Logger(args.log_path):
        run_cross_validation(args)


def run_cross_validation(args):
    print_full_run_args(args)
    x, records = load_dataset(
        args.data_path,
        args.seq_len,
        args.class0_labels,
        args.class1_labels,
        args.label_path,
    )
    labels = build_binary_labels(records, args.class0_labels, args.class1_labels)
    dataset = SingleAtlasTensorDataset(x, records, labels)
    fold_splits = stratified_kfold_indices(labels, args.n_folds, args.seed)
    print_experiment_config(args, x, labels)
    fold_stats = []
    best_fold_records = []

    for fold_idx, (train_indices, val_indices) in enumerate(fold_splits):
        print_fold_header(fold_idx, args.n_folds)
        train_loader, val_loader = build_single_atlas_fold_loaders(
            dataset,
            train_indices,
            val_indices,
            args.batch_size,
            args.seed,
            args.num_workers,
        )
        set_seed(args.seed)
        model = create_model(args)
        criterion = create_criterion(args, labels, train_indices).to(args.device)
        optimizer = create_optimizer(model, args)
        model_path = build_fold_model_path(args, fold_idx)
        best_record = None

        if args.model_mode == "train":
            best_record = train_fold(model, criterion, optimizer, train_loader, val_loader, args, fold_idx, model_path)

        val_stats = evaluate_saved_fold(
            model,
            criterion,
            val_loader,
            args.device,
            model_path,
        )
        fold_stats.append(val_stats)
        best_fold_records.append(best_record)
        print_fold_val_result(fold_idx, val_stats)

    summary = print_summary(fold_stats, args.n_folds)
    print_best_fold_summary(best_fold_records)
    return {
        "fold_stats": fold_stats,
        "summary": summary,
        "best_fold_records": best_fold_records,
    }


def create_model(args):
    model = ScaleHypergraphROINet(
        roi_num=args.roi_num,
        seq_len=args.seq_len,
        scales=args.scale_candidates,
        conv_channels=args.conv_channels,
        conv_stride=args.conv_stride,
        hgcn_hidden=args.hgcn_hidden,
        roi_attention_hidden_dim=args.roi_attention_hidden_dim,
        cls_hidden=args.cls_hidden,
        conv_dropout=args.conv_dropout,
        hgcn_dropout=args.hgcn_dropout,
        roi_attention_heads=args.roi_attention_heads,
        scale_evolution_hidden_dim=args.scale_evolution_hidden_dim,
        scale_evolution_attention_hidden_dim=args.scale_evolution_attention_hidden_dim,
        scale_evolution_attention_heads=args.scale_evolution_attention_heads,
        scale_evolution_dropout=args.scale_evolution_dropout,  # SGEM dropout.
    )
    return model.to(args.device)


def create_optimizer(model, args):
    return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def create_criterion(args, labels, train_indices):
    if args.class_weights is not None:
        class_weights = torch.tensor(args.class_weights, dtype=torch.float32)
    else:
        class_weights = compute_class_weights(labels, train_indices) if args.use_class_weight else None
    if class_weights is not None:
        print_class_weight(class_weights)
    return ClassificationLoss(
        class_weights=class_weights,
        label_smoothing=args.label_smoothing,
        scale_evolution_aux_weight=args.scale_evolution_aux_loss_weight,
    )


def create_ema_model(model):
    ema_model = deepcopy(model).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    return ema_model


@torch.no_grad()
def update_ema_model(ema_model, model, decay):
    model_state = model.state_dict()
    for name, ema_value in ema_model.state_dict().items():
        current_value = model_state[name].detach()
        if torch.is_floating_point(ema_value):
            ema_value.mul_(decay).add_(current_value, alpha=1.0 - decay)
        else:
            ema_value.copy_(current_value)


def train_fold(model, criterion, optimizer, train_loader, val_loader, args, fold_idx, model_path):
    best_score = None
    best_record = None
    ema_model = create_ema_model(model) if args.ema_decay < 1.0 else None
    # Keep the optimizer learning rate fixed for the entire training run.

    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(
            model,
            criterion,
            optimizer,
            train_loader,
            args.device,
            training=True,
            ema_model=ema_model,
            ema_decay=args.ema_decay,
        )
        validation_model = ema_model if ema_model is not None else model
        val_stats = run_epoch(
            validation_model,
            criterion,
            optimizer,
            val_loader,
            args.device,
            training=False,
        )
        print_epoch_progress(fold_idx, epoch, args.epochs, train_stats, val_stats)
        score = select_best_score(val_stats, args.best_metric)

        if is_better_score(score, best_score, args.best_metric):
            best_score = score
            best_record = build_best_fold_record(fold_idx, epoch, args.best_metric, score, val_stats, model_path)
            save_best_model(validation_model, model_path, epoch, args.best_metric, score)


    return best_record


def run_epoch(
    model,
    criterion,
    optimizer,
    data_loader,
    device,
    training,
    ema_model=None,
    ema_decay=0.99,
):
    model.train(training)
    class_weights = criterion.classification_loss.weight
    total_loss_weight = 0.0
    total_loss_sum = 0.0
    cls_loss_sum = 0.0
    scale_evolution_aux_loss_sum = 0.0
    has_scale_evolution_aux_loss = False
    logits_list = []
    labels_list = []
    grad_context = torch.enable_grad() if training else torch.no_grad()

    with grad_context:
        for batch_x, batch_labels, _ in tqdm(data_loader, desc="Train" if training else "Val", leave=False):
            batch_x = batch_x.to(device=device, dtype=torch.float32)
            batch_labels = batch_labels.to(device=device, dtype=torch.long)

            if training:
                optimizer.zero_grad(set_to_none=True)

            output = model(batch_x)
            loss_info = criterion(output, batch_labels)

            if training:
                loss_info["total_loss"].backward()
                optimizer.step()
                if ema_model is not None:
                    update_ema_model(ema_model, model, ema_decay)

            batch_loss_weight = (
                float(batch_labels.numel())
                if class_weights is None
                else class_weights[batch_labels].sum().item()
            )
            total_loss_weight += batch_loss_weight
            total_loss_sum += loss_info["total_loss"].detach().item() * batch_loss_weight
            cls_loss_sum += loss_info["classification_loss"].detach().item() * batch_loss_weight
            if "scale_evolution_aux_loss" in loss_info:
                has_scale_evolution_aux_loss = True
                scale_evolution_aux_loss_sum += loss_info["scale_evolution_aux_loss"].detach().item() * batch_loss_weight
            logits_list.append(output["logits"].detach().cpu())
            labels_list.append(batch_labels.detach().cpu())

    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    stats = compute_binary_metrics(logits, labels)
    stats["total_loss"] = total_loss_sum / total_loss_weight
    stats["cls_loss"] = cls_loss_sum / total_loss_weight
    if has_scale_evolution_aux_loss:
        stats["scale_evolution_aux_loss"] = scale_evolution_aux_loss_sum / total_loss_weight
    return stats


def evaluate_saved_fold(
    model,
    criterion,
    data_loader,
    device,
    model_path,
):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"{model_path} does not exist; train this fold first with --model_mode train")
    model.load_state_dict(torch.load(model_path, map_location=device))
    return run_epoch(
        model,
        criterion,
        None,
        data_loader,
        device,
        training=False,
    )


def compute_class_weights(labels, train_indices):
    train_labels = labels[train_indices].long()
    class_counts = torch.bincount(train_labels, minlength=2).float()
    if torch.any(class_counts == 0):
        raise ValueError(f"The current training fold has an empty class; cannot compute class weights: {class_counts.tolist()}")
    return train_labels.numel() / (2.0 * class_counts)


def build_fold_model_path(args, fold_idx):
    os.makedirs(args.fold_model_dir, exist_ok=True)
    return os.path.join(args.fold_model_dir, f"scale_hgcn_fold{fold_idx}.pth")


def save_best_model(model, save_path, epoch, best_metric, score):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print_best_model_saved(epoch, best_metric, score, save_path)


def build_best_fold_record(fold_idx, epoch, best_metric, score, val_stats, model_path):
    return {
        "fold": fold_idx + 1,
        "epoch": epoch,
        "best_metric": best_metric,
        "score": score,
        "stats": clone_stats(val_stats),
        "model_path": model_path,
    }


def clone_stats(stats):
    cloned = {}
    for key, value in stats.items():
        if isinstance(value, list):
            cloned[key] = [row[:] if isinstance(row, list) else row for row in value]
        else:
            cloned[key] = value
    return cloned


if __name__ == "__main__":
    main()
