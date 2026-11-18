import torch
import torch.nn as nn


class ClassificationLoss(nn.Module):
    def __init__(
        self,
        class_weights=None,
        label_smoothing=0.0,
        scale_evolution_aux_weight=0.0,
    ):
        super().__init__()
        if class_weights is not None and not torch.is_tensor(class_weights):
            class_weights = torch.as_tensor(class_weights, dtype=torch.float32)
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}")
        self.classification_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
        self.scale_evolution_aux_weight = float(scale_evolution_aux_weight)

    def forward(self, output, labels):
        logits = output["logits"]
        labels = self._format_labels(labels, logits.device)
        cls_loss = self.classification_loss(logits, labels)
        total_loss = cls_loss
        loss_info = {"total_loss": total_loss, "classification_loss": cls_loss}

        scale_logits = output.get("scale_evolution_logits")
        if self.scale_evolution_aux_weight > 0.0 and scale_logits is not None:
            scale_loss = self.classification_loss(scale_logits, labels)
            total_loss = total_loss + self.scale_evolution_aux_weight * scale_loss
            loss_info["total_loss"] = total_loss
            loss_info["scale_evolution_aux_loss"] = scale_loss

        return loss_info

    def _format_labels(self, labels, device):
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels)
        labels = labels.to(device=device, dtype=torch.long)
        if labels.ndim != 1:
            labels = labels.view(-1)
        return labels
