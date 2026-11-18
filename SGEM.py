import torch
import torch.nn as nn


class ScaleROIAttentionFC(nn.Module):
    def __init__(self, input_dim, hidden_dim, head_num, dropout=0.0, zero_diag=True):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.head_num = int(head_num)
        self.zero_diag = bool(zero_diag)
        if self.input_dim <= 0 or self.hidden_dim <= 0 or self.head_num <= 0:
            raise ValueError("input_dim, hidden_dim and head_num must be positive")
        if self.hidden_dim % self.head_num != 0:
            raise ValueError(
                "hidden_dim must be divisible by head_num, got "
                f"{self.hidden_dim} and {self.head_num}"
            )

        self.head_dim = self.hidden_dim // self.head_num
        self.q_proj = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        self.k_proj = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, roi_features):
        if roi_features.ndim != 3 or roi_features.shape[-1] != self.input_dim:
            raise ValueError(f"expected B x R x {self.input_dim}, got {tuple(roi_features.shape)}")

        # Derive a scale-specific ROI relation matrix from the window features.
        # The output shape is [batch, ROI, ROI].
        raise NotImplementedError("Core scale-specific ROI relation computation is not included.")

class MatrixGRUCell(nn.Module):
    def __init__(self, roi_num, hidden_dim):
        super().__init__()
        self.roi_num = int(roi_num)
        self.hidden_dim = int(hidden_dim)
        if self.roi_num <= 0 or self.hidden_dim <= 0:
            raise ValueError("roi_num and hidden_dim must be positive")

        self.x_z = nn.Linear(self.roi_num, self.hidden_dim)
        self.h_z = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.x_r = nn.Linear(self.roi_num, self.hidden_dim)
        self.h_r = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.x_h = nn.Linear(self.roi_num, self.hidden_dim)
        self.h_h = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

    def forward(self, fc_matrix, hidden):

        update_gate = torch.sigmoid(self.x_z(fc_matrix) + self.h_z(hidden))
        reset_gate = torch.sigmoid(self.x_r(fc_matrix) + self.h_r(hidden))
        candidate = torch.tanh(self.x_h(fc_matrix) + self.h_h(reset_gate * hidden))
        return (1.0 - update_gate) * hidden + update_gate * candidate


class MatrixGRUScaleEvolution(nn.Module):
    def __init__(self, roi_num, hidden_dim):
        super().__init__()
        self.roi_num = int(roi_num)
        self.hidden_dim = int(hidden_dim)
        self.cell = MatrixGRUCell(self.roi_num, self.hidden_dim)

    def forward(self, fc_sequence):
        if fc_sequence.ndim != 4 or fc_sequence.shape[2] != self.roi_num or fc_sequence.shape[3] != self.roi_num:
            raise ValueError(f"expected FC sequence B x S x {self.roi_num} x {self.roi_num}, got {tuple(fc_sequence.shape)}")

        batch_size, scale_num, roi_num, _ = fc_sequence.shape
        hidden = torch.zeros(batch_size,roi_num,self.hidden_dim,dtype=fc_sequence.dtype,device=fc_sequence.device)
        hidden_steps = []
        for scale_idx in range(scale_num):
            hidden = self.cell(fc_sequence[:, scale_idx], hidden)
            hidden_steps.append(hidden)
        return hidden, torch.stack(hidden_steps, dim=1)


class ScaleEvolutionBranch(nn.Module):
    def __init__(
        self,
        roi_num,
        window_counts,
        conv_channels,
        attention_hidden_dim,
        attention_heads,
        gru_hidden_dim,
        dropout=0.0,
        zero_diag=True,
    ):
        super().__init__()
        self.roi_num = int(roi_num)
        self.window_counts = [int(count) for count in window_counts]
        self.conv_channels = int(conv_channels)
        self.attention_hidden_dim = int(attention_hidden_dim)
        self.attention_heads = int(attention_heads)
        self.gru_hidden_dim = int(gru_hidden_dim)
        if not self.window_counts:
            raise ValueError("window_counts must not be empty")
        if min(self.window_counts) <= 0 or min(self.roi_num, self.conv_channels) <= 0:
            raise ValueError("roi_num, conv_channels and every window count must be positive")

        self.scale_fc_layers = nn.ModuleList(
            [
                ScaleROIAttentionFC(
                    input_dim=window_count * self.conv_channels,
                    hidden_dim=self.attention_hidden_dim,
                    head_num=self.attention_heads,
                    dropout=dropout,
                    zero_diag=zero_diag,
                )
                for window_count in self.window_counts
            ]
        )
        self.evolution = MatrixGRUScaleEvolution(self.roi_num, self.gru_hidden_dim)
        self.roi_hidden_norm = nn.LayerNorm(self.gru_hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, scale_windows, full_scale_fc=None):
        scale_fc = self.build_scale_fc(scale_windows)
        evolution_fc = scale_fc
        if full_scale_fc is not None:
            if full_scale_fc.ndim != 3 or full_scale_fc.shape[1] != self.roi_num or full_scale_fc.shape[2] != self.roi_num:
                raise ValueError(f"expected full_scale_fc B x {self.roi_num} x {self.roi_num}, got {tuple(full_scale_fc.shape)}")
            full_scale_fc = full_scale_fc.to(device=scale_fc.device, dtype=scale_fc.dtype).unsqueeze(1)
            evolution_fc = torch.cat([scale_fc, full_scale_fc], dim=1)
        roi_hidden, roi_hidden_steps = self.evolution(evolution_fc)
        # normalized_roi_hidden = self.roi_hidden_norm(roi_hidden)
        sample_vector = self.dropout(roi_hidden.mean(dim=1))
        return {
            "sample_vector": sample_vector,
            "scale_fc": scale_fc,
            "evolution_fc": evolution_fc,
            "full_scale_fc": None if full_scale_fc is None else full_scale_fc.squeeze(1),
            "roi_hidden": roi_hidden,
            "roi_hidden_steps": roi_hidden_steps,
        }

    def build_scale_fc(self, scale_windows):
        if len(scale_windows) != len(self.scale_fc_layers):
            raise ValueError(f"expected {len(self.scale_fc_layers)} scale tensors, got {len(scale_windows)}")

        fc_matrices = []
        for scale_window, fc_layer in zip(
            scale_windows,
            self.scale_fc_layers,
        ):
            roi_features = scale_window.flatten(start_dim=2)
            fc_matrix = fc_layer(roi_features)
            fc_matrices.append(fc_matrix)
        return torch.stack(fc_matrices, dim=1)
