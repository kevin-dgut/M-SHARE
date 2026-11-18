import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MTAFMBranch(nn.Module):

    def __init__(
        self,
        roi_num,
        seq_len,
        scales,
        window_counts,
        conv_channels,
        hgcn_hidden,
        roi_attention_hidden_dim,
        roi_attention_heads,
        conv_dropout=0.0,
        hgcn_dropout=0.0,
        conv_stride=1,
    ):
        super().__init__()
        self.roi_num = int(roi_num)
        self.seq_len = int(seq_len)
        self.scales = [int(scale) for scale in scales]
        self.scale_num = len(self.scales)
        self.conv_channels = int(conv_channels)
        self.hgcn_hidden = int(hgcn_hidden)
        self.roi_attention_hidden_dim = int(roi_attention_hidden_dim)
        self.roi_attention_heads = int(roi_attention_heads)
        self.conv_stride = int(conv_stride)

        self.window_counts = window_counts
        self.scale_slices = self._make_scale_slices(self.window_counts)
        self.total_windows = sum(self.window_counts)
        self.node_feature_dim = self.conv_channels

        self.scale_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=1,
                        out_channels=self.conv_channels,
                        kernel_size=scale,
                        stride=self.conv_stride,
                        padding=0,
                        bias=False,
                    ),
                    nn.BatchNorm1d(self.conv_channels),
                    nn.GELU(),
                    nn.Dropout(conv_dropout),
                )
                for scale in self.scales
            ]
        )
        h_assign, edge_degree, node_degree = self._make_incidence()
        self.hyperedge_num = int(h_assign.shape[1])
        self.register_buffer("H", h_assign, persistent=False)
        self.register_buffer("edge_degree", edge_degree, persistent=False)
        self.register_buffer("node_degree", node_degree, persistent=False)
        self.hgcn_projs = nn.ModuleList(
            [nn.Linear(self.node_feature_dim, self.hgcn_hidden, bias=False) for _ in range(self.roi_num)]
        )
        self.hgcn_dropout = nn.Dropout(hgcn_dropout)

        self.roi_feature_dim = self.total_windows * self.hgcn_hidden
        self.head_dim = self.roi_attention_hidden_dim // self.roi_attention_heads
        self.roi_attention_norm = nn.LayerNorm(self.roi_feature_dim)
        self.q_proj = nn.Linear(self.roi_feature_dim, self.roi_attention_hidden_dim, bias=False)
        self.k_proj = nn.Linear(self.roi_feature_dim, self.roi_attention_hidden_dim, bias=False)
        self.v_proj = nn.Linear(self.roi_feature_dim, self.roi_attention_hidden_dim, bias=False)

    def forward(self, x):
        embedding_output = self.conv_windows_embedding(x)
        hgcn_output = self.HGCN_encoder(embedding_output["scale_windows"])
        mha_output = self.MHA_RR(hgcn_output["hgcn_out"])
        return {
            "scale_windows": embedding_output["scale_windows"],
            "hgcn_out": hgcn_output["hgcn_out"],  # [B, ROI, total_windows, hgcn_hidden].
            "graph_feature": mha_output["graph_feature"],  # [B, ROI, roi_attention_hidden_dim].
            "roi_matrix": mha_output["roi_matrix"],
        }

    def conv_windows_embedding(self, x):
        if x.ndim != 3 or x.shape[1] != self.roi_num or x.shape[2] != self.seq_len:
            raise ValueError(f"expected B x {self.roi_num} x {self.seq_len}, got {tuple(x.shape)}")

        batch_size = x.shape[0]
        roi_signal = x.reshape(batch_size * self.roi_num, 1, self.seq_len)
        scale_windows = []

        for window_count, conv in zip(self.window_counts, self.scale_convs):
            window = conv(roi_signal).transpose(1, 2).contiguous()
            window = window.view(batch_size, self.roi_num, window_count, self.conv_channels)
            scale_windows.append(window)

        return {"scale_windows": scale_windows}

    def HGCN_encoder(self, scale_windows):
        window_nodes = torch.cat(scale_windows, dim=2)

        roi_hgcn_outputs = []
        for roi_idx in range(self.roi_num):
            node_feature = window_nodes[:, roi_idx]

            propagated = self._propagate_window_nodes(node_feature)
            updated_nodes = self.hgcn_dropout(F.gelu(self.hgcn_projs[roi_idx](propagated)))

            roi_hgcn_outputs.append(updated_nodes)

        hgcn_out = torch.stack(roi_hgcn_outputs, dim=1)  # [B, ROI, Window, hidden_dim].
        return {"hgcn_out": hgcn_out}

    def _propagate_window_nodes(self, node_feature):
        # Aggregate window-node context through scale and time-point hyperedges.
        # The output has the same shape as the input window-node features.
        raise NotImplementedError("Core hypergraph propagation is not included.")

    def MHA_RR(self, hgcn_out):

        batch_size = hgcn_out.shape[0]
        roi_features = hgcn_out.reshape(batch_size, self.roi_num, -1)  # [B, ROI, Window, hidden_dim] -> [B, ROI, Window*hidden_dim].
        roi_features = self.roi_attention_norm(roi_features)

        q = self.q_proj(roi_features).view(batch_size, self.roi_num, self.roi_attention_heads, self.head_dim)
        k = self.k_proj(roi_features).view(batch_size, self.roi_num, self.roi_attention_heads, self.head_dim)
        v = self.v_proj(roi_features).view(batch_size, self.roi_num, self.roi_attention_heads, self.head_dim)
        q = q.transpose(1, 2)  # [B, head, ROI, head_dim].
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attention = scores.softmax(dim=-1)
        graph_feature = torch.matmul(attention, v)
        graph_feature = graph_feature.transpose(1, 2).contiguous().view(
            batch_size,
            self.roi_num,
            self.roi_attention_hidden_dim,
        )
        roi_matrix = attention.mean(dim=1)
        eye = torch.eye(self.roi_num, device=roi_matrix.device, dtype=roi_matrix.dtype).unsqueeze(0)
        roi_matrix = roi_matrix * (1.0 - eye)
        return {"graph_feature": graph_feature, "roi_matrix": roi_matrix}

    @staticmethod
    def _make_scale_slices(window_counts):
        slices, start = [], 0
        for count in window_counts:
            slices.append((start, start + count))
            start += count
        return slices

    def _make_incidence(self):
        node_num = self.total_windows
        edge_num = self.scale_num + self.seq_len
        h_assign = torch.zeros(node_num, edge_num)

        for scale_idx, (start, end) in enumerate(self.scale_slices):
            h_assign[start:end, scale_idx] = 1.0

        for scale, window_count, (scale_start, _) in zip(
            self.scales, self.window_counts, self.scale_slices
        ):
            for window_idx in range(window_count):
                start_time = window_idx * self.conv_stride
                end_time = start_time + scale
                h_assign[
                    scale_start + window_idx,
                    self.scale_num + start_time : self.scale_num + end_time,
                ] = 1.0

        edge_degree = h_assign.sum(dim=0).clamp_min(1.0)
        node_degree = h_assign.sum(dim=1).clamp_min(1.0)
        return h_assign.float(), edge_degree.float(), node_degree.float()
