import torch
import torch.nn as nn

from MTAFM import MTAFMBranch
from SGEM import ScaleEvolutionBranch


class ScaleHypergraphROINet(nn.Module):

    def __init__(
        self,
        roi_num,
        seq_len,
        scales,
        conv_channels,
        hgcn_hidden,
        roi_attention_hidden_dim,
        cls_hidden,
        num_classes=2,
        conv_dropout=0.0,
        hgcn_dropout=0.0,
        roi_attention_heads=2,
        conv_stride=1,
        scale_evolution_hidden_dim=64,
        scale_evolution_attention_hidden_dim=64,
        scale_evolution_attention_heads=4,
        scale_evolution_dropout=0.0,
    ):
        super().__init__()
        self.roi_num = int(roi_num)
        self.seq_len = int(seq_len)
        self.scales = sorted(int(scale) for scale in scales)
        conv_channels = int(conv_channels)
        hgcn_hidden = int(hgcn_hidden)
        roi_attention_hidden_dim = int(roi_attention_hidden_dim)
        roi_attention_heads = int(roi_attention_heads)
        cls_hidden = int(cls_hidden)
        num_classes = int(num_classes)
        scale_evolution_hidden_dim = int(scale_evolution_hidden_dim)
        scale_evolution_dropout = float(scale_evolution_dropout)
        conv_stride = int(conv_stride)

        window_counts = [
            (self.seq_len - scale) // conv_stride + 1 for scale in self.scales
        ]

        self.mtafm_branch = MTAFMBranch(
            roi_num=self.roi_num,
            seq_len=self.seq_len,
            scales=self.scales,
            window_counts=window_counts,
            conv_channels=conv_channels,
            hgcn_hidden=hgcn_hidden,
            roi_attention_hidden_dim=roi_attention_hidden_dim,
            roi_attention_heads=roi_attention_heads,
            conv_dropout=conv_dropout,
            hgcn_dropout=hgcn_dropout,
            conv_stride=conv_stride,
        )

        self.scale_evolution_branch = ScaleEvolutionBranch(
            roi_num=self.roi_num,
            window_counts=window_counts,
            conv_channels=conv_channels,
            attention_hidden_dim=scale_evolution_attention_hidden_dim,
            attention_heads=scale_evolution_attention_heads,
            gru_hidden_dim=scale_evolution_hidden_dim,
            dropout=scale_evolution_dropout,
        )

        main_feature_dim = roi_attention_hidden_dim
        branch_feature_dim = scale_evolution_hidden_dim
        cls_input_dim = main_feature_dim + branch_feature_dim
        self.cls_norm = nn.LayerNorm(roi_attention_hidden_dim)
        self.main_sample_norm = nn.LayerNorm(roi_attention_hidden_dim)
        self.scale_evolution_sample_norm = nn.LayerNorm(branch_feature_dim)
        self.cls_fc1 = nn.Linear(cls_input_dim, cls_hidden)
        self.cls_fc2 = nn.Linear(cls_hidden, num_classes)
        self.scale_evolution_aux_head = nn.Linear(branch_feature_dim, num_classes)

    def forward(self, x):
        mtafm_output = self.mtafm_branch(x)
        graph_feature = mtafm_output["graph_feature"]
        scale_windows = mtafm_output["scale_windows"]
        main_classifier_feature = self._pool_graph_feature(graph_feature)
        main_classifier_feature = self.main_sample_norm(main_classifier_feature)
        full_scale_fc = mtafm_output["roi_matrix"]
        scale_evolution_output = self.scale_evolution_branch(scale_windows, full_scale_fc=full_scale_fc)
        scale_evolution_feature = self.scale_evolution_sample_norm(scale_evolution_output["sample_vector"])
        scale_evolution_logits = self.scale_evolution_aux_head(scale_evolution_feature)
        pre_fusion_norm_feature = torch.cat([main_classifier_feature, scale_evolution_feature], dim=-1)

        main_classifier_feature = pre_fusion_norm_feature
        logits, penultimate_feature = self._classify(main_classifier_feature, return_feature=True)

        return {
            "logits": logits,
            "roi_matrix": mtafm_output["roi_matrix"],
            "graph_feature": graph_feature,
            "pre_fusion_norm_feature": pre_fusion_norm_feature,
            "main_classifier_feature": main_classifier_feature,
            "penultimate_feature": penultimate_feature,
            "scale_evolution_feature": scale_evolution_feature,
            "scale_evolution_output": scale_evolution_output,
            "scale_evolution_logits": scale_evolution_logits,
        }

    def _pool_graph_feature(self, graph_feature):
        normalized = self.cls_norm(graph_feature)
        sample_feature = normalized.mean(dim=1)
        return sample_feature

    def _classify(self, main_classifier_feature, return_feature=False):
        hidden = self.cls_fc1(main_classifier_feature)
        logits = self.cls_fc2(hidden)
        if return_feature:
            return logits, hidden
        return logits
