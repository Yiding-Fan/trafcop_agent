"""
Context Awareness Module

Implements the mode-switching mechanism that decides whether to use the RL branch
(routine traffic) or LLM branch (emergency scenario). Combines numerical traffic
features with EMV detection signals via an MLP classifier.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from trafcop_agent.config import CONTEXT_AWARE


class ContextAwarenessModule(nn.Module):
    """
    Lightweight MLP classifier for emergency state detection.

    Inputs:
        - numerical_features: traffic detector features (queue lengths, occupancy, pressure)
        - emv_detection: binary/soft EMV presence signals from vision or simulator

    Outputs:
        - emergency_probability: scalar probability of emergency state per environment
        - mode_flag: binary 0 (routine) or 1 (emergency) per environment
    """

    def __init__(self, num_features=32, threshold=None):
        super().__init__()
        self.threshold = threshold or CONTEXT_AWARE["threshold"]
        hidden = CONTEXT_AWARE["hidden_dim"]
        self.classifier = nn.Sequential(
            nn.Linear(num_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, numerical_features):
        """
        Args:
            numerical_features: (B, num_features) concatenated traffic features

        Returns:
            emergency_prob: (B,) probability of emergency state
            mode_flag: (B,) binary mode flags
        """
        logits = self.classifier(numerical_features)
        emergency_prob = torch.sigmoid(logits).squeeze(-1)
        mode_flag = (emergency_prob > self.threshold).int()
        return emergency_prob, mode_flag

    def detect_emergency_from_graph(self, graph_dict):
        """
        Extract features and detect emergency directly from RMTC graph dict.

        Uses EMV position signals as the primary emergency indicator, which
        is equivalent to VLM-based detection when visual inputs are unavailable.

        Args:
            graph_dict: graph state dict from RMTC environment with keys:
                'signal_light', 'vehicle', 'emergency', 'edge_index_dict'

        Returns:
            mode_flag: binary (B,) emergency flag tensor
            emergency_prob: float probability of emergency state
        """
        emv_features = graph_dict.get("emergency", None)
        if emv_features is None or emv_features.numel() == 0:
            return 0, 0.0

        # If any EMV exists in the scene (non-empty emergency node features),
        # classify as emergency mode. This matches VLM detection when visual inputs
        # are unavailable per paper's experimental setup.
        has_emv = emv_features.shape[0] > 0
        prob = float(torch.sigmoid(
            self.classifier(emv_features[:, :32].mean(dim=0, keepdim=True))
        ).item()) if has_emv else 0.0

        mode = 1 if has_emv and prob > self.threshold else 0
        return mode, prob


class ModeController:
    """
    Runtime mode controller managing switch between RL and LLM branches.

    Maintains the context-aware classifier and applies the mode switching
    criterion from the paper (Equation in Section "Mode-switching criterion").
    """

    def __init__(self, num_features=32, threshold=None):
        self.context_module = ContextAwarenessModule(num_features, threshold)

    @torch.no_grad()
    def get_mode(self, numerical_features):
        """Get current mode from numerical features."""
        _, mode_flag = self.context_module(numerical_features)
        return int(mode_flag[0].item())

    @torch.no_grad()
    def detect_emergency_from_graph(self, graph_dict):
        """Detect emergency directly from graph dict (no VLM needed)."""
        return self.context_module.detect_emergency_from_graph(graph_dict)
