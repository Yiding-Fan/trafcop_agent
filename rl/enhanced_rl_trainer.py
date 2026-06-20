"""
Enhanced RL Trainer for TrafCopAgent.

Extends RMTC's R_MAPPO with:
1. Role-aware intrinsic reward (Section "Enhanced Reinforcement Learning Module")
2. EMV Position Mutual Information (Pos-MI) loss
3. EMV Trajectory Mutual Information (Traj-MI) loss
4. Role Consistency Loss

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from trafcop_agent.config import ENHANCED_RL


class RoleAwareIntrinsicReward:
    """
    Computes role-aware intrinsic reward for agents near EMVs.

    """

    def __init__(self, embedding_dim=64, intrinsic_coef=None):
        self.embedding_dim = embedding_dim
        self.intrinsic_coef = intrinsic_coef or ENHANCED_RL["intrinsic_coef"]

    @torch.no_grad()
    def compute(self, role_embeddings, emv_role_embeddings, extrinsic_rewards):
        """
        Args:
            role_embeddings: (B, N_agent, D) agent role embeddings h_n^(t)
            emv_role_embeddings: (B, N_emv, D) EMV role embeddings h_{emv,i}^{(t)}
            extrinsic_rewards: (B, N_agent) extrinsic rewards r_{n,extr}

        Returns:
            total_rewards: (B, N_agent) combined rewards with intrinsic component
        """
        B, N_agent, D = role_embeddings.shape
        N_emv = emv_role_embeddings.shape[1]

        # Cosine similarity between agents and EMVs
        sim = F.cosine_similarity(
            role_embeddings.unsqueeze(2).expand(-1, -1, N_emv, -1),
            emv_role_embeddings.unsqueeze(1).expand(-1, N_agent, -1, -1),
            dim=-1,  # (B, N_agent, N_emv)
        )

        # Average similarity across EMVs as importance weights
        importance = sim.mean(dim=2)  # (B, N_agent)

        # Scale by extrinsic reward magnitude
        reward_scale = torch.abs(extrinsic_rewards).mean(dim=1, keepdim=True) + 1e-8
        intrinsic_reward = importance * reward_scale

        total_rewards = extrinsic_rewards + self.intrinsic_coef * intrinsic_reward
        return total_rewards


class MutualInformationLoss:
    """
    Implements Pos-MI and Traj-MI losses for role encoder training.

    Uses MINE-style variational lower bound of mutual information.
    """

    def __init__(self, temperature=0.1):
        self.temperature = temperature or ENHANCED_RL["mi_temperature"]

    def pos_mi_lower_bound(self, agent_embeddings, emv_pos_features):
        """
        EMV Position Mutual Information (Pos-MI).

        Maximizes variational lower bound of I(agent_role; emv_position).
        Equation from paper Section "EMV Position Mutual Information".
        """
        B, N_agent, D = agent_embeddings.shape
        # Compute similarity between agent roles and EMV positions
        sim = torch.einsum('bid,bjd->bij', agent_embeddings, emv_pos_features) / self.temperature

        # MINE-style lower bound: (1/N) * [T(h_i, h_emv) - log(sum_j exp(T(h_j, h_emv)))]
        pos_mi = sim.diagonal(dim1=1, dim2=2).mean() - torch.log(
            sim.exp().sum(dim=2).mean() + 1e-8
        )
        return pos_mi

    def traj_mi_lower_bound(self, agent_embeddings, emv_traj_features):
        """
        EMV Trajectory Mutual Information (Traj-MI).

        Maximizes variational lower bound of I(agent_role; emv_trajectory).
        """
        B, N_agent, D = agent_embeddings.shape
        sim = torch.einsum('bid,bjd->bij', agent_embeddings, emv_traj_features) / self.temperature

        # Positive pairs on diagonal
        pos_sim = sim.diagonal(dim1=1, dim2=2)
        # All similarities as negative samples
        all_sim = sim.view(B * N_agent, -1)
        traj_mi = pos_sim - torch.log(all_sim.exp().sum(dim=1))

        return traj_mi.mean()

    def role_consistency_loss(self, role_embeddings_t, role_embeddings_t1):
        """
        Role Consistency Loss.

        Enforces temporal smoothness: consecutive embeddings should be similar,
        while unrelated pairs are pushed apart (InfoNCE form).
        """
        B, N_agent, D = role_embeddings_t.shape
        # Similarity matrix between t and t+1
        sim = torch.einsum(
            'bid,bjd->bij',
            F.normalize(role_embeddings_t, dim=-1),
            F.normalize(role_embeddings_t1, dim=-1),
        ) / self.temperature

        # Positive pairs on diagonal
        pos_sim = sim.diagonal(dim1=1, dim2=2)
        # InfoNCE: pull positive pairs, push negative apart
        loss = -pos_sim + torch.log(sim.exp().sum(dim=2))
        return loss.mean()


class EnhancedRLTrainer:
    """
    Main trainer combining RMTC's MAPPO with TrafCopAgent enhancements.

    Integrates role-aware intrinsic rewards and mutual information losses
    for emergency traffic control. This extends the base R_MAPPO class from
    RMTC without modifying its source code.
    """

    def __init__(self, args, device=None):
        self.args = args
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Role-aware intrinsic reward module
        emb_dim = ENHANCED_RL["embedding_dim"]
        self.intrinsic_reward = RoleAwareIntrinsicReward(
            embedding_dim=emb_dim,
            intrinsic_coef=ENHANCED_RL["intrinsic_coef"],
        )

        # Mutual information loss modules
        self.mi_loss_module = MutualInformationLoss(
            temperature=ENHANCED_RL["mi_temperature"],
        )

        # Extrinsic reward weights from config
        self.queue_weight = ENHANCED_RL["queue_weight"]
        self.waiting_weight = ENHANCED_RL["waiting_weight"]
        self.pressure_var_weight = ENHANCED_RL["pressure_var_weight"]
        self.emv_proximity_weight = ENHANCED_RL["emv_proximity_weight"]

    def compute_extrinsic_reward(self, queue_lengths, waiting_times, pressure_vals,
                                  has_emv_nearby):
        """
        Compute extrinsic reward from traffic features.
        """
        B = queue_lengths.shape[0]
        reward = (
            self.queue_weight * queue_lengths.float()
            + self.waiting_weight * waiting_times.float()
            + self.pressure_var_weight * pressure_vals.float()
        )
        if has_emv_nearby is not None:
            reward += self.emv_proximity_weight * has_emv_nearby.float()
        return -reward  # negative because lower queue/wait = better

    def compute_total_loss(self, policy_loss, value_loss, critic_loss,
                           role_embeddings_t, role_embeddings_t1,
                           emv_pos_features, emv_traj_features):
        """
        Compute combined training objective
        """
        pos_mi = self.mi_loss_module.pos_mi_lower_bound(
            role_embeddings_t, emv_pos_features
        )
        traj_mi = self.mi_loss_module.traj_mi_lower_bound(
            role_embeddings_t, emv_traj_features
        )
        role_cons = self.mi_loss_module.role_consistency_loss(
            role_embeddings_t, role_embeddings_t1
        )

        # L_role = -(Pos-MI + Traj-MI) + L_cons (maximize MI, minimize consistency loss)
        role_loss = (
            -ENHANCED_RL["pos_mi_weight"] * pos_mi
            -ENHANCED_RL["traj_mi_weight"] * traj_mi
            + ENHANCED_RL["cons_loss_weight"] * role_cons
        )

        total_loss = (-policy_loss) + critic_loss + role_loss
        metrics = {
            "pos_mi": pos_mi.item(),
            "traj_mi": traj_mi.item(),
            "role_cons": role_cons.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": total_loss.item(),
        }
        return total_loss, metrics

    def compute_role_embeddings(self, graph_dict):
        """
        Extract role embeddings from heterogeneous graph.
        """
        h_typedict = graph_dict.get('x_dict', None)
        if h_typedict is None:
            # Fallback: use graph node features directly
            x_dict = {k: v.float() for k, v in graph_dict.items()
                      if isinstance(v, torch.Tensor)}
            h = {k: F.normalize(v, dim=-1) for k, v in x_dict.items()}
        else:
            h = {k: F.normalize(v, dim=-1) for k, v in h_typedict.items()}
        return h
