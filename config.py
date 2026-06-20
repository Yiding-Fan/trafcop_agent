"""
Configuration for TrafCopAgent.

Defines scenarios, hyperparameters, and mode-switching thresholds matching the paper's
experimental setup (Table 1: Grid 4x4, Avenue 4x4, Cologne 8, FengLin).
"""

# Scenario definitions matching paper Table 1
SCENARIOS = {
    "grid4x4": {
        "sumocfg": "sumo_files_marl/scenarios/resco_envs/cologne8_test/cologne8.sumocfg",
        # Override with grid4x4 path in practice
        "num_phases": 4,
        "n_edges": 12,
    },
    "avenue4x4": {
        "sumocfg": "sumo_files_marl/scenarios/resco_envs/arterial4x4_trip/arterial4x4.sumocfg",
        "num_phases": 4,
        "n_edges": 16,
    },
    "cologne8": {
        "sumocfg": "sumo_files_marl/scenarios/resco_envs/cologne8/cologne8.sumocfg",
        "num_phases": 8,
        "n_edges": 24,
    },
    "fenglin": {
        "sumocfg": "sumo_files_marl/scenarios/sumo_fenglin_base_road_trip/base.sumocfg",
        "num_phases": 6,
        "n_edges": 20,
    },
}

# Context awareness module hyperparameters (paper Section: Sensitivity to Activation Threshold)
CONTEXT_AWARE = {
    "threshold": 0.7,       # tau_m from Equation in paper - default threshold for mode switching
    "hidden_dim": 64,        # MLP hidden dimension for classifier
    "text_feat_dim": 768,    # CLIP/VLM text feature dimension
    "num_classes": 2,        # routine (0) vs emergency (1)
}

# Enhanced RL module hyperparameters (paper Section: Reward Function & Loss Design)
ENHANCED_RL = {
    # Extrinsic reward weights (Equation for r_t in paper)
    "queue_weight": 1.0,       # alpha - queue length term
    "waiting_weight": 1.0,     # beta - waiting time term
    "pressure_var_weight": 0.5, # gamma - pressure variance term
    "emv_proximity_weight": 2.0, # delta - EMV proximity indicator

    # Intrinsic reward
    "intrinsic_coef": 0.01,    # lambda - intrinsic reward scaling factor
    "embedding_dim": 64,       # role embedding dimension (h_i^(t))

    # Role encoder losses (Section: Ablation of Individual Loss Terms)
    "pos_mi_weight": 1.0,     # coefficient for Pos-MI loss
    "traj_mi_weight": 1.0,    # coefficient for Traj-MI loss
    "cons_loss_weight": 0.5,  # coefficient for Role Consistency loss
    "mi_temperature": 0.1,    # tau in mutual information estimation
}

# LLM agent configuration (paper Section: Multi-Agent Language Reasoning Module)
LLM_AGENTS = {
    "max_check_attempts": 3,  # N_check from paper - bounded verification loop
    "response_format": "json",
}

# Inference pipeline settings
INFERENCE = {
    "fallback_to_rl": True,    # fallback to RL if LLM fails (paper Algorithm 2)
    "min_green_duration": 5,   # minimum green time in seconds
    "amber_duration": 3,       # amber/all-red interval in seconds
}
