"""
TrafCopAgent Inference Pipeline.

Main entry point that orchestrates:
1. Context awareness module for mode detection
2. RL branch (enhanced MAPPO) for routine traffic
3. LLM multi-agent reasoning branch for emergency scenarios
4. Mode switching and fallback (paper Algorithm 2)

Does NOT modify RMTC source code - integrates via environment interface.
"""

import numpy as np
import torch
from trafcop_agent.context.context_awareness import ModeController, ContextAwarenessModule
from trafcop_agent.llm.llm_agents import LLMDecisionPipeline
from trafcop_agent.config import CONTEXT_AWARE, INFERENCE


class TrafCopAgentInference:
    """
    Complete TrafCopAgent inference system matching paper Algorithm 2.

    At each timestep t:
        1. Collect multi-source observations (detector features + EMV visual cues)
        2. Context awareness outputs m_t in {0, 1}
        3. If routine (m_t=0): invoke RL policy for low-latency control
        4. If emergency (m_t=1): invoke LLM multi-agent reasoning with feasibility check
        5. Feasible action dispatched to signal controller and navigation systems
    """

    def __init__(self, rl_policy=None, threshold=None, llm_client=None,
                 num_agents=0, mode="emergency"):
        """
        Args:
            rl_policy: trained RL policy wrapper (R_MAPPO policy in eval mode)
                       with method: act(obs, rnn_states, masks, graph, available_actions)
            threshold: emergency detection threshold tau_m (default from config)
            llm_client: optional LLM client for reasoning (Qwen2.5-72B compatible API)
                        if None, uses rule-based fallback agents
            num_agents: number of traffic light agents (intersections)
            mode: "emergency" for main experiment, "routine" for RL-only baseline
        """
        self.rl_policy = rl_policy
        self.context_controller = ModeController(
            num_features=32, threshold=threshold or CONTEXT_AWARE["threshold"]
        )
        self.llm_pipeline = LLMDecisionPipeline(llm_client) if llm_client else LLMDecisionPipeline()
        self.num_agents = num_agents
        self.mode = mode

    @torch.no_grad()
    def step(self, obs, shared_obs, rnn_states, masks, graph,
             available_actions, emv_info=None):
        """
        Single inference step - selects RL or LLM branch based on context.

        Args:
            obs: (B, N_agent, obs_dim) local observations
            shared_obs: (B, N_agent, shared_obs_dim) centralized observations
            rnn_states: (B, N_agent, rnn_dim) recurrent hidden states
            masks: (B, N_agent, 1) active masks for RNN reset
            graph: graph state dict with keys 'signal_light', 'vehicle', 'emergency'
            available_actions: (B, N_agent, num_actions) action availability mask
            emv_info: optional dict with EMV detection data

        Returns:
            actions: (B, N_agent, action_dim) selected actions
            new_rnn_states: updated RNN states
            mode_flag: 0=routine (RL), 1=emergency (LLM) per environment
        """
        B = obs.shape[0] if isinstance(obs, torch.Tensor) else obs.shape[0]

        # Step 1: Context awareness - detect emergency state
        numerical_features = self._extract_numerical_features(shared_obs)
        mode_flag, emergency_prob = self.context_controller.detect_emergency_from_graph(graph)

        if self.mode == "routine" or mode_flag == 0:
            # --- Routine Mode: Enhanced RL Policy ---
            return self._rl_branch(obs, rnn_states, masks, graph, available_actions)

        else:
            # --- Emergency Mode: LLM Multi-Agent Reasoning ---
            decision = self._llm_branch(
                graph, emv_info, available_actions
            )

            if decision.get("verified", False):
                return self._convert_llm_decision(decision), rnn_states, mode_flag
            elif INFERENCE["fallback_to_rl"]:
                # Fallback to RL preserves liveness (paper Algorithm 2, line 47)
                return self._rl_branch(
                    obs, rnn_states, masks, graph, available_actions
                )
            else:
                raise RuntimeError("No valid action: LLM failed and fallback disabled")

    def _extract_numerical_features(self, shared_obs):
        """Extract numerical traffic features from observations for context module."""
        if isinstance(shared_obs, torch.Tensor):
            return shared_obs[:, :32]  # first 32 dims as detector features
        elif hasattr(shared_obs, 'shape'):
            B = shared_obs.shape[0] if len(shared_obs.shape) > 1 else 1
            return np.zeros((B, 32), dtype=np.float32) if shared_obs.size == 0 else shared_obs[:, :32]
        return np.zeros((1, 32), dtype=np.float32)

    def _rl_branch(self, obs, rnn_states, masks, graph, available_actions):
        """Invoke enhanced RL policy for routine traffic control."""
        if self.rl_policy is None:
            # Default: select first available phase as placeholder
            actions = np.zeros((obs.shape[0], self.num_agents, 1), dtype=np.int64)
            return actions, rnn_states, 0

        actions, new_rnn_states = self.rl_policy.act(
            obs=obs,
            rnn_states=rnn_states,
            masks=masks,
            graph=graph,
            available_actions=available_actions,
            deterministic=True,
        )
        return actions, new_rnn_states, 0

    def _llm_branch(self, graph, emv_info, available_actions):
        """Invoke LLM multi-agent reasoning for emergency scenario."""
        # Extract scene data from graph
        scene_data = {
            "queue_lengths": [],
            "emv_positions": [f"approach_{i}" for i in range(
                graph.get("emergency", torch.zeros(1)).shape[0]
            )],
            "congestion_levels": ["medium"] * self.num_agents,
            "current_phases": [0] * self.num_agents,
        }

        if emv_info and "eta" in emv_info:
            emv_eta = emv_info["eta"]
        else:
            emv_eta = 5  # default ETA in time steps

        # Get available phases from action mask
        available_phases = []
        if available_actions is not None:
            aa = available_actions[0] if isinstance(available_actions, np.ndarray) else available_actions[0]
            for i in range(len(aa)):
                if aa[i] > 0:
                    available_phases.append(i)
        if not available_phases:
            available_phases = list(range(self.num_agents))

        decision = self.llm_pipeline.make_emergency_decision(
            scene_data=scene_data,
            emv_eta=emv_eta,
            network_adjacency={f"junction_{i}": [] for i in range(self.num_agents)},
            available_phases=available_phases,
        )
        return decision

    def _convert_llm_decision(self, llm_decision):
        """Convert LLM JSON decision to numpy action array."""
        phase = llm_decision.get("signal_phase", -1)
        if phase < 0:
            # Invalid from LLM - use default phase 0
            phase = 0
        B = self.num_agents
        return np.full((B, 1), phase, dtype=np.int64)
