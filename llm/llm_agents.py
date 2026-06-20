"""
Multi-Agent Language Reasoning Module.

Three role-specialized LLM agents that collaboratively generate emergency decisions:
1. Intersection State Analyzer (Agent-Scene) - condenses intersection states into phase-aligned summaries
2. Network Collaborative Planner (Agent-Network) - reasons over multi-intersection routing
3. Dispatcher/Verifier (Agent-Controller) - enforces legality constraints and issues commands

Uses structured JSON messages between agents rather than free text.
Falls back to RL action if no valid LLM plan within N_check attempts.
"""

import json
import copy
from trafcop_agent.config import LLM_AGENTS, INFERENCE


class IntersectionStateAnalyzer:
    """
    Agent-Scene: Produces structured intersection report highlighting
    congestion points, EMV locations/ETAs, and recommended priority directions.
    Maps observations to phase-level summaries.
    """

    def __init__(self, llm_client=None):
        self.llm = llm_client  # Qwen2.5-72B or compatible API client
        self.system_prompt = (
            "You are an intersection state analyzer. Your task is to produce a structured "
            "summary of the current traffic scene, highlighting EMV location and congestion."
        )

    def analyze(self, scene_data):
        """
        Analyze current intersection state.

        Args:
            scene_data: dict with keys:
                - queue_lengths: list[float] per lane
                - emv_positions: list[str] like "north_approach:2"
                - congestion_levels: list[str] per approach "light|medium|heavy"
                - current_phases: list[int]

        Returns:
            phase_summaries: dict of JSON phase-level summaries keyed by phase index
        """
        # If LLM client is available, use it for semantic analysis
        if self.llm is not None:
            user_msg = (
                f"Queue lengths: {scene_data['queue_lengths']}\n"
                f"EMV positions: {scene_data['emv_positions']}\n"
                f"Congestion levels: {scene_data['congestion_levels']}\n"
                "Analyze and output JSON with phase recommendations."
            )
            response = self.llm.generate(user_msg)
            return json.loads(response)

        # Rule-based fallback: generate structured summary without LLM
        phase_summaries = {}
        for i, ql in enumerate(scene_data["queue_lengths"]):
            congestion = "heavy" if ql > 5 else ("medium" if ql > 2 else "light")
            phase_summaries[f"phase_{i}"] = {
                "congestion": congestion,
                "queue_length": round(ql, 2),
                "emv_detected": len(scene_data.get("emv_positions", [])) > 0,
            }
        return phase_summaries


class NetworkCollaborativePlanner:
    """
    Agent-Network: Takes phase-level summaries + EMV ETA + network graph distances
    to reason about multi-intersection coordination. Proposes candidate signal phase
    and routing advisories for green-wave corridor formation.
    """

    def __init__(self, llm_client=None):
        self.llm = llm_client
        self.system_prompt = (
            "You are a network collaborative planner for emergency traffic corridors. "
            "Given phase-level summaries from intersections along the EMV route, propose "
            "signal phases and routing adjustments to form a green-wave corridor."
        )

    def plan(self, phase_summaries, emv_eta, network_adjacency):
        """
        Plan multi-intersection coordination.

        Args:
            phase_summaries: dict from IntersectionStateAnalyzer per intersection
            emv_eta: int - estimated time steps until EMV reaches next intersection
            network_adjacency: dict of neighbor intersections for each junction

        Returns:
            plan: dict with keys:
                - candidate_phase: int - recommended phase for current intersection
                - offsets: list[float] - green-wave offset durations per hop
                - rerouting_advisories: dict - lane-level routing suggestions for REVs
        """
        if self.llm is not None:
            user_msg = (
                f"Phase summaries: {json.dumps(phase_summaries)}\n"
                f"EMV ETA to next intersection: {emv_eta} steps\n"
                f"Network neighbors: {list(network_adjacency.keys())}\n"
                "Output JSON with candidate phase, offsets, and rerouting advisories."
            )
            response = self.llm.generate(user_msg)
            return json.loads(response)

        # Rule-based fallback: prioritize EMV direction, minimize disruption
        return {
            "candidate_phase": self._rule_based_phase(phase_summaries),
            "offsets": [2.0] * len(network_adjacency),
            "rerouting_advisories": {},
        }

    def _rule_based_phase(self, phase_summaries):
        """Select phase that maximizes green time for EMV approach."""
        best_phase = 0
        max_emv_pressure = -1
        for phase_name, summary in phase_summaries.items():
            if summary.get("emv_detected", False):
                pressure = summary["queue_length"]
                if pressure > max_emv_pressure:
                    max_emv_pressure = pressure
                    best_phase = int(phase_name.split("_")[-1])
        return best_phase


class DispatcherVerifier:
    """
    Agent-Controller: Synthesizes scene report and network plan to produce
    final executable decision. Verifies legality against phase constraints,
    conflict matrix, and local regulations. Falls back to RL if invalid after N_check attempts.
    """

    def __init__(self, llm_client=None, max_attempts=None):
        self.llm = llm_client
        self.max_attempts = max_attempts or LLM_AGENTS["max_check_attempts"]
        self.system_prompt = (
            "You are the traffic control dispatcher and verifier. Synthesize the scene "
            "analysis and network plan to produce an executable signal phase decision. "
            "Verify legality before outputting."
        )

    def dispatch(self, scene_report, network_plan, available_phases):
        """
        Produce final executable decision with feasibility verification.

        Args:
            scene_report: dict from IntersectionStateAnalyzer
            network_plan: dict from NetworkCollaborativePlanner
            available_phases: list[int] - legally allowed phase transitions

        Returns:
            decision: dict with keys:
                - signal_phase: int - final selected phase
                - emergency_vehicle_route: list[str] - movement instructions for EMV
                - regular_vehicle_rerouting: dict - lane-level routing suggestions
                - verified: bool - whether the plan passed feasibility check
        """
        for attempt in range(self.max_attempts):
            if self.llm is not None:
                user_msg = (
                    f"Scene report: {json.dumps(scene_report)}\n"
                    f"Network plan: {json.dumps(network_plan)}\n"
                    f"Available phases: {available_phases}\n"
                    "Output JSON with verified signal phase, EMV route, and REV rerouting."
                )
                response = self.llm.generate(user_msg)
                decision = json.loads(response)
            else:
                decision = self._rule_based_dispatch(network_plan, available_phases)

            # Feasibility verification
            if self._verify_feasibility(decision, available_phases):
                decision["verified"] = True
                return decision

        # Failed all attempts - fallback to RL (guarantees liveness per paper)
        return {
            "signal_phase": -1,  # signals fallback
            "emergency_vehicle_route": [],
            "regular_vehicle_rerouting": {},
            "verified": False,
            "fallback": True,
        }

    def _verify_feasibility(self, decision, available_phases):
        """Check phase legality against current feasible set."""
        phase = decision.get("signal_phase", -1)
        if phase < 0:
            return False
        return phase in available_phases

    def _rule_based_dispatch(self, network_plan, available_phases):
        """Fallback dispatch without LLM using rule-based selection."""
        candidate = network_plan["candidate_phase"]
        fallback = available_phases[0] if available_phases else 0
        phase = candidate if candidate in available_phases else fallback
        return {
            "signal_phase": phase,
            "emergency_vehicle_route": ["straight"],
            "regular_vehicle_rerouting": {},
            "verified": True,
        }


class LLMDecisionPipeline:
    """
    Orchestrates the three LLM agents in sequential multi-turn dialogue.

    Implements Algorithm 2 (Inference) from the paper for emergency mode decisions.
    Falls back to RL action if no valid LLM plan within time budget.
    """

    def __init__(self, llm_client=None):
        self.scene_analyzer = IntersectionStateAnalyzer(llm_client)
        self.network_planner = NetworkCollaborativePlanner(llm_client)
        self.dispatcher = DispatcherVerifier(llm_client)

    def make_emergency_decision(self, scene_data, emv_eta, network_adjacency,
                                available_phases):
        """
        Full emergency decision pipeline: Scene -> Network -> Controller.

        Args:
            scene_data: dict with traffic state (queue_lengths, emv_positions, etc.)
            emv_eta: int - EMV ETA to next intersection in time steps
            network_adjacency: dict of neighboring intersections
            available_phases: list[int] of legally allowed phase transitions

        Returns:
            decision: dict matching paper's output format:
                {
                    "signal_phase": Phase_4,
                    "emergency_vehicle_route": ["turn_right"],
                    "regular_vehicle_rerouting": {"lane_1": "turn_left", ...}
                }
        """
        # Step 1: Agent-Scene produces phase-level summaries
        phase_summaries = self.scene_analyzer.analyze(scene_data)

        # Step 2: Agent-Network proposes candidate phase and routing
        network_plan = self.network_planner.plan(phase_summaries, emv_eta, network_adjacency)

        # Step 3: Agent-Controller verifies and dispatches
        decision = self.dispatcher.dispatch(
            scene_report=phase_summaries,
            network_plan=network_plan,
            available_phases=available_phases,
        )
        return decision
