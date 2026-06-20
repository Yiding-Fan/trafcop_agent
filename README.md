# TrafCopAgent

TrafCopAgent implementation based on RMTC.

## Setup

```bash
pip install -r /requirements.txt
```

## Usage

```bash
# Train (4 scenarios from the paper)
python train_trafcop.py --scenario grid4x4 --seed 1
python train_trafcop.py --scenario avenue4x4 --seed 1
python train_trafcop.py --scenario cologne8 --seed 1
python train_trafcop.py --scenario fenglin --seed 1

# Main experiment (all scenarios x 10 seeds)
python run_main_experiments.py
```

## Structure

| File | Description |
|---|---|
| `config.py` | Scenario configs, hyperparameters |
| `context/context_awareness.py` | Emergency detection module |
| `rl/enhanced_rl_trainer.py` | Role-aware intrinsic reward + Pos-MI/Traj-MI/RoleCons losses |
| `llm/llm_agents.py` | Three LLM agents (Scene, Network, Controller) |
| `inference/pipeline.py` | Mode switching + RL/LLM fallback pipeline |
| `train_trafcop.py` | Main training script |
| `evaluate_trafcop.py` | Evaluation script |
