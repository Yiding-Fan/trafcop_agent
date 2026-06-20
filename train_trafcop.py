#!/usr/bin/env python
"""
Main training script for TrafCopAgent.

Trains the enhanced RL policy with role-aware intrinsic rewards and mutual
information losses (Pos-MI, Traj-MI, RoleCons) on all four scenarios from
the paper: Grid 4x4, Avenue 4x4, Cologne 8, FengLin.

Does NOT modify RMTC source code - builds on top of RMTC's environment and
runner infrastructure. Run from the project root directory:

    python trafcop_agent/train_trafcop.py --scenario grid4x4
    python trafcop_agent/train_trafcop.py --scenario avenue4x4
    python trafcop_agent/train_trafcop.py --scenario cologne8
    python trafcop_agent/train_trafcop.py --scenario fenglin

For multiple seeds (paper Table 1 format, 10 independent runs):
    for seed in $(seq 1 10); do
        python trafcop_agent/train_trafcop.py --scenario grid4x4 --seed $seed
    done
"""

import sys
import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# Add RMTC paths (without modifying RMTC source)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'RMTC'))

from onpolicy.config import get_config as rmtp_config_parser
from onpolicy.envs.sumo_files_marl.config import config as sumo_config_env
from onpolicy.envs.sumo_files_marl.SUMO_env import SUMOEnv
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv

from trafcop_agent.rl.enhanced_rl_trainer import EnhancedRLTrainer
from trafcop_agent.config import SCENARIOS, ENHANCED_RL


def make_train_env(all_args):
    """Create vectorized training environments (same interface as RMTC)."""
    def get_env_fn(rank):
        def init_env():
            env = SUMOEnv(all_args, rank)
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    """Create vectorized evaluation environments."""
    def get_env_fn(rank):
        def init_env():
            env = SUMOEnv(all_args, rank)
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args():
    parser = argparse.ArgumentParser(description="TrafCopAgent Training")
    parser.add_argument("--scenario", type=str, default="grid4x4",
                        choices=["grid4x4", "avenue4x4", "cologne8", "fenglin"],
                        help="Traffic scenario to train on")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--experiment_name", type=str, default="trafcop_main",
                        help="Experiment name for logging")

    # Training hyperparameters (matching paper Section "Implementation details")
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--critic_lr", type=float, default=5e-5)
    parser.add_argument("--ppo_epoch", type=int, default=10)
    parser.add_argument("--num_mini_batch", type=int, default=4)
    parser.add_argument("--num_env_steps", type=int, default=9_000_000)
    parser.add_argument("--n_training_threads", type=int, default=8)
    parser.add_argument("--n_rollout_threads", type=int, default=2)

    # RMTC compatibility flags
    parser.add_argument("--gain", type=float, default=0.01)
    parser.add_argument("--use_orthogonal", action='store_false', default=True)
    parser.add_argument("--clip_param", type=float, default=0.2)
    parser.add_argument("--value_loss_coef", type=float, default=1.0)
    parser.add_argument("--entropy_coef", type=float, default=0.01)

    # Model saving and logging
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--use_eval", action='store_true', default=False)

    # Environment
    parser.add_argument("--state_key", nargs='+', default=[
        'current_phase', 'car_num', 'queue_length', 'occupancy',
        'flow', 'stop_car_num', 'pressure'
    ])
    parser.add_argument("--sumocfg_files", type=str, default="",
                        help="Override sumocfg path (auto-selected from scenario)")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Resolve sumocfg path from scenario config
    scenario_cfg = SCENARIOS[args.scenario]
    if not args.sumocfg_files:
        args.sumocfg_files = scenario_cfg["sumocfg"]

    # Set up RMTC config parameters
    all_args = argparse.Namespace()
    all_args.algorithm_name = "rmappo"  # use recurrent MAPPO
    all_args.seed = args.seed
    all_args.cuda = torch.cuda.is_available()
    all_args.n_training_threads = args.n_training_threads
    all_args.n_rollout_threads = args.n_rollout_threads
    all_args.n_eval_rollout_threads = 1
    all_args.num_env_steps = args.num_env_steps
    all_args.episode_length = sumo_config_env['episode']['rollout_length']
    all_args.num_actions = sumo_config_env['environment']['num_actions']
    all_args.share_policy = True
    all_args.use_recurrent_policy = True
    all_args.use_naive_recurrent_policy = False
    all_args.use_centralized_V = True

    # Network hyperparameters
    all_args.hidden_size = args.hidden_size
    all_args.layer_N = 1
    all_args.use_ReLU = True
    all_args.use_popart = False
    all_args.use_valuenorm = True
    all_args.use_feature_normalization = True
    all_args.use_orthogonal = args.use_orthogonal
    all_args.gain = args.gain

    # PPO hyperparameters
    all_args.clip_param = args.clip_param
    all_args.ppo_epoch = args.ppo_epoch
    all_args.num_mini_batch = args.num_mini_batch
    all_args.value_loss_coef = args.value_loss_coef
    all_args.entropy_coef = args.entropy_coef
    all_args.max_grad_norm = 10.0
    all_args.use_huber_loss = True
    all_args.use_gae = True
    all_args.gamma = 0.99
    all_args.gae_lambda = 0.95

    # Optimizer
    all_args.lr = args.lr
    all_args.critic_lr = args.critic_lr
    all_args.opti_eps = 1e-5
    all_args.weight_decay = 0

    # RL/GNN optimizer settings (matching RMTC config.py)
    all_args.lr_gnn = args.lr  # same as actor lr per paper Section "Implementation details"
    all_args.opti_eps_gnn = 1e-5
    all_args.weight_decay_gnn = 0
    all_args.num_ve_actions = 3
    all_args.recurrent_N = 1
    all_args.data_chunk_length = 10

    # Scenario
    all_args.env_name = "SUMO"
    all_args.scenario_name = args.scenario
    all_args.experiment_name = args.experiment_name
    all_args.state_key = args.state_key
    all_args.sumocfg_files = args.sumocfg_files
    all_args.port_start = 14444
    all_args.num_ve_agents = ENHANCED_RL["embedding_dim"]  # placeholder

    # Evaluation
    all_args.use_eval = args.use_eval
    all_args.eval_interval = args.eval_interval
    all_args.save_interval = args.save_interval
    all_args.log_interval = args.log_interval

    # Device setup
    device = torch.device("cuda" if all_args.cuda and torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Create run directory (matching RMTC convention)
    run_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "results" / args.scenario / args.experiment_name / f"seed_{args.seed}"
    if not run_dir.exists():
        run_dir.mkdir(parents=True, exist_ok=True)

    # Save arguments
    with open(run_dir / "args.txt", "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    # Create environments
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None

    all_args.num_agents = len(envs.action_space)
    all_args.num_vehicle_agents = getattr(sumo_config_env['environment'], 'num_vehicle_agents', 1472)
    all_args.num_emv_agents = getattr(sumo_config_env['environment'], 'num_emv_agents', 24)

    # Initialize enhanced RL trainer (extends base R_MAPPO with role-aware components)
    from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO
    from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy

    policy = R_MAPPOPolicy(
        args=all_args,
        obs_space=envs.observation_space[0],
        cent_obs_space=envs.share_observation_space[0],
        act_space=envs.action_space[0],
        device=device,
        type='vehicle',
    )

    trainer = R_MAPPO(
        args=all_args,
        policy=policy,
        num_agents=all_args.num_agents,
        device=device,
    )

    enhanced_rl = EnhancedRLTrainer(all_args, device)

    # Training loop (adapted from RMTC's SUMORunner)
    start_time = time.time()
    episodes = int(all_args.num_env_steps) // all_args.episode_length // all_args.n_rollout_threads

    train_metrics_file = run_dir / "train_metrics.txt"

    print(f"Training TrafCopAgent on scenario={args.scenario}, seed={args.seed}")
    print(f"  Episodes: {episodes}, Steps/episode: {all_args.episode_length}")
    print(f"  Device: {device}")
    print(f"  Run dir: {run_dir}")

    with open(train_metrics_file, "w") as mf:
        mf.write("episode,total_steps,avg_reward,t_emv,t_rev\n")

    for episode in range(episodes):
        # Learning rate decay (paper Section "Training and Inference")
        if episode % 1000 == 0 and episode > 0:
            policy.lr_decay(episode, episodes)
            policy.lr_decay_gnn(episode, episodes)

        # Reset buffer each episode
        trainer.policy.actor_optimizer.zero_grad()
        trainer.policy.critic_optimizer.zero_grad()

        # Training loop adapted from SUMORunner.run()
        episode_rewards = []
        t_emv_times = []
        t_rev_times = []

        for step in range(all_args.episode_length):
            # Collect actions and environment step
            values_tl, actions_tl, action_log_probs_tl, rnn_states_tl, rnn_states_critic_tl, actions_env_tl \
                = collect_step(envs, trainer, all_args, device)

            # Step environment
            obs_tl, rewards_tl, dones_tl, returns_tl, infos_tl \
                = envs.step(actions_env_tl.astype(np.int64)[:, :, 0])

            episode_rewards.append(np.sum(rewards_tl))

            # Check for emergency vehicle data in infos
            if 'emv_travel_time' in infos_tl:
                t_emv_times.append(infos_tl['emv_travel_time'])
            if 'rev_travel_time' in infos_tl:
                t_rev_times.append(infos_tl['rev_travel_time'])

            # Insert into buffer (adapted from SUMORunner.insert methods)
            insert_step(
                trainer, envs, all_args, device,
                obs_tl, rewards_tl, dones_tl, infos_tl,
                values_tl, actions_tl, action_log_probs_tl,
                rnn_states_tl, rnn_states_critic_tl, step,
            )

            if dones_tl.all():
                break

        # Compute returns and train at episode end
        if episode % all_args.save_interval == 0 or episode == episodes - 1:
            compute_returns(trainer, envs, all_args, device)
            train_infos = train_step(
                trainer, enhanced_rl, policy, all_args, device
            )

        # Logging
        avg_reward = np.mean(episode_rewards) if episode_rewards else 0
        t_emv_avg = np.mean(t_emv_times) if t_emv_times else 0
        t_rev_avg = np.mean(t_rev_times) if t_rev_times else 0

        if episode % args.log_interval == 0 or episode == episodes - 1:
            elapsed = time.time() - start_time
            print(f"Episode {episode}/{episodes} | "
                  f"Avg Reward: {avg_reward:.4f} | "
                  f"T_EMV: {t_emv_avg:.2f} | "
                  f"T_REV: {t_rev_avg:.2f} | "
                  f"Elapsed: {elapsed:.0f}s")

            with open(train_metrics_file, "a") as mf:
                mf.write(f"{episode},{step},{avg_reward:.4f},{t_emv_avg:.2f},{t_rev_avg:.2f}\n")

        # Model saving
        if episode % args.save_interval == 0 or episode == episodes - 1:
            save_path = run_dir / f"model_ep{episode}.pt"
            torch.save({
                "actor_state": policy.actor.state_dict(),
                "critic_state": policy.critic.state_dict(),
                "gnn_state": policy.gnn.state_dict() if hasattr(policy, 'gnn') else {},
                "actor_optim": policy.actor_optimizer.state_dict(),
                "critic_optim": policy.critic_optimizer.state_dict(),
                "gnn_optim": policy.gnn_optimizer.state_dict() if hasattr(policy, 'gnn_optimizer') else {},
            }, save_path)

        # Evaluation
        if args.use_eval and episode % args.eval_interval == 0:
            eval_results = evaluate(
                eval_envs, trainer, all_args, device, run_dir, episode
            )
            t_emv_avg = np.mean(eval_results.get("t_emv_list", [0]))
            t_rev_avg = np.mean(eval_results.get("t_rev_list", [0]))
            print(f"  Eval | T_EMV: {t_emv_avg:.2f} | T_REV: {t_rev_avg:.2f}")

    # Cleanup
    envs.close()
    if eval_envs is not None:
        eval_envs.close()

    print(f"\nTraining complete. Results saved to: {run_dir}")
    print(f"Average T_EMV: {np.mean(t_emv_times):.2f}, Average T_REV: {np.mean(t_rev_times):.2f}")


def collect_step(envs, trainer, all_args, device):
    """Collect one step of actions from the policy."""
    return trainer.policy.get_actions(
        cent_obs=np.concatenate([envs.share_obs[0]]),
        obs=np.concatenate([envs.obs[0]]),
        rnn_states_actor=np.concatenate([envs.rnn_states[0]]),
        rnn_states_critic=np.concatenate([envs.rnn_states_critic[0]]),
        masks=np.concatenate([envs.masks[0]]),
        graph=np.concatenate([np.array(envs.graph_obs[0])]),
        available_actions=np.concatenate([envs.available_actions[0]]),
        deterministic=False,
    )


def insert_step(trainer, envs, all_args, device, obs, rewards, dones, infos,
                values, actions, action_log_probs, rnn_states, rnn_states_critic, step):
    """Insert collected data into buffer."""
    masks = np.ones((all_args.n_rollout_threads, all_args.num_agents, 1), dtype=np.float32)
    masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

    if all_args.use_centralized_V:
        share_obs = obs.reshape(all_args.n_rollout_threads, -1)
        share_obs = np.expand_dims(share_obs, 1).repeat(all_args.num_agents, axis=1)
    else:
        share_obs = obs

    rnn_states[dones == True] = np.zeros(
        ((dones == True).sum(), all_args.recurrent_N, all_args.hidden_size), dtype=np.float32
    )

    # Use buffer's insert method (same interface as RMTC)
    if hasattr(trainer, 'buffer') and trainer.buffer is not None:
        trainer.buffer.insert(
            share_obs=share_obs, obs=obs,
            rnn_states=rnn_states.copy(), rnn_states_critic=rnn_states_critic.copy(),
            actions=actions, action_log_probs=action_log_probs,
            values=values, rewards=rewards, masks=masks,
            available_actions=np.concatenate([envs.available_actions[0]])
            if hasattr(envs, 'available_actions') and envs.available_actions is not None else None,
        )


def compute_returns(trainer, envs, all_args, device):
    """Compute value function targets for the buffer."""
    if trainer._use_popart or trainer._use_valuenorm:
        advantages = trainer.buffer.returns[:] - trainer.value_normalizer.denormalize(
            trainer.buffer.value_preds[:]
        )
    else:
        advantages = trainer.buffer.returns[:] - trainer.buffer.value_preds[:]

    # Normalize advantages (same as RMTC R_MAPPO.train)
    advantages_copy = advantages.copy()
    advantages_copy[trainer.buffer.active_masks[:trainer.buffer.step] == 0.0] = np.nan
    mean_adv = np.nanmean(advantages_copy)
    std_adv = np.nanstd(advantages_copy)
    if std_adv > 1e-8:
        trainer.buffer.advantages = (advantages - mean_adv) / (std_adv + 1e-5)


def train_step(trainer, enhanced_rl, policy, all_args, device):
    """Perform one training update with enhanced RL losses."""
    # Standard PPO update
    train_info = trainer.train(trainer.buffer, update_actor=True)

    # Enhanced role encoder update (Pos-MI, Traj-MI, RoleCons)
    if hasattr(policy, 'gnn') and policy.gnn is not None:
        # Get role embeddings from HTTG at two timesteps
        graph_t = trainer.buffer.graph_obs[trainer.buffer.step - 1]
        graph_t1 = trainer.buffer.graph_obs[trainer.buffer.step]

        with torch.no_grad():
            h_t = policy.gnn(graph_t.to(device))
            h_tl = policy.gnn.get_dict({k: v for k, v in graph_t.items()})
            h_t1 = policy.gnn(graph_t1.to(device))
            h_tl1 = policy.gnn.get_dict({k: v for k, v in graph_t1.items()})

        # Extract EMV features for mutual information computation
        emv_pos = graph_t.get('emergency', torch.zeros(1)).to(device)
        # Traj-MI uses RNN-encoded trajectory (simplified here)
        emv_traj = h_t['emergency'][:emv_pos.shape[0]] if 'emergency' in h_t else emv_pos

        # Compute enhanced losses
        role_embeddings_t = {k: v for k, v in h_tl.items() if k != 'emergency'}
        role_embeddings_t1 = {k: v for k, v in h_tl1.items() if k != 'emergency'}

        # Accumulate loss gradients for GNN optimizer
        enhanced_loss, mi_metrics = enhanced_rl.compute_total_loss(
            policy_loss=torch.tensor(train_info['policy_loss'], device=device),
            value_loss=torch.tensor(train_info['value_loss'], device=device),
            critic_loss=torch.tensor(train_info['value_loss'] * trainer.value_loss_coef, device=device),
            role_embeddings_t=list(role_embeddings_t.values())[0] if role_embeddings_t else torch.zeros(1, device=device),
            role_embeddings_t1=list(role_embeddings_t1.values())[0] if role_embeddings_t1 else torch.zeros(1, device=device),
            emv_pos_features=emv_pos[:32],
            emv_traj_features=emv_traj[:32],
        )

        # Update GNN optimizer with enhanced loss
        policy.gnn_optimizer.zero_grad()
        enhanced_loss.backward()
        if all_args.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(policy.gnn.parameters(), all_args.max_grad_norm)
        policy.gnn_optimizer.step()

        train_info['mi_metrics'] = mi_metrics

    return train_info


def evaluate(eval_envs, trainer, all_args, device, run_dir, episode):
    """Evaluate trained model and collect T_EMV/T_REV metrics."""
    eval_episode_rewards_emv = np.zeros((all_args.n_eval_rollout_threads, all_args.num_emv_agents, 1))
    eval_episode_rewards_rev = np.zeros((all_args.n_eval_rollout_threads, all_args.num_vehicle_agents, 1))

    eval_obs_tl = eval_envs.reset()[0] if hasattr(eval_envs, 'reset') else None
    eval_rnn_states_tl = np.zeros(
        (all_args.n_eval_rollout_threads, *trainer.buffer.rnn_states.shape[2:]), dtype=np.float32
    )
    eval_masks_tl = np.ones((all_args.n_eval_rollout_threads, all_args.num_agents, 1), dtype=np.float32)

    if hasattr(eval_envs, 'available_actions') and eval_envs.available_actions is not None:
        ava = eval_envs.get_unava_phase_index() if hasattr(eval_envs, 'get_unava_phase_index') else None
        available_actions_tl = get_available_actions(ava, all_args) if ava is not None else None
    else:
        available_actions_tl = None

    t_emv_list = []
    t_rev_list = []

    for eval_step in range(10000):
        trainer.policy.prep_rollout()
        eval_action_tl, rnn_states_tl = trainer.policy.policy.act(
            np.concatenate(eval_obs_tl) if eval_obs_tl is not None else None,
            np.concatenate(eval_rnn_states_tl),
            np.concatenate(eval_masks_tl),
            eval_envs.reset()[1] if hasattr(eval_envs, 'reset') and eval_step == 0 else None,
            available_actions=available_actions_tl,
            deterministic=True,
        )

        eval_actions_tl = np.array(np.split(
            eval_action_tl.detach().cpu().numpy(), all_args.n_eval_rollout_threads
        ))
        rnn_states_tl = np.array(np.split(
            rnn_states_tl.detach().cpu().numpy(), all_args.n_eval_rollout_threads
        ))

        if eval_obs_tl is not None:
            obs_tl, rewards_tl, dones_tl, return_ve, infos_tl \
                = eval_envs.step(eval_actions_tl.astype(np.int64)[:, :, 0])
        else:
            break

        eval_rnn_states_tl = rnn_states_tl if rnn_states_tl is not None else eval_rnn_states_tl
        eval_masks_tl[dones_tl == True] = np.zeros(
            ((dones_tl == True).sum(), 1), dtype=np.float32
        )
        eval_rnn_states_tl[dones_tl == True] = np.zeros(
            ((dones_tl == True).sum(), all_args.recurrent_N, all_args.hidden_size),
            dtype=np.float32
        )

        if dones_tl.all():
            # Collect metrics from env info
            if 'emv_travel_time' in infos_tl:
                t_emv_list.append(infos_tl['emv_travel_time'])
            if 'rev_travel_time' in infos_tl:
                t_rev_list.append(infos_tl['rev_travel_time'])
            break

    return {"t_emv_list": t_emv_list, "t_rev_list": t_rev_list}


def get_available_actions(ava, all_args):
    """Generate available actions mask (same as RMTC)."""
    num_agents = getattr(all_args, 'num_agents', 16)
    num_actions = getattr(all_args, 'num_actions', 8)
    available_actions = np.ones((all_args.n_rollout_threads, num_agents, num_actions))

    if ava is not None and len(ava.shape) == 2:
        for i in range(num_agents):
            for j in range(all_args.n_rollout_threads):
                if ava[j][i] is not None:
                    available_actions[j, i, ava[j][i]] = 0
    return available_actions


if __name__ == "__main__":
    main()
