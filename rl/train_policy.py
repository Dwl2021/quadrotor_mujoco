from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .env_collision import QuadrotorCollisionEnv
from .policy import CollisionPolicy, encode_state, save_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train collision-aware rebound policy with tabular Q-learning.")
    parser.add_argument("--episodes", type=int, default=2500, help="Number of training episodes.")
    parser.add_argument("--lr", type=float, default=0.12, help="Q-learning learning rate.")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor.")
    parser.add_argument("--epsilon-start", type=float, default=0.2, help="Initial epsilon for epsilon-greedy exploration.")
    parser.add_argument("--epsilon-min", type=float, default=0.02, help="Minimum epsilon.")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="Multiplicative epsilon decay per episode.")
    parser.add_argument("--output", type=Path, default=Path("policies/collision_policy.npz"), help="Output path for trained policy.")
    parser.add_argument("--log-interval", type=int, default=50, help="Episodes between progress logs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    env = QuadrotorCollisionEnv(seed=args.seed)
    policy = CollisionPolicy()

    print(
        f"Starting training for {args.episodes} episodes "
        f"(log every {args.log_interval} episodes)...",
        flush=True,
    )

    epsilon = args.epsilon_start
    best_mean_reward = -np.inf
    reward_history: list[float] = []
    success_history: list[float] = []
    start_time = time.time()

    for episode in range(1, args.episodes + 1):
        obs = env.reset(seed=rng.integers(0, 1_000_000))
        done = False
        total_reward = 0.0
        last_info: dict = {}

        while not done:
            state_tuple = env.discretise_observation(obs)
            state_idx = encode_state(state_tuple)
            action = policy.select_action(state_idx, epsilon=epsilon, rng=rng)

            next_obs, reward, done, info = env.step(action)
            last_info = info
            total_reward += reward

            next_state_tuple = env.discretise_observation(next_obs)
            next_state_idx = encode_state(next_state_tuple)

            target = reward
            if not done:
                target += args.gamma * np.max(policy.q_table[next_state_idx])

            policy.update(state_idx, action, target, args.lr)
            obs = next_obs

        reward_history.append(total_reward)
        success_history.append(1.0 if last_info.get("success", False) else 0.0)

        epsilon = max(args.epsilon_min, epsilon * args.epsilon_decay)

        if episode % args.log_interval == 0:
            recent_rewards = reward_history[-args.log_interval :]
            recent_success = success_history[-args.log_interval :]
            mean_reward = float(np.mean(recent_rewards))
            success_rate = float(np.mean(recent_success))
            best_mean_reward = max(best_mean_reward, mean_reward)
            elapsed = time.time() - start_time
            print(
                f"[Episode {episode:4d}/{args.episodes}] "
                f"avg_reward={mean_reward:.3f} "
                f"success_rate={success_rate:.2%} "
                f"epsilon={epsilon:.3f} "
                f"time={elapsed:.1f}s"
            , flush=True)

    metadata = {
        "episodes": args.episodes,
        "learning_rate": args.lr,
        "gamma": args.gamma,
        "epsilon_start": args.epsilon_start,
        "epsilon_min": args.epsilon_min,
        "epsilon_decay": args.epsilon_decay,
        "best_mean_reward": best_mean_reward,
        "timestamp": time.time(),
    }
    save_policy(args.output, policy, metadata=metadata)
    env.close()
    print(f"Policy saved to {args.output}", flush=True)


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
