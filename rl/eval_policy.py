from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import mujoco.viewer as mjviewer

from .env_collision import QuadrotorCollisionEnv
from typing import Optional

from .policy import CollisionPolicy, encode_state, load_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained collision-aware rebound policy.")
    parser.add_argument("--policy", type=Path, default=Path("policies/collision_policy.npz"), help="Path to trained policy file.")
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument("--render", action="store_true", help="Open MuJoCo viewer to visualise each episode.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep after each viewer sync (slows down playback).")
    parser.add_argument("--baseline", action="store_true", help="Disable the learned collision response (no policy, no rebound).")
    return parser.parse_args()


def evaluate(policy: Optional[CollisionPolicy], args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    env = QuadrotorCollisionEnv(seed=args.seed, policy_enabled=not args.baseline)

    rewards: list[float] = []
    successes: list[bool] = []
    detection_latencies: list[float] = []

    viewer_ctx = None
    viewer = None

    if args.render:
        viewer_ctx = mjviewer.launch_passive(env.model, env.data)
        viewer = viewer_ctx.__enter__()

    try:
        for episode in range(1, args.episodes + 1):
            episode_seed = rng.integers(0, 1_000_000)
            obs = env.reset(seed=episode_seed)
            if viewer is not None:
                viewer.sync()
            done = False
            total_reward = 0.0
            collision_time = None
            trigger_time = None

            while not done:
                state_tuple = env.discretise_observation(obs)
                if policy is not None:
                    state_idx = encode_state(state_tuple)
                    action = policy.select_action(state_idx, epsilon=0.0, rng=rng)
                else:
                    action = 0  # Baseline: never trigger rebound
                obs, reward, done, info = env.step(action)
                total_reward += reward

                if viewer is not None:
                    viewer.sync()
                    if args.sleep > 0:
                        time.sleep(args.sleep)

                if collision_time is None and info.get("collision_time") is not None:
                    collision_time = info["collision_time"]

                if trigger_time is None and info.get("trigger_time") is not None:
                    trigger_time = info["trigger_time"]

            rewards.append(total_reward)
            successes.append(bool(info.get("success", False)))

            if collision_time is not None and trigger_time is not None:
                detection_latencies.append(max(0.0, trigger_time - collision_time))
    finally:
        if viewer_ctx is not None:
            viewer_ctx.__exit__(None, None, None)

    env.close()

    mean_reward = float(np.mean(rewards)) if rewards else 0.0
    success_rate = float(np.mean(successes)) if successes else 0.0
    mean_latency = float(np.mean(detection_latencies)) if detection_latencies else float("nan")

    print(f"Evaluated {args.episodes} episodes", flush=True)
    print(f"Average reward     : {mean_reward:.3f}", flush=True)
    print(f"Success rate       : {success_rate:.2%}", flush=True)
    if detection_latencies:
        print(f"Detection latency  : {mean_latency*1000:.1f} ms (mean)", flush=True)
    else:
        print("Detection latency  : n/a", flush=True)


def main() -> None:
    args = parse_args()
    policy = None if args.baseline else load_policy(args.policy)
    evaluate(policy, args)


if __name__ == "__main__":
    main()
