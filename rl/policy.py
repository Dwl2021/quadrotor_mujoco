from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

STATE_SHAPE = (3, 3, 2, 2)
NUM_ACTIONS = 2
STATE_SPACE_SIZE = np.prod(STATE_SHAPE)


def encode_state(state: Tuple[int, int, int, int]) -> int:
    s0, s1, s2, s3 = state
    if not (
        0 <= s0 < STATE_SHAPE[0]
        and 0 <= s1 < STATE_SHAPE[1]
        and 0 <= s2 < STATE_SHAPE[2]
        and 0 <= s3 < STATE_SHAPE[3]
    ):
        raise ValueError(f"State {state} out of bounds for shape {STATE_SHAPE}")
    index = (
        ((s0 * STATE_SHAPE[1] + s1) * STATE_SHAPE[2] + s2) * STATE_SHAPE[3] + s3
    )
    return int(index)


def decode_state(index: int) -> Tuple[int, int, int, int]:
    if not (0 <= index < STATE_SPACE_SIZE):
        raise ValueError(f"Index {index} out of range for state space size {STATE_SPACE_SIZE}")
    s3 = index % STATE_SHAPE[3]
    index //= STATE_SHAPE[3]
    s2 = index % STATE_SHAPE[2]
    index //= STATE_SHAPE[2]
    s1 = index % STATE_SHAPE[1]
    index //= STATE_SHAPE[1]
    s0 = index
    return s0, s1, s2, s3


@dataclass
class CollisionPolicy:
    q_table: np.ndarray = field(default_factory=lambda: np.zeros((STATE_SPACE_SIZE, NUM_ACTIONS), dtype=np.float32))

    def select_action(self, state_idx: int, epsilon: float = 0.0, rng: Optional[np.random.Generator] = None) -> int:
        if rng is None:
            rng = np.random.default_rng()
        if epsilon > 0.0 and rng.random() < epsilon:
            return int(rng.integers(0, NUM_ACTIONS))
        return int(np.argmax(self.q_table[state_idx]))

    def update(self, state_idx: int, action: int, target: float, lr: float) -> None:
        self.q_table[state_idx, action] += lr * (target - self.q_table[state_idx, action])


PathLike = Union[str, Path]


def save_policy(path: PathLike, policy: CollisionPolicy, metadata: Optional[dict] = None) -> None:
    path = Path(path)  # type: ignore[arg-type]
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"q_table": policy.q_table}
    if metadata is not None:
        payload["metadata"] = json.dumps(metadata)
    np.savez_compressed(path, **payload)


def load_policy(path: PathLike) -> CollisionPolicy:
    path = Path(path)  # type: ignore[arg-type]
    if not path.exists():
        raise FileNotFoundError(f"Policy file not found: {path}")
    payload = np.load(path, allow_pickle=True)
    q_table = payload["q_table"]
    return CollisionPolicy(q_table=q_table)
