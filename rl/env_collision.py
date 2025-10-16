from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import mujoco
import numpy as np

from motor_mixer import Mixer
from quad_utils import PARAMS, calc_motor_input
from se3_controller import SE3Controller, State


FORWARD_DIRECTION = np.array([1.0, 0.0, 0.0])


@dataclass
class CollisionObservation:
    velocity: np.ndarray
    acceleration: np.ndarray
    delta_velocity: np.ndarray
    delta_acceleration: np.ndarray
    contact: bool
    rebound_alignment: float
    mode: str
    time_since_collision: float

    def as_vector(self) -> np.ndarray:
        """Return a flat vector representation suitable for learning."""
        return np.concatenate(
            [
                self.velocity,
                self.acceleration,
                self.delta_velocity,
                self.delta_acceleration,
                np.array(
                    [
                        float(self.contact),
                        self.rebound_alignment,
                        self.time_since_collision,
                    ]
                ),
            ]
        )


class QuadrotorCollisionEnv:
    """MuJoCo-based environment for collision-triggered rebound control."""

    def __init__(
        self,
        backoff_distance: float = 1.0,
        frame_skip: int = 5,
        detection_window: float = 0.05,
        max_episode_time: float = 3.0,
        seed: Optional[int] = None,
        policy_enabled: bool = True,
    ):
        self.model = mujoco.MjModel.from_xml_path("crazyfile/scene.xml")
        self.data = mujoco.MjData(self.model)
        self.controller = SE3Controller()
        self.controller.kx = 0.6
        self.controller.kv = 0.4
        self.controller.kR = 6.0
        self.controller.kw = 1.0
        self.mixer = Mixer()

        self.backoff_distance = backoff_distance
        self.backoff_speed = 1.2
        self.frame_skip = frame_skip
        self.detection_window = detection_window
        self.max_episode_time = max_episode_time

        self.gravity = PARAMS.gravity
        self.mass = PARAMS.mass
        self.torque_scale = 0.001
        self.dt = self.model.opt.timestep

        self.wall_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "wall")
        if self.wall_geom_id < 0:
            raise RuntimeError("Wall geometry 'wall' not found in scene.")

        self.rng = np.random.default_rng(seed)

        # Episode state
        self.prev_velocity = np.zeros(3)
        self.prev_acceleration = np.zeros(3)
        self.collision_detected = False
        self.collision_time: Optional[float] = None
        self.trigger_time: Optional[float] = None
        self.pre_collision_velocity = np.zeros(3)
        self.mode = "forward"
        self.triggered = False
        self.false_positive = False
        self.backoff_direction = FORWARD_DIRECTION.copy()
        self.backoff_start_position = np.zeros(3)
        self.backoff_target = np.zeros(3)
        self.done = False
        self.success = False
        self.step_count = 0
        self.policy_enabled = policy_enabled

        # Rewards
        self.false_positive_penalty = 0.6
        self.correct_trigger_reward = 1.2
        self.success_reward = 1.5
        self.missed_detection_penalty = 1.2
        self.stability_penalty_scale = 0.0005
        self.progress_reward_scale = 0.2
        self.step_penalty = 0.001

        self._randomise_forward_profile()
        mujoco.mj_forward(self.model, self.data)
        self.prev_velocity = self.data.qvel[:3].copy()
        self.prev_acceleration = self.data.sensordata[3:6].copy()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)
        # Initialise pose slightly above the ground for consistent episodes.
        self.data.qpos[:3] = np.array([0.0, 0.0, 0.05])
        self.data.qpos[3:7] = np.array([0.0, 0.0, 0.0, 1.0])
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.prev_velocity = self.data.qvel[:3].copy()
        self.prev_acceleration = self.data.sensordata[3:6].copy()
        self.collision_detected = False
        self.collision_time = None
        self.trigger_time = None
        self.pre_collision_velocity = np.zeros(3)
        self.mode = "forward"
        self.triggered = False
        self.false_positive = False
        self.backoff_direction = FORWARD_DIRECTION.copy()
        self.backoff_start_position = self.data.qpos[:3].copy()
        self.backoff_target = self.backoff_start_position.copy()
        self.done = False
        self.success = False
        self.step_count = 0

        self._randomise_forward_profile()

        return self._build_observation().as_vector()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        if self.done:
            raise RuntimeError("Call reset() before stepping a finished episode.")
        if action not in (0, 1):
            raise ValueError("Action must be 0 (hold trajectory) or 1 (trigger rebound).")

        reward = -self.step_penalty
        info: dict = {}

        # Handle high-level action.
        if action == 1 and not self.triggered and self.policy_enabled:
            if not self.collision_detected:
                self.false_positive = True
                reward -= self.false_positive_penalty
                self._start_backoff(velocity=self.prev_velocity, false_positive=True)
            else:
                reward += self.correct_trigger_reward
                self._start_backoff(velocity=self.pre_collision_velocity, false_positive=False)
                self.trigger_time = self.data.time

        # Run low-level control for frame_skip steps.
        for _ in range(self.frame_skip):
            goal_pos, goal_vel, goal_acc, goal_heading = self._goal_for_mode()
            self._apply_controller(goal_pos, goal_vel, goal_acc, goal_heading)
            mujoco.mj_step(self.model, self.data)
            self._update_collision_state()
            if self.done:
                break

        obs = self._build_observation()
        velocity = obs.velocity
        acceleration = obs.acceleration
        delta_velocity = obs.delta_velocity

        # Reward shaping
        if self.mode == "backoff" and self.triggered:
            progress = np.dot(self.data.qpos[:3] - self.backoff_start_position, self.backoff_direction)
            progress = np.clip(progress, 0.0, self.backoff_distance)
            reward += self.progress_reward_scale * (progress / self.backoff_distance)

        # Penalise angular motion spikes to keep vehicle stable.
        omega = self.data.sensordata[:3]
        reward -= self.stability_penalty_scale * float(np.linalg.norm(omega) ** 2)
        reward -= 0.0005 * float(np.linalg.norm(delta_velocity))

        # Missed detection handling.
        if self.collision_detected and not self.triggered and self.policy_enabled:
            time_since_collision = self.data.time - (self.collision_time or self.data.time)
            if time_since_collision > self.detection_window:
                reward -= self.missed_detection_penalty
                self.done = True

        # Episode termination checks.
        if self.data.time >= self.max_episode_time:
            self.done = True

        if self.false_positive and self.triggered:
            # End the episode early after applying the penalty.
            self.done = True

        if self.mode == "settle" and self.triggered and not self.false_positive:
            if (
                np.linalg.norm(velocity) < 0.05
                and np.linalg.norm(self.data.qpos[:3] - self.backoff_target) < 0.05
            ):
                self.success = True
                reward += self.success_reward
                self.done = True

        self.prev_velocity = velocity.copy()
        self.prev_acceleration = acceleration.copy()
        self.step_count += 1

        info.update(
            {
                "collision_detected": self.collision_detected,
                "triggered": self.triggered,
                "false_positive": self.false_positive,
                "success": self.success,
                "mode": self.mode,
                "time": self.data.time,
                "trigger_time": self.trigger_time,
                "collision_time": self.collision_time,
            }
        )

        return obs.as_vector(), reward, self.done, info

    def close(self) -> None:
        """Release MuJoCo resources."""
        del self.data
        del self.model

    # ------------------------------------------------------------------ #
    # Helper methods
    # ------------------------------------------------------------------ #
    def _apply_controller(
        self,
        goal_pos: np.ndarray,
        goal_vel: np.ndarray,
        goal_acc: np.ndarray,
        goal_heading: np.ndarray,
    ) -> None:
        current_pos = self.data.qpos[:3].copy()
        current_vel = self.data.qvel[:3].copy()
        sensor = self.data.sensordata
        gyro = sensor[:3].copy()
        linacc = sensor[3:6].copy()
        quat = np.array([sensor[7], sensor[8], sensor[9], sensor[6]])  # x y z w

        curr_state = State(current_pos, current_vel, quat, gyro, acc=linacc)
        goal_quat = np.array([0.0, 0.0, 0.0, 1.0])
        goal_omega = np.zeros(3)
        goal_state = State(goal_pos, goal_vel, goal_quat, goal_omega, acc=goal_acc)

        command = self.controller.control_update(curr_state, goal_state, self.dt, goal_heading)
        thrust = command.thrust * self.gravity * self.mass
        torque = command.angular * self.torque_scale
        thrust = float(np.clip(thrust, 0.0, self.mixer.max_thrust * 4.0))
        torque = np.clip(torque, -self.mixer.max_torque, self.mixer.max_torque)
        motor_speed = self.mixer.calculate(thrust, torque[0], torque[1], torque[2])
        motor_speed = np.nan_to_num(motor_speed, nan=0.0, posinf=self.mixer.max_speed, neginf=0.0)
        motor_speed = np.clip(motor_speed, 0.0, self.mixer.max_speed)
        self.data.ctrl[:] = calc_motor_input(motor_speed)

    def _update_collision_state(self) -> None:
        collision_with_wall = False
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.geom1 == self.wall_geom_id or contact.geom2 == self.wall_geom_id:
                collision_with_wall = True
                break

        if collision_with_wall and not self.collision_detected:
            self.collision_detected = True
            self.collision_time = self.data.time
            self.pre_collision_velocity = self.prev_velocity.copy()
        elif not collision_with_wall and self.collision_detected and self.mode != "backoff":
            # Contact resolved but no rebound triggered yet.
            pass

    def _build_observation(self) -> CollisionObservation:
        velocity = self.data.qvel[:3].copy()
        acceleration = self.data.sensordata[3:6].copy()
        delta_velocity = velocity - self.prev_velocity
        delta_acceleration = acceleration - self.prev_acceleration
        forward_speed = float(np.dot(velocity, FORWARD_DIRECTION))
        contact = self.collision_detected
        if self.collision_time is None:
            time_since_collision = -1.0
        else:
            time_since_collision = max(0.0, self.data.time - self.collision_time)
        return CollisionObservation(
            velocity=velocity,
            acceleration=acceleration,
            delta_velocity=delta_velocity,
            delta_acceleration=delta_acceleration,
            contact=contact,
            rebound_alignment=forward_speed,
            mode=self.mode,
            time_since_collision=time_since_collision,
        )

    def _goal_for_mode(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.mode == "forward":
            return self._forward_trajectory(self.data.time)

        if self.mode == "backoff":
            progress = np.dot(self.data.qpos[:3] - self.backoff_start_position, self.backoff_direction)
            if progress >= self.backoff_distance - 0.05:
                self.mode = "settle"
            goal_pos = self.backoff_target
            goal_vel = self.backoff_direction * self.backoff_speed * 0.2
            if np.linalg.norm(self.data.qpos[:3] - self.backoff_target) < 0.1:
                goal_vel = np.zeros(3)
            goal_acc = np.zeros(3)
            goal_heading = FORWARD_DIRECTION
            return goal_pos, goal_vel, goal_acc, goal_heading

        # settle
        goal_pos = self.backoff_target
        goal_vel = np.zeros(3)
        goal_acc = np.zeros(3)
        goal_heading = FORWARD_DIRECTION
        return goal_pos, goal_vel, goal_acc, goal_heading

    def _forward_trajectory(self, time: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        hover_time = self.forward_profile["hover_time"]
        height = self.forward_profile["height"]
        attack_acc = self.forward_profile["attack_acc"]
        max_speed = self.forward_profile["max_speed"]
        goal_heading = FORWARD_DIRECTION

        if time < hover_time:
            pos = np.array([0.0, 0.0, height])
            vel = np.zeros(3)
            acc = np.zeros(3)
            return pos, vel, acc, goal_heading

        tau = time - hover_time
        accel_time = max_speed / attack_acc
        accel_distance = 0.5 * attack_acc * accel_time**2

        if tau < accel_time:
            pos_x = 0.5 * attack_acc * tau**2
            vel_x = attack_acc * tau
            acc_x = attack_acc
        else:
            cruise_time = tau - accel_time
            pos_x = accel_distance + max_speed * cruise_time
            vel_x = max_speed
            acc_x = 0.0

        pos = np.array([pos_x, 0.0, height])
        vel = np.array([vel_x, 0.0, 0.0])
        acc = np.array([acc_x, 0.0, 0.0])
        return pos, vel, acc, goal_heading

    def _start_backoff(self, velocity: np.ndarray, false_positive: bool) -> None:
        direction = -velocity
        if np.linalg.norm(direction) < 1e-3:
            direction = -FORWARD_DIRECTION
        direction = direction / np.linalg.norm(direction)
        self.backoff_direction = direction
        self.backoff_start_position = self.data.qpos[:3].copy()
        self.backoff_target = self.backoff_start_position + direction * self.backoff_distance
        self.mode = "backoff"
        self.triggered = True
        self.false_positive = false_positive

    def discretise_observation(self, obs_vector: np.ndarray) -> Tuple[int, int, int, int]:
        """Discretise observation vector into a compact state for tabular RL."""
        delta_velocity = obs_vector[6:9]
        delta_acc = obs_vector[9:12]
        contact_flag = bool(obs_vector[12])
        forward_speed = obs_vector[13]

        delta_v_norm = float(np.linalg.norm(delta_velocity))
        delta_a_norm = float(np.linalg.norm(delta_acc))

        speed_band = 0
        if delta_v_norm > 0.4:
            speed_band = 1
        if delta_v_norm > 0.9:
            speed_band = 2

        acc_band = 0
        if delta_a_norm > 4.0:
            acc_band = 1
        if delta_a_norm > 8.0:
            acc_band = 2

        rebound_flag = 1 if forward_speed < -0.1 else 0
        contact_int = 1 if contact_flag else 0

        return speed_band, acc_band, contact_int, rebound_flag

    def _randomise_forward_profile(self) -> None:
        self.forward_profile = {
            "hover_time": 0.8 + self.rng.uniform(-0.3, 0.2),
            "height": 0.28 + self.rng.uniform(-0.05, 0.05),
            "attack_acc": 3.0 + self.rng.uniform(-0.6, 0.6),
            "max_speed": 1.8 + self.rng.uniform(-0.3, 0.3),
        }
