from dataclasses import dataclass
from typing import Union

import numpy as np


@dataclass(frozen=True)
class QuadrotorParams:
    """Physical parameters for the Crazyflie 2 quadrotor."""

    gravity: float = 9.8066  # m/s^2
    mass: float = 0.033  # kg
    thrust_coeff: float = 3.25e-4  # N/krpm^2
    torque_coeff: float = 7.9379e-6  # Nm/krpm^2
    arm_length: float = 0.065 / 2.0  # m
    max_motor_speed: float = 22.0  # krpm
    max_motor_thrust: float = 0.1573  # N
    max_motor_torque: float = 3.842e-3  # Nm


PARAMS = QuadrotorParams()


ArrayLike = Union[float, np.ndarray]


def calc_motor_force(krpm: ArrayLike, params: QuadrotorParams = PARAMS) -> np.ndarray:
    """Convert motor speed (krpm) to thrust force (N)."""
    krpm = np.asarray(krpm)
    return params.thrust_coeff * krpm**2


def calc_motor_torque(krpm: ArrayLike, params: QuadrotorParams = PARAMS) -> np.ndarray:
    """Convert motor speed (krpm) to reaction torque (Nm)."""
    krpm = np.asarray(krpm)
    return params.torque_coeff * krpm**2


def calc_motor_speed_by_force(force: ArrayLike, params: QuadrotorParams = PARAMS) -> np.ndarray:
    """Return motor speed (krpm) for a desired thrust (N)."""
    force = np.asarray(force)
    clipped = np.clip(force, 0.0, params.max_motor_thrust)
    return np.sqrt(clipped / params.thrust_coeff)


def calc_motor_speed_by_torque(torque: ArrayLike, params: QuadrotorParams = PARAMS) -> np.ndarray:
    """Return motor speed (krpm) for a desired reaction torque (Nm)."""
    torque = np.asarray(torque)
    clipped = np.clip(np.abs(torque), 0.0, params.max_motor_torque)
    return np.sqrt(clipped / params.torque_coeff)


def calc_motor_input(krpm: ArrayLike, params: QuadrotorParams = PARAMS) -> np.ndarray:
    """Normalise motor speeds (krpm) to actuator inputs in [0, 1]."""
    krpm = np.clip(krpm, 0.0, params.max_motor_speed)
    force = calc_motor_force(krpm, params)
    normalised = force / params.max_motor_thrust
    return np.clip(normalised, 0.0, 1.0)
