"""
Reinforcement Learning environment wrappers for optimisation.

This module provides callable classes that wrap Gymnasium_ (with a fallback
to the legacy ``gym`` package) environments so they can be used as black-box
objective functions by the ASHGF optimiser.

Each class instantiates a neural-network policy parameterised by the vector
``x`` (weights and biases folded into a single 1-D array) and runs a
fixed-length episode.  The scalar return value is the cumulative reward,
which the optimiser tries to **maximise**.

.. _Gymnasium: https://gymnasium.farama.org/

Environment details
-------------------
==================================== =========== =========== ========
Environment                          ``obs_dim`` ``act_dim`` ``h``
==================================== =========== =========== ========
``RLEnvironmentPendulum`` (Pendulum) 3           1           5
``RLEnvironmentCartPole`` (CartPole) 4           1           4
==================================== =========== =========== ========

All environments use a single-hidden-layer ReLU network with hidden size
``h``.  The policy parameter vector ``x`` must have length
``obs_dim * h + act_dim * h``.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.special import expit

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _relu(x: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit activation.

    .. math::

        \\operatorname{ReLU}(z) = \\max(0, z) = |z| \\cdot \\mathbf{1}_{z > 0}

    Parameters
    ----------
    x : np.ndarray
        Input array of any shape.

    Returns
    -------
    np.ndarray
        Element-wise ReLU of ``x``.
    """
    return np.abs(x) * (x > 0)


def _get_gym_module() -> tuple[object, str]:
    """Attempt to import gymnasium, falling back to gym.

    Returns
    -------
    tuple[module, str]
        The imported module and a string identifier (``"gymnasium"`` or
        ``"gym"``).

    Raises
    ------
    ImportError
        If neither ``gymnasium`` nor ``gym`` can be imported.
    """
    try:
        import gymnasium as gym_module

        logger.debug("Using gymnasium for RL environments.")
        return gym_module, "gymnasium"
    except ImportError:
        logger.debug("gymnasium not found; trying legacy gym.")
        try:
            import gym as gym_module

            logger.debug("Using legacy gym for RL environments.")
            return gym_module, "gym"
        except ImportError:
            logger.error(
                "Neither gymnasium nor gym is installed. "
                "RL environments are unavailable."
            )
            raise ImportError(
                "Neither gymnasium nor gym is installed. "
                "Install one of them to use RL environments."
            )


# ---------------------------------------------------------------------------
# RL environment classes
# ---------------------------------------------------------------------------


class RLEnvironmentPendulum:
    r"""Pendulum-v1 RL environment wrapper for optimisation.

    The Pendulum environment requires the agent to swing up and balance a
    pendulum by applying a continuous torque in :math:`[-2, 2]`.  The
    observation is 3-D (cos(theta), sin(theta), angular velocity) and the
    action is 1-D.

    The policy is a single-hidden-layer ReLU network of width ``h = 5``
    with a sigmoid output that is scaled to the action range.

    Parameters
    ----------
    seed_env : int
        Random seed passed to the environment's ``reset`` method for
        reproducibility.

    Notes
    -----
    The parameter vector ``x`` must be of length
    ``obs_dim * h + act_dim * h = 3 * 5 + 1 * 5 = 20``, arranged as:

    * ``x[:15]`` → weight matrix :math:`W_1` of shape ``(3, 5)``
      (row-major).
    * ``x[15:]`` → weight matrix :math:`W_2` of shape ``(1, 5)``
      (row-major).

    The network computes:
    :math:`a = \sigma\bigl(W_2\,\operatorname{ReLU}(W_1^T o)\bigr)`
    where :math:`o` is the observation vector and :math:`\sigma` is the
    logistic sigmoid.
    """

    # Network architecture constants
    _OBS_DIM: int = 3
    _ACT_DIM: int = 1
    _HIDDEN: int = 5  # h
    _EPISODE_LENGTH: int = 200

    def __init__(self, seed_env: int = 0) -> None:
        self.seed_env: int = seed_env

    def __call__(self, x: np.ndarray) -> float:
        """Evaluate the policy parameterised by ``x``.

        Parameters
        ----------
        x : np.ndarray
            Policy parameter vector of shape ``(20,)``.

        Returns
        -------
        float
            Total cumulative reward over the episode (to be maximised).
        """
        gym_module, _ = _get_gym_module()
        env = gym_module.make("Pendulum-v1")

        obs_space: int = self._OBS_DIM
        act_space: int = self._ACT_DIM
        h: int = self._HIDDEN

        try:
            observation, _ = env.reset(seed=self.seed_env)
            rewards: list[float] = []

            W_1: np.ndarray = x[: obs_space * h].reshape((obs_space, h))
            W_2: np.ndarray = x[obs_space * h :].reshape((act_space, h))

            for _ in range(self._EPISODE_LENGTH):
                action: np.ndarray = expit(
                    np.dot(
                        W_2,
                        _relu(
                            np.dot(
                                W_1.T,
                                observation.reshape((obs_space, 1)),
                            )
                        ),
                    )
                )
                observation, reward, terminated, truncated, _info = env.step(action)
                rewards.append(float(reward))
                if terminated or truncated:
                    break
        finally:
            env.close()

        return float(np.sum(rewards))


class RLEnvironmentCartPole:
    r"""CartPole-v1 RL environment wrapper for optimisation.

    The CartPole environment requires the agent to balance a pole on a cart
    by applying a discrete force (left or right).  The observation is 4-D
    (cart position, cart velocity, pole angle, pole angular velocity) and
    the action is discrete with two values (0 or 1).

    The policy is a single-hidden-layer ReLU network of width ``h = 4``
    whose output is thresholded at zero to produce the discrete action.

    Parameters
    ----------
    seed_env : int
        Random seed passed to the environment's ``reset`` method for
        reproducibility.

    Notes
    -----
    The parameter vector ``x`` must be of length
    ``obs_dim * h + act_dim * h = 4 * 4 + 1 * 4 = 20``, arranged as:

    * ``x[:16]`` → weight matrix :math:`W_1` of shape ``(4, 4)``
      (row-major).
    * ``x[16:]`` → weight matrix :math:`W_2` of shape ``(1, 4)``
      (row-major).

    The network computes:
    :math:`a = W_2\,\operatorname{ReLU}(W_1^T o)`.
    If :math:`a > 0` the action is ``1``, otherwise ``0``.
    """

    # Network architecture constants
    _OBS_DIM: int = 4
    _ACT_DIM: int = 1
    _HIDDEN: int = 4  # h
    _EPISODE_LENGTH: int = 200

    def __init__(self, seed_env: int = 0) -> None:
        self.seed_env: int = seed_env

    def __call__(self, x: np.ndarray) -> float:
        """Evaluate the policy parameterised by ``x``.

        Parameters
        ----------
        x : np.ndarray
            Policy parameter vector of shape ``(20,)``.

        Returns
        -------
        float
            Total cumulative reward over the episode (to be maximised).
        """
        gym_module, _ = _get_gym_module()
        env = gym_module.make("CartPole-v1")

        obs_space: int = self._OBS_DIM
        act_space: int = self._ACT_DIM
        h: int = self._HIDDEN

        try:
            observation, _ = env.reset(seed=self.seed_env)
            rewards: list[float] = []

            W_1: np.ndarray = x[: obs_space * h].reshape((obs_space, h))
            W_2: np.ndarray = x[obs_space * h :].reshape((act_space, h))

            for _ in range(self._EPISODE_LENGTH):
                action_raw: float = float(
                    np.dot(
                        W_2,
                        _relu(
                            np.dot(
                                W_1.T,
                                observation.reshape((obs_space, 1)),
                            )
                        ),
                    )
                )
                action: int = 1 if action_raw > 0 else 0
                observation, reward, terminated, truncated, _info = env.step(action)
                rewards.append(float(reward))
                if terminated or truncated:
                    break
        finally:
            env.close()

        return float(np.sum(rewards))
