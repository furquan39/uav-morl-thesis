"""
Multi‑objective reinforcement learning implementation for a coverage path planning (CPP) 
problem.  The code in this module illustrates how to build a grid‑based CPP
environment for an unmanned aerial vehicle (UAV), define multiple reward
objectives, scalarise them to obtain a single signal and train an agent
using Proximal Policy Optimisation (PPO).  The implementation is self
contained – it does not depend on the thesis supervisor’s original code –
but follows the same structure: a grid world where the agent can move in
four directions and must cover all cells while avoiding obstacles.  To
extend the problem from single objective coverage to a multi‑objective
setting the environment returns a vector of rewards for each step and
complete coverage episode.  A separate wrapper combines the reward vector
into a scalar value using a weight vector.  By training the agent with
different weight vectors or by adjusting the weights during training one
can obtain policies that trade off between competing objectives and build
an approximation to the Pareto front of optimal solutions.

**Important:** The code uses `gymnasium` and `stable_baselines3` for the
environment API and PPO implementation.  Those libraries must be
installed in your Python environment (e.g. via `pip install gymnasium
stable-baselines3 torch`).  If they are not available the code will not
run.  The core logic of the environment and reward functions can be
reused in other RL frameworks or custom training loops.

### Environment design

The coverage environment is inspired by the discrete cell‑based model
described in Garrido‑Castañeda et al.’s 2025 Sensors paper【864997031185730†L580-L621】.
The UAV operates on an `m × n` grid where each cell may be free,
obstructed or already covered.  At each time step the agent moves one
cell in one of the four cardinal directions (north, east, south or
west)【864997031185730†L582-L618】.  The state is defined by the agent’s
coordinates, a vector of local sensor readings (indicating whether
neighbouring cells are free or obstructed) and the fraction of cells
already covered【864997031185730†L573-L579】.  Each action updates the
position, refreshes the sensor readings and marks newly visited cells as
covered【864997031185730†L631-L647】.  The episode terminates when all
cells have been covered.

### Reward vector

Following the reward shaping used by Garrido‑Castañeda et al., the
environment defines three separate reward components【864997031185730†L664-L700】:

* **Coverage reward** – a small positive value is awarded whenever the
  agent visits a previously uncovered cell, encouraging exploration and
  complete coverage【864997031185730†L671-L699】.
* **Step penalty** – every time step incurs a small negative reward to
  minimise the overall trajectory length【864997031185730†L668-L700】.
* **Collision penalty** – a significant negative reward is given when
  the agent attempts to move into a cell containing an obstacle; this
  discourages collisions and promotes safe navigation【864997031185730†L664-L700】.
  When a collision occurs the agent remains in its previous cell.

The final reward signal provided by the environment is a tuple of these
three values.  At episode termination a large positive reward is added
to the coverage component if the agent has successfully covered all
cells, otherwise a penalty is applied.

### Scalarisation wrapper

Multi‑objective reinforcement learning algorithms often require a
scalarised reward to update a single policy network.  A common approach
is to compute a weighted sum of the individual objectives, tuning the
weights to reflect the desired trade off【51935317549949†L65-L79】.  Note
that weighted sums may not cover non‑convex regions of the Pareto
front【51935317549949†L65-L79】, but they are widely used for their
simplicity.  The `ScalarisedEnv` class wraps the multi‑objective
environment and multiplies the reward vector by a user‑supplied weight
vector.  Adjusting the weight vector produces different scalar
rewards and, consequently, different trained policies.

### Training

The example at the bottom of this file demonstrates how to train a PPO
agent on the scalarised environment.  Multiple weight vectors can be
explored by iterating over them and training separate policies.  The
resulting policies can then be evaluated and compared to approximate
trade‑offs between coverage efficiency, path length and collision
avoidance.  While the code uses PPO for demonstration purposes, any
policy gradient or value‑based algorithm can be applied to the
scalarised environment.

To use this module:

1. Define your grid dimensions and obstacles.
2. Choose the reward weights reflecting your objectives (e.g.
   `[0.8, 0.1, 0.1]` emphasises coverage over path length and
   collisions).
3. Wrap the environment with `ScalarisedEnv` using those weights.
4. Train a PPO agent on the wrapper.
5. Repeat for different weight vectors to build a set of policies.

For brevity the default training loop is kept short.  Serious
experiments should increase the number of steps and adjust PPO
hyper‑parameters accordingly.

"""

from __future__ import annotations

import math
import numpy as np
from typing import Tuple, List, Optional, Dict

# The environment depends on gymnasium; if it is not installed this module
# will raise an ImportError.  Users should install gymnasium via pip.
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:
    raise ImportError(
        "gymnasium is required for this module. Install it with `pip install gymnasium`."
    )


class UAVCoverageEnv(gym.Env):
    """Grid‑based coverage path planning environment with vector rewards.

    Attributes
    ----------
    grid_size : Tuple[int, int]
        Dimensions of the coverage grid (rows, columns).
    obstacle_ratio : float
        Proportion of cells randomly initialised as obstacles (0 ≤ obstacle_ratio < 1).
    max_steps : int
        Maximum number of steps allowed per episode.  When reached the episode
        terminates even if not all cells have been covered.
    rng : np.random.Generator
        Random number generator for reproducible environment initialisation.
    state : np.ndarray
        Current state representation (agent position, sensor readings, coverage
        fraction).  Used as observation.
    coverage_mask : np.ndarray
        Boolean array marking whether each cell has been visited.
    obstacle_mask : np.ndarray
        Boolean array marking obstacle cells.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        grid_size: Tuple[int, int] = (10, 10),
        obstacle_ratio: float = 0.1,
        max_steps: int = 500,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        assert 0.0 <= obstacle_ratio < 1.0, "obstacle_ratio must be in [0,1)"
        self.grid_size = grid_size
        self.obstacle_ratio = obstacle_ratio
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        # Observation: x, y coordinates, 4 binary sensor readings, coverage fraction
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
        )
        # Actions: 0 = north, 1 = east, 2 = south, 3 = west
        self.action_space = spaces.Discrete(4)

        # Internal state variables
        self.agent_pos: Tuple[int, int] = (0, 0)
        self.coverage_mask: np.ndarray = np.zeros(self.grid_size, dtype=bool)
        self.obstacle_mask: np.ndarray = np.zeros(self.grid_size, dtype=bool)
        self.num_covered: int = 0
        self.total_cells: int = self.grid_size[0] * self.grid_size[1]
        self.step_count: int = 0

    def _reset_grid(self) -> None:
        """Initialise the coverage and obstacle masks and place the agent."""
        rows, cols = self.grid_size
        # Reset masks
        self.coverage_mask[:] = False
        # Randomly place obstacles based on obstacle_ratio
        num_obstacles = int(self.total_cells * self.obstacle_ratio)
        obstacle_indices = self.rng.choice(self.total_cells, num_obstacles, replace=False)
        self.obstacle_mask[:] = False
        for idx in obstacle_indices:
            r = idx // cols
            c = idx % cols
            self.obstacle_mask[r, c] = True
        # Ensure agent start cell is not an obstacle
        free_positions = np.argwhere(~self.obstacle_mask)
        start_idx = self.rng.integers(len(free_positions))
        self.agent_pos = tuple(free_positions[start_idx])
        # Mark starting cell as covered
        self.coverage_mask[self.agent_pos] = True
        self.num_covered = 1
        self.step_count = 0

    def _get_sensor_readings(self, pos: Tuple[int, int]) -> np.ndarray:
        """Return binary sensor readings for N, E, S, W (1 = free, 0 = obstacle/boundary)."""
        x, y = pos
        rows, cols = self.grid_size
        readings = np.zeros(4, dtype=np.float32)
        # North
        if x > 0 and not self.obstacle_mask[x - 1, y]:
            readings[0] = 1.0
        # East
        if y < cols - 1 and not self.obstacle_mask[x, y + 1]:
            readings[1] = 1.0
        # South
        if x < rows - 1 and not self.obstacle_mask[x + 1, y]:
            readings[2] = 1.0
        # West
        if y > 0 and not self.obstacle_mask[x, y - 1]:
            readings[3] = 1.0
        return readings

    def _build_observation(self) -> np.ndarray:
        """Construct the observation vector."""
        x, y = self.agent_pos
        rows, cols = self.grid_size
        # Normalise coordinates to [0,1]
        pos_norm = np.array([x / (rows - 1), y / (cols - 1)], dtype=np.float32)
        sensors = self._get_sensor_readings(self.agent_pos)
        coverage_fraction = np.array([self.num_covered / self.total_cells], dtype=np.float32)
        obs = np.concatenate([pos_norm, sensors, coverage_fraction])
        return obs

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment and return the initial observation.

        Parameters
        ----------
        seed : Optional[int]
            Optional seed to reinitialise the random number generator.
        options : Optional[dict]
            Unused, present for API compatibility.

        Returns
        -------
        obs : np.ndarray
            The initial observation.
        info : dict
            Additional information (empty).
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._reset_grid()
        obs = self._build_observation()
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, Tuple[float, float, float], bool, bool, Dict]:
        """Take a step in the environment.

        Parameters
        ----------
        action : int
            Action index: 0 (north), 1 (east), 2 (south), 3 (west).

        Returns
        -------
        obs : np.ndarray
            Next observation.
        reward_vector : Tuple[float, float, float]
            A 3‑tuple containing the coverage reward, step penalty and collision penalty.
        terminated : bool
            Whether the episode has terminated because all cells are covered.
        truncated : bool
            Whether the episode was truncated due to exceeding max_steps.
        info : dict
            Additional diagnostic information.
        """
        assert self.action_space.contains(action), f"Invalid action {action}"
        self.step_count += 1

        # Compute new position based on action
        x, y = self.agent_pos
        new_x, new_y = x, y
        if action == 0:  # north
            new_x = x - 1
        elif action == 1:  # east
            new_y = y + 1
        elif action == 2:  # south
            new_x = x + 1
        elif action == 3:  # west
            new_y = y - 1
        # Check boundary conditions
        collision = False
        rows, cols = self.grid_size
        if 0 <= new_x < rows and 0 <= new_y < cols and not self.obstacle_mask[new_x, new_y]:
            self.agent_pos = (new_x, new_y)
        else:
            # Attempted move is invalid – collision or boundary
            collision = True
        # Update coverage
        coverage_reward = 0.0
        if not self.coverage_mask[self.agent_pos]:
            self.coverage_mask[self.agent_pos] = True
            self.num_covered += 1
            coverage_reward = 1.0
        step_penalty = -0.1  # small negative reward per step
        collision_penalty = -10.0 if collision else 0.0
        # Determine termination conditions
        terminated = self.num_covered == self.total_cells
        truncated = self.step_count >= self.max_steps
        # Final reward when coverage complete
        if terminated:
            coverage_reward += 10.0
        obs = self._build_observation()
        reward_vector = (coverage_reward, step_penalty, collision_penalty)
        return obs, reward_vector, terminated, truncated, {}


class ScalarisedEnv(gym.Env):
    """Wrapper that converts a vector reward environment into a scalar reward environment.

    Parameters
    ----------
    env : UAVCoverageEnv
        Underlying multi‑objective environment.
    weights : List[float]
        Weight vector used for linear scalarisation.  The length must match
        the length of the reward tuple returned by `env.step()`.

    Notes
    -----
    The weights should sum to 1.0 for interpretability, but this is not
    strictly required.  Negative weights invert the desirability of an
    objective and should be used with caution.
    """

    def __init__(self, env: UAVCoverageEnv, weights: List[float]):
        super().__init__()
        assert len(weights) == 3, "weights must have length 3"
        self.env = env
        self.weights = np.array(weights, dtype=np.float32)
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        return self.env.reset(seed=seed, options=options)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward_vector, terminated, truncated, info = self.env.step(action)
        scalar_reward = float(np.dot(self.weights, reward_vector))
        return obs, scalar_reward, terminated, truncated, info


def train_single_weight(
    weights: List[float],
    total_timesteps: int = 50_000,
    grid_size: Tuple[int, int] = (10, 10),
    obstacle_ratio: float = 0.1,
    max_steps: int = 500,
    seed: Optional[int] = None,
) -> "stable_baselines3.PPO":
    """Train a PPO agent with a specified weight vector.

    Parameters
    ----------
    weights : List[float]
        The weight vector for scalarising the reward.  This dictates the
        relative importance of coverage, path length and collision avoidance.
    total_timesteps : int
        The number of training time steps.  For meaningful learning this
        should be set to a large value (e.g. hundreds of thousands or
        millions) as in Garrido‑Castañeda et al.’s experiments【864997031185730†L880-L883】.
    grid_size : Tuple[int, int]
        Size of the coverage grid.
    obstacle_ratio : float
        Fraction of grid cells that are obstacles.
    max_steps : int
        Maximum episode length.
    seed : Optional[int]
        Seed for reproducibility.

    Returns
    -------
    model : stable_baselines3.PPO
        The trained PPO model.

    Notes
    -----
    This function requires `stable_baselines3` and `torch`.  It will
    raise an ImportError if they are not available.  The returned model
    can be evaluated with the `.predict` method to obtain actions.
    """
    # Import here to avoid requiring stable_baselines3 unless training is invoked
    try:
        from stable_baselines3 import PPO
    except ImportError as e:
        raise ImportError(
            "stable_baselines3 must be installed to train the agent. Use `pip install stable-baselines3`"
        ) from e

    # Create environment and wrapper
    base_env = UAVCoverageEnv(grid_size=grid_size, obstacle_ratio=obstacle_ratio, max_steps=max_steps, seed=seed)
    env = ScalarisedEnv(base_env, weights)
    # Use a simple multilayer perceptron policy
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        seed=seed,
    )
    model.learn(total_timesteps=total_timesteps)
    return model


def evaluate_policy(
    model: "stable_baselines3.PPO",
    env: UAVCoverageEnv,
    weights: List[float],
    n_episodes: int = 5,
) -> Dict[str, float]:
    """Evaluate a trained policy on the multi‑objective environment.

    The evaluation function returns the average of each reward component
    across the episodes as well as the average scalarised reward for the
    given weight vector.

    Parameters
    ----------
    model : stable_baselines3.PPO
        A trained PPO model.
    env : UAVCoverageEnv
        The underlying multi‑objective environment.
    weights : List[float]
        Weight vector used for scalarisation during evaluation.
    n_episodes : int
        Number of episodes to run for evaluation.

    Returns
    -------
    results : Dict[str, float]
        Dictionary with keys ``coverage``, ``steps``, ``collisions``, ``scalar_reward``
        containing average per‑episode values.
    """
    try:
        import torch  # ensure torch is available for model.predict
    except ImportError as e:
        raise ImportError(
            "torch must be installed to evaluate the model. Use `pip install torch`."
        ) from e
    reward_sums = np.zeros(3, dtype=np.float32)
    scalar_sums = 0.0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward_vector, terminated, truncated, info = env.step(int(action))
            reward_sums += np.array(reward_vector, dtype=np.float32)
            scalar_sums += float(np.dot(weights, reward_vector))
    # Average per episode
    reward_sums /= n_episodes
    scalar_sums /= n_episodes
    return {
        "coverage": float(reward_sums[0]),
        "step_penalty": float(reward_sums[1]),
        "collision_penalty": float(reward_sums[2]),
        "scalar_reward": float(scalar_sums),
    }


if __name__ == "__main__":
    """Example usage: train and evaluate policies with different weight vectors."""
    import argparse
    parser = argparse.ArgumentParser(description="Train MORL agent for coverage path planning")
    parser.add_argument("--weights", type=str, default="0.8,0.15,0.05",
                        help="Comma separated weights for (coverage, step penalty, collision penalty)")
    parser.add_argument("--timesteps", type=int, default=10_000,
                        help="Number of training timesteps (increase for better performance)")
    parser.add_argument("--grid", type=str, default="10,10",
                        help="Grid size as rows,cols")
    parser.add_argument("--obstacles", type=float, default=0.1,
                        help="Obstacle ratio (0 ≤ r < 1)")
    parser.add_argument("--max_steps", type=int, default=500,
                        help="Maximum steps per episode")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    args = parser.parse_args()
    weights = [float(x) for x in args.weights.split(",")]
    grid_rows, grid_cols = [int(x) for x in args.grid.split(",")]
    # Train agent
    model = train_single_weight(
        weights=weights,
        total_timesteps=args.timesteps,
        grid_size=(grid_rows, grid_cols),
        obstacle_ratio=args.obstacles,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    # Evaluate agent
    base_env = UAVCoverageEnv(
        grid_size=(grid_rows, grid_cols), obstacle_ratio=args.obstacles, max_steps=args.max_steps, seed=args.seed
    )
    results = evaluate_policy(model, base_env, weights, n_episodes=3)
    print(f"Evaluation results for weights {weights}: {results}")