"""
Creates Hopper-v5 environment

Types of non-stationarity:
    - stationary  — standard Hopper
    - sudden      — parameters jump to a new random value every 'shift_interval' steps.
    - gradual     — parameters drift slowly each step by ±drift_rate of their range.
    - cyclic      — parameters oscillate sinusoidally with period 'cycle_period'.
    - degradation — parameters drift uniformly toward their "worst-case" value.

Parameter ranges (min, max, default):
    gravity          : (-20.0, -0.5, -9.81)
    floor_friction   : (0.1,    2.0,  0.8 )
    torso_mass_scale : (0.5,    2.0,  1.0 )
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import gymnasium as gym

_HERE = os.path.dirname(os.path.abspath(__file__))

ENV_CONFIG = {
    "config_path": os.path.join(_HERE, "config-neat-hopper.ini"),
    "num_inputs":  11,
    "num_outputs": 3,
    "max_steps":   1000,
    "fitness_source": "reward",
    "env_name":    "Hopper-v5",
}

NODE_NAMES = {
    # Inputs
    -1:  "z (height)",
    -2:  "torso angle",
    -3:  "thigh jnt",
    -4:  "leg jnt",
    -5:  "foot jnt",
    -6:  "x vel",
    -7:  "z vel",
    -8:  "torso ang vel",
    -9:  "thigh ang vel",
    -10: "leg ang vel",
    -11: "foot ang vel",
    # Outputs
    0:   "thigh torque",
    1:   "leg torque",
    2:   "foot torque",
}

@dataclass
class ShiftDetails:
    """Encodes the type and strength of environment non-stationarity."""
    shift_type:     str   = "sudden"
    shift_interval: int   = 200
    drift_rate:     float = 0.005
    cycle_period:   int   = 500
    magnitude:      float = 0.5
    shift_seed:     Optional[int] = None


class NonStationaryHopper(gym.Wrapper):
    """
    Hopper with non-stationarity.

    Parameter ranges (min, max, default):
        gravity          : (-20.0, -0.5, -9.81)
        floor_friction   : (0.1,    2.0,  0.8 )
        torso_mass_scale : (0.5,    2.0,  1.0 )
    """

    PARAM_RANGES = {
        "gravity":          (-20.0, -0.5, -9.81),
        "floor_friction":   (0.1,    2.0,  0.8),
        "torso_mass_scale": (0.5,    2.0,  1.0),
    }

    def __init__(
        self,
        nsType: Optional[ShiftDetails] = None,
        params_to_vary: Optional[list] = None,
    ):
        env = gym.make("Hopper-v5")
        super().__init__(env)

        self.nsType         = nsType or ShiftDetails()
        self.params_to_vary = params_to_vary or ["gravity", "floor_friction"]
        self.nsSeed         = np.random.RandomState(self.nsType.shift_seed)
        self._step_count    = 0

        model = self.env.unwrapped.model
        self._floor_geom_idx  = self._find_geom(model,  "floor")
        self._torso_body_idx  = self._find_body(model,  "torso")
        self._base_torso_mass = (
            float(model.body_mass[self._torso_body_idx])
            if self._torso_body_idx is not None else 1.0
        )

        self._current = {p: self.PARAM_RANGES[p][2] for p in self.params_to_vary}
        self._apply_params()

    @staticmethod
    def _find_geom(model, name: str) -> Optional[int]:
        try:
            import mujoco
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        except Exception:
            try:
                names = model.geom("floor")
                return int(model.geom(name).id)
            except Exception:
                return None

    @staticmethod
    def _find_body(model, name: str) -> Optional[int]:
        try:
            return int(model.body(name).id)
        except Exception:
            return None

    def _apply_params(self):
        model = self.env.unwrapped.model
        for p, v in self._current.items():
            lo, hi, _ = self.PARAM_RANGES[p]
            v = float(np.clip(v, lo, hi))
            self._current[p] = v

            if p == "gravity":
                model.opt.gravity[2] = v

            elif p == "floor_friction" and self._floor_geom_idx is not None:
                model.geom_friction[self._floor_geom_idx, 0] = v

            elif p == "torso_mass_scale" and self._torso_body_idx is not None:
                model.body_mass[self._torso_body_idx] = self._base_torso_mass * v

    def _update_params(self):
        ns = self.nsType
        t  = self._step_count

        for p in self.params_to_vary:
            lo, hi, default = self.PARAM_RANGES[p]
            p_range = hi - lo
            cv      = self._current[p]

            if ns.shift_type == "sudden":
                if t > 0 and t % ns.shift_interval == 0:
                    lo_shifted = default + ns.magnitude * (lo - default)
                    hi_shifted = default + ns.magnitude * (hi - default)
                    self._current[p] = self.nsSeed.uniform(
                        min(lo_shifted, hi_shifted),
                        max(lo_shifted, hi_shifted),
                    )

            elif ns.shift_type == "gradual":
                lo_shifted = default + ns.drift_rate * ns.magnitude * (lo - default)
                hi_shifted = default + ns.drift_rate * ns.magnitude * (hi - default)
                self._current[p] = self.nsSeed.uniform(
                    min(lo_shifted, hi_shifted),
                    max(lo_shifted, hi_shifted),
                )

            elif ns.shift_type == "cyclic":
                a  = ns.magnitude * p_range / 2.0
                if a > 1e-8:
                    frac = np.clip((default - lo) / p_range * 2.0 - 1.0, -1.0, 1.0)
                    ps   = (ns.cycle_period / (2.0 * np.pi)) * np.arcsin(frac)
                else:
                    ps = 0.0
                centre = (lo + hi) / 2.0
                self._current[p] = centre + a * np.sin(
                    2.0 * np.pi * (t - ps) / ns.cycle_period
                )

            elif ns.shift_type == "degradation":
                # Worst-case: high gravity magnitude, low friction, heavy torso
                if p == "gravity":
                    direction = -1.0   # toward more negative (stronger gravity)
                elif p == "floor_friction":
                    direction = -1.0   # toward lower friction (slippier)
                else:
                    direction = 1.0    # torso_mass_scale → heavier
                self._current[p] = cv + direction * ns.drift_rate * ns.magnitude * p_range

            else:
                raise ValueError(f"Unknown shift_type: {ns.shift_type!r}")

        self._apply_params()

    def reset(self, **kwargs):
        self._step_count = 0
        self._current = {p: self.PARAM_RANGES[p][2] for p in self.params_to_vary}
        self._apply_params()
        return self.env.reset(**kwargs)

    def step(self, action):
        self._step_count += 1
        self._update_params()
        return self.env.step(action)


def make_env(
    env_type: str = "stationary",
    seed: Optional[int] = None,
    magnitude: float = 0.5,
    shift_interval: int = 200,
) -> gym.Env:
    if env_type == "stationary":
        return gym.make("Hopper-v5")
    elif env_type in ("sudden", "gradual", "cyclic", "degradation"):
        return NonStationaryHopper(
            nsType=ShiftDetails(
                shift_type=env_type,
                shift_seed=seed,
                magnitude=magnitude,
                shift_interval=shift_interval,
            ),
            params_to_vary=["gravity", "floor_friction"],
        )
    else:
        raise ValueError(
            f"Unknown env_type {env_type!r} for Hopper. "
            "Valid: stationary | sudden | gradual | cyclic | degradation"
        )


def eval_genomes(
    genomes,
    config,
    *,
    env_type: str = "stationary",
    episodes_per_genome: int = 3,
    max_steps: int = 1000,
    forced_rule: Optional[str] = None,
    seed: Optional[int] = None,
    magnitude: float = 0.5,
    shift_interval: int = 200,
):
    """
    Evaluate each on Hopper.

    Fitness = mean cumulative reward across episodes.

    Action:
        Hopper expects a 3D vector in [-1, 1].
        The three outputs are clipped to this range.

    Reward:
        Hopper's reward includes an alive bonus (1 per step) and a
        forward-velocity term, but the alive bonus can trap evolution
        into making hopper stand still. We add extra rewards:

        1. Forward-progress bonus: += 0.5 * x_velocity (obs[5]).
        2. Uprightness penalty: -= 0.05 * |torso_angle| (obs[1]).
    """
    from plastic_neat_core import PlasticFeedForwardNetwork
    import multiprocessing as mp

    # Iridis set-up:
    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", mp.cpu_count()))
    n_workers = min(n_workers, len(genomes))

    eval_args = [
        (genome, config, env_type, episodes_per_genome, max_steps,
         forced_rule, seed, magnitude, shift_interval)
        for _gid, genome in genomes
    ]

    if n_workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            fitnesses = pool.map(_eval_single_genome, eval_args)
    else:
        fitnesses = [_eval_single_genome(a) for a in eval_args]

    for (_gid, genome), fitness in zip(genomes, fitnesses):
        genome.fitness = fitness


def _eval_single_genome(args):
    """
    Evaluates one genome (rather than them all) and returns fitness.
    This is for "multiprocessing" purposes.
    """
    (genome, config, env_type, episodes_per_genome, max_steps,
     forced_rule, seed, magnitude, shift_interval) = args

    from plastic_neat_core import PlasticFeedForwardNetwork

    env = make_env(env_type, seed=seed, magnitude=magnitude,
                   shift_interval=shift_interval)
    net = PlasticFeedForwardNetwork.create(genome, config, forced_rule=forced_rule)

    episode_returns = []
    for _ in range(episodes_per_genome):
        net.reset()
        obs, _ = env.reset()
        total_reward = 0.0

        for _t in range(max_steps):
            action_values = net.activate(list(obs))
            action = np.clip(action_values, -1.0, 1.0)

            obs, reward, terminated, truncated, _ = env.step(action)

            # obs: [z, torso_angle, thigh_jnt, leg_jnt, foot_jnt,
            #       x_vel, z_vel, torso_ang_vel, thigh_av, leg_av, foot_av]
            x_vel   = obs[5]
            t_angle = obs[1]

            forward_bonus     =  0.5  * max(x_vel, 0.0)
            uprightness_bonus = -0.05 * abs(t_angle)
            reward += forward_bonus + uprightness_bonus

            total_reward += reward
            if terminated or truncated:
                break

        episode_returns.append(total_reward)

    env.close()
    return float(np.mean(episode_returns))