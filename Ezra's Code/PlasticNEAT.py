"""
NEAT-Neuroplasticity Rule CartPole Implementation (extended for use on LunarLander aswell)

This file extends the standard NEAT algorithm to include per-synapse
neuroplasticity rules as attributes of each connection. This means that
NEAT treats the neuroplasticity parameters (learning rate and rule type)
as mutatable/evolvable properties. No extra package is used to do this.

========================================================================

Outer loop (NEAT):
    Per-synapse evolution:
        - initial_weight  (s⁰)  : initial weight of synapse (nothing new)
        - learning_rate   (η)   : FloatAttribute, range [0.0, 0.1]
        - rule            (str) : StringAttribute, one of the six rule names

Inner loop (plasticity):
    At each timestep, each synapse weight is updated according to its rule and learning rate:
        Δw = η · rule(s, α, β)
        w  ← clamp(w + Δw, -MAX_W, MAX_W)
    where α = pre-synaptic activation, β = post-synaptic activation.

========================================================================

Plasticity Rules:
    fixed           : 0
    hebbian         : α · β
    oja_plus        : β · (α − s · β)
    oja_minus       : (1−β) · (α − s · (1−β))
    correlation     : α · (2β − 1)
    anticorrelation : −α · (2β − 1)

========================================================================
Execution:
========================================================================

    Run the following line in the python terminal (with default parameters):
        python PlasticNEAT.py

    Optional arguments:
        - --gens: number of generations to evolve (default: 50)
        - --pop: population size (default: 150)
        - --episodes: number of episodes to evaluate each genome (default: 5)
        - --seed: random seed for reproducibility (default: None)
        - --rule: allows specifying a forced plasticity rule (default: None - uses complete set of rules)
        - --task: defines the environment (default: cartpole, options: cartpole, lunarlander)
        - --nonstat: form that non-stationarity takes (default: None - stationary, options: sudden, gradual, cyclic, degradation)
        - --magnitude: strength of non-stationarity (default: 0.5, range: 0 to 1)
        - --shift-interval: encodes the frequency of parameter shifts for sudden non-stationarity (default: 200 steps)
        - --render: whether to render the environment when demoing the winner (default: False)

========================================================================

Abbreviations/acronyms:
    - ns: non-stationary
    - cv: current value
    - ps: phase shift

========================================================================
CartPole-v1 environment:
========================================================================

- At each timestep, the environment returns a state vector containing:
    - Cart position             (-1)
    - Cart velocity             (-2)
    - Pole angle                (-3)
    - Pole angular velocity     (-4)
- Given this state, the agent must decide whether to push the cart left (0) or right (1).
- This code returns a policy network that tells the agent how to act based on the state variable
- In this script, the network starts as a simple feedforward network with no hidden layers (4 input neurons, and 2 output neurons).
- NEAT can add hidden layers and connections over generations, via mutations.
- The output neurons represent the action values for left and right.
- We take a majority vote from the output neurons to decide the action.
- In NEAT, a neural network is encoded as a "genome" that specifies the network's structure and weights.

========================================================================
LunarLander-v3 environment:
========================================================================

- At each timestep, the environment returns a state vector containing:
    - x position (0 = centre)       (0)
    - y position (0 = ground)       (1)
    - x velocity                    (2)
    - y velocity                    (3)
    - angle (0 = upright)           (4)
    - angular velocity              (5)
    - left leg contact (bool)       (6)
    - right leg contact (bool)      (7)
"""

import os
import argparse
import pickle

import neat
from neat.genes import DefaultConnectionGene
from neat.genome import DefaultGenome
from neat.attributes import FloatAttribute, StringAttribute

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import gymnasium as gym
from dataclasses import dataclass
from typing import Optional


@dataclass
class ShiftDetails:
    """Encodes the details of the shifts in enrivonment."""
    shift_type:     str   = "sudden"   # "sudden" | "gradual" | "cyclic" | "degradation"
    shift_interval: int   = 200        # frequency of parameter shifts (sudden mode only)
    drift_rate:     float = 0.005      # fractional change per step (gradual/degradation)
    cycle_period:   int   = 500        # time period for cyclic shifts
    magnitude:      float = 0.5        # 0 = no change in parameters, 1 = can vary in the full range either side of initial value
    shift_seed: Optional[int] = None


class NonStationaryCartPole(gym.Wrapper):
    """
    CartPole-v1 with non-stationarity, implemented as a Gymnasium Wrapper.

    Four types of non-stationarity (encoded via ShiftDetails.shift_type):
        - sudden      — parameters jump to a new random value every 'shift_interval' steps.
        - gradual     — parameters drift slowly each step by ±drift_rate of their range.
        - cyclic      — parameters oscillate sinusoidally with period 'cycle_period'.
        - degradation — parameters drift uniformly toward their "worst-case" value.

    The gravity and force_mag parameters are varied by default.
    You can pass a list of parameters to vary:
        params_to_vary=['gravity','masscart','masspole','length','force_mag']

    Parameter ranges (min, max, default):
        gravity   : (5.0,  15.0,  9.8 )
        masscart  : (0.5,   2.0,  1.0 )
        masspole  : (0.05,  0.3,  0.1 )
        length    : (0.25,  1.0,  0.5 )
        force_mag : (5.0,  20.0, 10.0 )
    """

    PARAM_RANGES = {
        "gravity":   (5.0,  15.0,  9.8),
        "masscart":  (0.5,   2.0,  1.0),
        "masspole":  (0.05,  0.3,  0.1),
        "length":    (0.25,  1.0,  0.5),
        "force_mag": (5.0,  20.0, 10.0),
    }

    def __init__(
        self,
        nsType: Optional[ShiftDetails] = None,
        params_to_vary: Optional[list] = None,
    ):
        env = gym.make("CartPole-v1")
        super().__init__(env)
        self.nsType       = nsType or ShiftDetails()
        self.params_to_vary = params_to_vary or ["gravity", "force_mag"]
        self.nsSeed            = np.random.RandomState(self.nsType.shift_seed)
        self._step_count    = 0

        # Initialise parameters:
        self._current = {p: self.PARAM_RANGES[p][2] for p in self.params_to_vary}
        self._initial = dict(self._current)
        self._apply_params()


    def _apply_params(self):
        uw = self.env.unwrapped
        for p, v in self._current.items():
            lo, hi, _ = self.PARAM_RANGES[p]
            v = float(np.clip(v, lo, hi))
            self._current[p] = v
            setattr(uw, p, v)
        if hasattr(uw, "total_mass"):
            uw.total_mass        = uw.masscart + uw.masspole
            uw.polemass_length   = uw.masspole * uw.length

    def _update_params(self):
        """Update parameters according to the shift details."""
        ns = self.nsType
        t  = self._step_count

        for p in self.params_to_vary:
            lo, hi, default = self.PARAM_RANGES[p]
            range = hi - lo
            cv = self._current[p]

            if ns.shift_type == "sudden":
                if t > 0 and t % ns.shift_interval == 0: # if t is a multiple of shift_interval (and not 0)
                    self._current[p] = self.nsSeed.uniform(
                        default - ns.magnitude * (lo - default),
                        default + ns.magnitude * (hi - default)
                    )

            elif ns.shift_type == "gradual":
                self._current[p] = self.nsSeed.uniform(
                        default - ns.drift_rate * ns.magnitude * (lo - default),
                        default + ns.drift_rate * ns.magnitude * (hi - default)
                    )

            elif ns.shift_type == "cyclic":
                a = ns.magnitude * range / 2 # amplitude
                ps = (ns.cycle_period / (2 * np.pi)) * np.arcsin(default / a) # phase shift to start at default value
                self._current[p] = a * np.sin(2 * np.pi * (t - ps) / ns.cycle_period)

            elif ns.shift_type == "degradation":
                # "Worst-case" values: max gravity, max masscart, max masspole, max length, min force_mag
                if p == "force_mag":
                    direction = -1.0 # degrade toward minimum
                else:
                    direction = 1.0  # degrade toward maximum
                self._current[p] = cv + direction * ns.drift_rate * ns.magnitude * range

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



class NonStationaryLunarLander(gym.Wrapper):
    """
    LunarLander-v3 with non-stationarity, implemented as a Gymnasium Wrapper.

    Parameter ranges (min, max, default):
        wind_power        :  (0.0,  20.0,  0.0)
        turbulence_power  :  (0.0,   2.0,  0.0)
        gravity           : (-20.0, -0.5, -10.0)

    Types of non-stationarity (same as before):
        sudden      — jumps at shift_interval steps
        gradual     — random gradual drift
        cyclic      — sinusoidal oscillation
        degradation — uniform drift toward "worst-case" values
    """

    PARAM_RANGES = {
        "wind_power":       (0.0,  20.0,  0.0),
        "turbulence_power": (0.0,   2.0,  0.0),
        "gravity":          (-20.0, -0.5, -10.0),
    }

    def __init__(
        self,
        nsType: Optional[ShiftDetails] = None,
        params_to_vary: Optional[list] = None,
    ):
        env = gym.make("LunarLander-v3", enable_wind=True,
                       wind_power=0.0, turbulence_power=0.0)
        super().__init__(env)
        self.nsType = nsType or ShiftDetails()
        self.params_to_vary = params_to_vary or ["wind_power", "turbulence_power"]
        self.nsSeed = np.random.RandomState(self.nsType.shift_seed)
        self._step_count = 0

        self._current = {p: self.PARAM_RANGES[p][2] for p in self.params_to_vary}
        self._apply_params()

    def _apply_params(self):
        uw = self.env.unwrapped
        for p, v in self._current.items():
            lo, hi, _ = self.PARAM_RANGES[p]
            v = float(np.clip(v, lo, hi))
            self._current[p] = v
            setattr(uw, p, v)

    def _update_params(self):
        ns = self.nsType
        t  = self._step_count

        for p in self.params_to_vary:
            lo, hi, default = self.PARAM_RANGES[p]
            range = hi - lo
            cv = self._current[p]

            if ns.shift_type == "sudden":
                if t > 0 and t % ns.shift_interval == 0:
                    self._current[p] = self.nsSeed.uniform(
                        default - ns.magnitude * (lo - default),
                        default + ns.magnitude * (hi - default)
                    )

            elif ns.shift_type == "gradual":
                self._current[p] = self.nsSeed.uniform(
                        default - ns.drift_rate * ns.magnitude * (lo - default),
                        default + ns.drift_rate * ns.magnitude * (hi - default)
                    )

            elif ns.shift_type == "cyclic":
                a = ns.magnitude * range / 2
                ps = (ns.cycle_period / (2 * np.pi)) * np.arcsin(default / a)
                self._current[p] = a * np.sin(2 * np.pi * (t - ps) / ns.cycle_period)

            elif ns.shift_type == "degradation":
                # "Worst-case" values: max wind_power, max turbulence_power, max gravity
                if p == "gravity": # gravity is measured in negatively
                    direction = -1.0
                else:
                    direction = 1.0 
                self._current[p] = cv + direction * ns.drift_rate * ns.magnitude * range

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
    env_type: str,
    seed: Optional[int] = None,
    magnitude: float = 0.5,
    shift_interval: int = 200,
) -> gym.Env:
    """
    This function creates the environment for us to deploy NEAT on. It is used in the eval_genomes function.

    env_type options:
        "stationary"  — standard CartPole-v1 (control condition)
        "sudden"      — sudden parameter jumps every shift_interval steps
        "gradual"     — slow continuous drift
        "cyclic"      — sinusoidal oscillation
        "degradation" — monotonic decay toward worst-case parameters

    magnitude       : 0 = no change from default physics, 1 = full parameter range
    shift_interval  : steps between sudden jumps (sudden mode only)
    """
    if env_type == "stationary":
        return gym.make("CartPole-v1")
    elif env_type == "lunar_stationary":
        return gym.make("LunarLander-v3")
    elif env_type in ("sudden", "gradual", "cyclic", "degradation"):
        return NonStationaryCartPole(
            nsType=ShiftDetails(
                shift_type=env_type,
                shift_seed=seed,
                magnitude=magnitude,
                shift_interval=shift_interval,
            ),
            params_to_vary=["gravity", "force_mag"],
        )
    elif env_type in ("lunar_sudden", "lunar_gradual", "lunar_cyclic", "lunar_degradation"):
        shift_type = env_type.split("_", 1)[1]
        return NonStationaryLunarLander(
            nsType=ShiftDetails(
                shift_type=shift_type,
                shift_seed=seed,
                magnitude=magnitude,
                shift_interval=shift_interval,
            ),
            params_to_vary=["wind_power", "turbulence_power"],
        )
    else:
        raise ValueError(
            f"Unknown env_type {env_type!r}. "
            "CartPole: stationary | sudden | gradual | cyclic | degradation. "
            "LunarLander: lunar_stationary | lunar_sudden | lunar_gradual | "
            "lunar_cyclic | lunar_degradation"
        )


RULES = ["fixed", "hebbian", "oja_plus", "oja_minus", "correlation", "anticorrelation"]
MAX_W = 5.0   # maximum weight change

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def apply_rule(rule: str, s: float, alpha: float, beta: float) -> float:
    """
    This function computes the delta from the plasticity rules.

    Inputs:
     - rule  : name of the plasticity rule
     - s     : current synaptic weight
     - alpha : pre-synaptic activation
     - beta  : post-synaptic activation

    Output:
     - Δw (still needs to be multiplied by η)
    """
    if rule == "fixed":
        return 0.0
    elif rule == "hebbian":
        return alpha * beta
    elif rule == "oja_plus":
        return beta * (alpha - s * beta)
    elif rule == "oja_minus":
        b = 1.0 - beta
        return b * (alpha - s * b)
    elif rule == "correlation":
        return alpha * (2.0 * beta - 1.0)
    elif rule == "anticorrelation":
        return -alpha * (2.0 * beta - 1.0)
    else:
        raise ValueError(f"Unknown plasticity rule: {rule!r}")


class PlasticConnectionGene(DefaultConnectionGene):
    """
    This class is created to be NEAT gene that can evolve two extra attributes:
     - learning_rate (η): FloatAttribute
     - rule: StringAttribute

    neat-python's built-in mutation function (mutate()) handles this automatically.
    """

    _gene_attributes = DefaultConnectionGene._gene_attributes + [
        FloatAttribute("learning_rate"),
        StringAttribute("rule"),
    ]

    # No __init__ needed — neat-python initialises the extra attributes automatically using the config values.


class PlasticGenome(DefaultGenome):
    """
    This class encodes a genome that uses PlasticConnectionGene instead of DefaultConnectionGene.
    (Allows learning_rate and rule per synapse to evolve)
    """

    @classmethod
    def parse_config(cls, param_dict):
        from neat.genome import DefaultGenomeConfig
        from neat.genes import DefaultNodeGene
        param_dict["node_gene_type"]       = DefaultNodeGene
        param_dict["connection_gene_type"] = PlasticConnectionGene
        return DefaultGenomeConfig(param_dict, cls.__name__)


class PlasticFeedForwardNetwork:
    """
    This class encodes a feedforward network that applies the plasticity rules at each timestep.
    """

    def __init__(self, inputs, outputs, node_evals, init_weights, rules, etas):
        """
        Inputs:
         - inputs       : list of input node keys
         - outputs      : list of output node keys
         - node_evals   : list of (node, activation function, aggregate function, bias, response, connection details)
         - init_weights : dict mapping for initial weight
         - rules        : dict mapping for rule name string
         - etas         : dict mapping for learning rate η
        """
        self.input_nodes  = inputs
        self.output_nodes = outputs
        self.node_evals   = node_evals

        self.weights   = dict(init_weights)
        self.init_weights = dict(init_weights)
        self.rules     = rules
        self.etas      = etas

        self.activations: dict = {}

    def reset(self):
        """This function resets the weights to their initial value (used inbetween episodes)"""
        self.weights = dict(self.init_weights)
        self.activations = {}

    def activate(self, inputs: list) -> list:
        """
        Run a forward pass with the plasticity rule updates.

        Inputs:
         - inputs : dict list of nodes

        Output:
         - Activations as a list.
        """
        for k, v in zip(self.input_nodes, inputs):
            self.activations[k] = v

        for node, act_fn, agg_fn, bias, response, conns in self.node_evals:
            node_inputs = []
            for conn_key, src_key in conns:
                w = self.weights.get(conn_key, 0.0)
                node_inputs.append(w * self.activations.get(src_key, 0.0))
            s = agg_fn(node_inputs)
            self.activations[node] = act_fn(bias + response * s)

        # Plasticity update:
        for node, _act_fn, _agg_fn, _bias, _response, conns in self.node_evals:
            beta = self.activations.get(node, 0.0)
            for conn_key, src_key in conns:
                alpha = self.activations.get(src_key, 0.0)
                s     = self.weights[conn_key]
                rule  = self.rules[conn_key]
                eta   = self.etas[conn_key]

                delta = eta * apply_rule(rule, s, alpha, beta)
                self.weights[conn_key] = max(-MAX_W, min(MAX_W, s + delta)) # "clamp" the weight change

        return [self.activations[k] for k in self.output_nodes]

    @classmethod
    def create(cls, genome: PlasticGenome, config, forced_rule: str = None):
        """
        Builds a PlasticFeedForwardNetwork from a PlasticGenome.

        Inputs:
         - genome      : PlasticGenome instance
         - config      : neat.Config
         - forced_rule : only used if we force a fixed rule over the whole network
        """
        from neat.nn.feed_forward import feed_forward_layers

        connections = [cg for cg in genome.connections.values() if cg.enabled]

        layers, _required = feed_forward_layers(
            config.genome_config.input_keys,
            config.genome_config.output_keys,
            [(c.key[0], c.key[1]) for c in connections],
        )

        node_evals = []
        for layer in layers:
            for node in layer:
                if node not in genome.nodes:
                    continue

                node_inputs = []
                for conn in connections:
                    if conn.key[1] == node:
                        node_inputs.append((conn.key, conn.key[0]))

                ng     = genome.nodes[node]
                act_fn = config.genome_config.activation_defs.get(ng.activation)
                agg_fn = config.genome_config.aggregation_function_defs.get(ng.aggregation)
                node_evals.append((node, act_fn, agg_fn, ng.bias, ng.response, node_inputs))

        init_weights = {}
        rules        = {}
        etas         = {}
        for conn in connections:
            k              = conn.key
            init_weights[k] = conn.weight
            rules[k]        = forced_rule if forced_rule else conn.rule
            etas[k]         = conn.learning_rate

        return cls(
            config.genome_config.input_keys,
            config.genome_config.output_keys,
            node_evals,
            init_weights,
            rules,
            etas,
        )


def step_env(env, action):
    obs, reward, terminated, truncated, _info = env.step(action)
    return obs, reward, terminated or truncated


def eval_genomes(
    genomes, config,
    env_type="stationary",
    episodes_per_genome=5,
    max_steps=500,
    forced_rule=None,
    seed=None,
    magnitude=0.5,
    shift_interval=200,
):
    """
    This function is used to evaluate the fitness of a genome over 'episodes_per_genome' episodes.

    Inputs:
     - env_type       : "stationary", "sudden", "gradual", "cyclic", or "degradation"
     - forced_rule    : if set/inputted, this overrides each genome's evolved rule with a global rule (useful for ablation?).
     - magnitude      : strength/magnitude of non-stationarity
     - shift_interval : frequency of sudden shifts in parameters
    """
    is_lunar = env_type.startswith("lunar")
    env = make_env(env_type, seed=seed, magnitude=magnitude, shift_interval=shift_interval)

    for _gid, genome in genomes:
        net = PlasticFeedForwardNetwork.create(genome, config, forced_rule=forced_rule)

        episode_returns = []
        for _ in range(episodes_per_genome):
            net.reset()
            obs = env.reset()[0]

            total_reward = 0.0
            total_steps  = 0
            for _t in range(max_steps):
                action_values = net.activate(obs)
                action        = int(np.argmax(action_values))
                obs, reward, done = step_env(env, action)

                if is_lunar:
                    # Reward calculation:
                    # Due to slow convergence, we add extra reward terms to try and help convergence.

                    x, y, vx, vy, angle, omega, leg_l, leg_r = obs

                    # Penalise distance from centre and height from ground:
                    proximity   = -0.05 * (abs(x) + abs(y))
                    # Penalise high velocities:
                    gentleness  = -0.02 * (abs(vx) + abs(vy))
                    # Penalise big angles:
                    uprightness = -0.05 * abs(angle)
                    # Reward leg contact:
                    contact     =  0.10 * (leg_l + leg_r)
                    reward += proximity + gentleness + uprightness + contact

                total_reward += reward
                total_steps  += 1
                if done:
                    break

            # In CartPole fitness is calculated as the number of steps survived
            # In LunarLander fitness is calculated as a cumulative reward
            episode_returns.append(total_reward if is_lunar else total_steps)

        genome.fitness = float(np.mean(episode_returns))

    env.close()


def build_config(config_path: str, forced_rule: str = None, num_inputs: int = None, num_outputs: int = None) -> neat.Config:
    """
    The two extra attributes need these config keys:

        learning_rate_init_mean    = 0.05
        learning_rate_init_stdev   = 0.01
        learning_rate_max_value    = 0.1
        learning_rate_min_value    = 0.0
        learning_rate_mutate_power = 0.01
        learning_rate_mutate_rate  = 0.5
        learning_rate_replace_rate = 0.1
        learning_rate_init_type    = gaussian

        rule_options               = fixed hebbian oja_plus oja_minus correlation anticorrelation
        rule_mutate_rate           = 0.1
        rule_replace_rate          = 0.1
    """
    import configparser

    raw = configparser.ConfigParser()
    raw.read(config_path)

    # Our genome is called PlasticGenome, so we copy the attributes from DefaultGenome into PlasticGenome and add the plasticity keys:
    default_items = dict(raw.items("DefaultGenome"))
    raw.add_section("PlasticGenome")
    for key, val in default_items.items():
        raw.set("PlasticGenome", key, val)

    section = "PlasticGenome"

    if num_inputs is not None:
        raw.set(section, "num_inputs",  str(num_inputs))
    if num_outputs is not None:
        raw.set(section, "num_outputs", str(num_outputs))

    raw.set(section, "learning_rate_init_type",    "gaussian")
    raw.set(section, "learning_rate_init_mean",    "0.05")
    raw.set(section, "learning_rate_init_stdev",   "0.02")
    raw.set(section, "learning_rate_max_value",    "0.1")
    raw.set(section, "learning_rate_min_value",    "0.0")
    raw.set(section, "learning_rate_mutate_power", "0.01")
    raw.set(section, "learning_rate_mutate_rate",  "0.5")
    raw.set(section, "learning_rate_replace_rate", "0.05")

    rule_options = forced_rule if forced_rule else " ".join(RULES)
    raw.set(section, "rule_default",      "random")
    raw.set(section, "rule_options",       rule_options)
    raw.set(section, "rule_mutate_rate",   "0.0" if forced_rule else "0.1")
    raw.set(section, "rule_replace_rate",  "0.0" if forced_rule else "0.1")

    import tempfile
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".ini", delete=False
    )
    raw.write(tmp)
    tmp.flush()
    tmp.close()

    config = neat.Config(
        PlasticGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        tmp.name,
    )
    os.unlink(tmp.name)
    return config


def run_neat(
    env_type="stationary",
    gens=50,
    episodes=5,
    seed=None,
    forced_rule=None,
    magnitude=0.5,
    shift_interval=200,
):
    is_lunar = env_type.startswith("lunar")
    # If LunarLander, use "config-neat-lunar.ini".
    config_name = "config-neat-lunar.ini" if is_lunar else "config-neat.ini"
    config_path = os.path.join(os.path.dirname(__file__), config_name)
    config = build_config(config_path, forced_rule=forced_rule)

    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    rule_label = forced_rule if forced_rule else "evolved (all rules)"
    print(f"\n{'='*60}")
    print(f"  Plastic NEAT  |  rule: {rule_label}  |  env: {env_type}")
    print(f"{'='*60}\n")

    winner = p.run(
        lambda genomes, cfg: eval_genomes(
            genomes, cfg,
            env_type=env_type,
            episodes_per_genome=episodes,
            forced_rule=forced_rule,
            seed=seed,
            magnitude=magnitude,
            shift_interval=shift_interval,
        ),
        gens,
    )

    rule_tag = f"_{forced_rule}" if forced_rule else ""
    env_tag  = f"_{env_type}" if env_type != "stationary" else ""
    save_name = os.path.join(RESULTS_DIR, f"plastic_neat_winner{rule_tag}{env_tag}.pkl")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(save_name, "wb") as f:
        pickle.dump(winner, f)

    print(f"\nBest genome:\n{winner}")
    return winner, config, stats


def demo_winner(winner, config, env_type="stationary", episodes=3, forced_rule=None):
    env = gym.make("CartPole-v1", render_mode="human")
    net = PlasticFeedForwardNetwork.create(winner, config, forced_rule=forced_rule)

    for ep in range(episodes):
        net.reset()
        obs = env.reset()[0]
        total_steps = 0

        for _t in range(500):
            action = int(np.argmax(net.activate(obs)))
            obs, _reward, done = step_env(env, action)
            total_steps += 1
            if done:
                break

        print(f"Demo episode {ep+1}: survived {total_steps} steps")

    env.close()


def visualise_winner(winner, config, forced_rule=None, env_type="stationary", filename=None):
    if filename is None:
        filename = os.path.join(RESULTS_DIR, "plastic_neat_winner_network.png")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    """
    Renders the winning PlasticGenome as a neural network diagram.
    Each edge is annotated with: weight | rule | η
    Colour encodes sign of weight; thickness encodes magnitude.
    """
    BG        = "#0d1117"
    INPUT_C   = "#58a6ff"
    HIDDEN_C  = "#3fb950"
    OUTPUT_C  = "#f78166"
    NODE_EDGE = "#ffffff"
    LABEL_C   = "#e6edf3"
    ANNOT_C   = "#8b949e"

    RULE_SHORT = {
        "fixed":           "FX",
        "hebbian":         "HB",
        "oja_plus":        "O+",
        "oja_minus":       "O−",
        "correlation":     "CR",
        "anticorrelation": "AC",
    }

    INPUT_LABELS  = ["Cart\nPos", "Cart\nVel", "Pole\nAngle", "Pole\nAngVel"]
    OUTPUT_LABELS = ["← Left", "Right →"]

    connections = {k: v for k, v in winner.connections.items() if v.enabled}
    input_keys  = config.genome_config.input_keys
    output_keys = config.genome_config.output_keys
    hidden_keys = [k for k in winner.nodes if k not in output_keys]

    def col_pos(keys, x):
        ys = np.linspace(1, 0, len(keys))
        return {k: (x, y) for k, y in zip(keys, ys)}

    pos = {}
    pos.update(col_pos(input_keys, 0.0))
    if hidden_keys:
        pos.update(col_pos(hidden_keys, 0.5))
    pos.update(col_pos(output_keys, 1.0))

    node_colors = {k: INPUT_C  for k in input_keys}
    node_colors.update({k: HIDDEN_C for k in hidden_keys})
    node_colors.update({k: OUTPUT_C for k in output_keys})

    weights = [c.weight for c in connections.values()]
    wmax    = max((abs(w) for w in weights), default=1.0) or 1.0

    cmap_pos = plt.cm.Blues
    cmap_neg = plt.cm.Reds

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.15, 1.15)
    ax.axis("off")

    for (src, tgt), conn in connections.items():
        if src not in pos or tgt not in pos:
            continue
        x0, y0 = pos[src]
        x1, y1 = pos[tgt]
        w   = conn.weight
        t   = 0.3 + 2.2 * abs(w) / wmax
        α   = 0.25 + 0.65 * abs(w) / wmax
        c   = (cmap_pos if w >= 0 else cmap_neg)(0.4 + 0.5 * abs(w) / wmax)
        ax.plot([x0, x1], [y0, y1], color=c, linewidth=t, alpha=α, zorder=1)

        # Edge label: weight | rule abbreviation | η
        rule_str = forced_rule if forced_rule else conn.rule
        eta_val  = conn.learning_rate
        label    = f"{w:+.2f} | {RULE_SHORT.get(rule_str, rule_str)} | η={eta_val:.3f}"
        mx = x0 + 0.12 * (x1 - x0)
        my = y0 + 0.12 * (y1 - y0)
        ax.text(mx, my, label, ha="center", va="center",
                fontsize=7, color=LABEL_C, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.12", facecolor=BG,
                          edgecolor="none", alpha=0.8),
                zorder=2)

    NODE_R = 0.045
    for key, (x, y) in pos.items():
        ax.add_patch(plt.Circle((x, y), NODE_R, color=node_colors[key],
                                zorder=3, linewidth=1.8, ec=NODE_EDGE))
        ax.text(x, y, str(key), ha="center", va="center",
                fontsize=13, color="white", fontweight="bold",
                fontfamily="monospace", zorder=4)

    for i, k in enumerate(input_keys):
        x, y = pos[k]
        ax.text(x - NODE_R - 0.015, y,
                INPUT_LABELS[i] if i < len(INPUT_LABELS) else f"In {i}",
                ha="right", va="center", fontsize=11, color=LABEL_C)

    for i, k in enumerate(output_keys):
        x, y = pos[k]
        ax.text(x + NODE_R + 0.015, y,
                OUTPUT_LABELS[i] if i < len(OUTPUT_LABELS) else f"Out {i}",
                ha="left", va="center", fontsize=11, color=LABEL_C)

    headers = {"Inputs": 0.0, "Outputs": 1.0}
    if hidden_keys:
        headers["Hidden"] = 0.5
    for lbl, x in headers.items():
        ax.text(x, 1.11, lbl, ha="center", fontsize=13, color=ANNOT_C,
                fontstyle="italic")

    n_nodes = len(input_keys) + len(hidden_keys) + len(output_keys)
    n_conns = len(connections)
    fitness  = winner.fitness or 0
    rule_label = forced_rule if forced_rule else "evolved (all rules)"
    ax.text(0.5, -0.12,
            f"Plastic NEAT  ·  rule: {rule_label}  ·  {n_nodes} nodes  "
            f"·  {n_conns} connections  ·  fitness {fitness:.0f}",
            ha="center", fontsize=11, color=ANNOT_C,
            fontfamily="monospace", transform=ax.transData)

    task_label = "LunarLander" if "lunar" in env_type else "CartPole"
    ax.set_title(f"Winning Plastic Neural Network — {task_label} ({env_type})",
                 fontsize=18, color=LABEL_C, pad=12, fontweight="bold")

    legend_handles = [
        mpatches.Patch(color=INPUT_C,         label="Input node"),
        mpatches.Patch(color=HIDDEN_C,        label="Hidden node"),
        mpatches.Patch(color=OUTPUT_C,        label="Output node"),
        mpatches.Patch(color=cmap_pos(0.75),  label="Positive weight"),
        mpatches.Patch(color=cmap_neg(0.75),  label="Negative weight"),
    ]
    ax.legend(handles=legend_handles, loc="center right",
              framealpha=0.15, labelcolor=LABEL_C,
              facecolor=BG, edgecolor="#30363d", fontsize=11)

    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Network visualisation saved → {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Plastic NEAT on CartPole with native plasticity genes."
    )
    parser.add_argument("--gens",     type=int,  default=50)
    parser.add_argument("--pop",      type=int,  default=150)
    parser.add_argument("--episodes", type=int,  default=5,
                        help="Episodes per genome evaluation")
    parser.add_argument("--seed",     type=int,  default=None)
    parser.add_argument("--rule",     type=str,  default=None,
                        choices=RULES,
                        help="Force all synapses to use this rule (ablation). "
                             "Omit to let NEAT evolve rule per synapse.")
    parser.add_argument("--task", type=str, default="cartpole",
                        choices=["cartpole", "lunarlander"],
                        help="Control task to evolve on (default: cartpole).")
    parser.add_argument("--nonstat",  type=str,  default=None,
                        choices=["sudden", "gradual", "cyclic", "degradation"],
                        help="Non-stationarity type. Omit for stationary baseline.")
    parser.add_argument("--magnitude", type=float, default=0.5,
                        help="Non-stationarity strength: 0=none, 1=full param range (default 0.5).")
    parser.add_argument("--shift-interval", type=int, default=200,
                        help="Steps between sudden parameter jumps (default 200).")
    parser.add_argument("--render",   action="store_true",
                        help="Render winner after training")
    args = parser.parse_args()

    prefix   = "lunar_" if args.task == "lunarlander" else ""
    base     = args.nonstat if args.nonstat else "stationary"
    env_type = f"{prefix}{base}" if args.task == "lunarlander" else base

    winner, config, stats = run_neat(
        env_type=env_type,
        gens=args.gens,
        episodes=args.episodes,
        seed=args.seed,
        forced_rule=args.rule,
        magnitude=args.magnitude,
        shift_interval=args.shift_interval,
    )

    if args.render:
        demo_winner(winner, config, env_type=env_type, forced_rule=args.rule)

    rule_tag = f"_{args.rule}" if args.rule else ""
    env_tag  = f"_{env_type}" if env_type != "stationary" else ""
    visualise_winner(winner, config,
                     forced_rule=args.rule,
                     env_type=env_type,
                     filename=os.path.join(RESULTS_DIR, f"plastic_neat_winner_network{rule_tag}{env_tag}.png"))


if __name__ == "__main__":
    main()