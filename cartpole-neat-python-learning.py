#!/usr/bin/env python3

import argparse
import pickle
import random
from pathlib import Path

import gymnasium as gym
import neat
import numpy as np
from neat.attributes import FloatAttribute
from neat.genes import DefaultConnectionGene, DefaultNodeGene
from neat.genome import DefaultGenome, DefaultGenomeConfig


CONFIG_TEXT = """
[NEAT]
fitness_criterion     = max
fitness_threshold     = 500
pop_size              = 120
reset_on_extinction   = False
no_fitness_termination = False

[PlasticGenome]
num_inputs              = 4
num_outputs             = 1
num_hidden              = 0
feed_forward            = True
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5
conn_add_prob           = 0.30
conn_delete_prob        = 0.15
node_add_prob           = 0.20
node_delete_prob        = 0.10
single_structural_mutation = False
structural_mutation_surer = default
initial_connection      = full_direct

activation_default      = tanh
activation_mutate_rate  = 0.0
activation_options      = tanh

aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum

bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1

response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0

enabled_default         = True
enabled_mutate_rate     = 0.01

weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 8.0
weight_min_value        = -8.0
weight_mutate_power     = 0.4
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1

plastic_eta_init_mean       = 0.01
plastic_eta_init_stdev      = 0.01
plastic_eta_max_value       = 0.20
plastic_eta_min_value       = -0.20
plastic_eta_mutate_power    = 0.01
plastic_eta_mutate_rate     = 0.5
plastic_eta_replace_rate    = 0.1

plastic_decay_init_mean     = 0.0005
plastic_decay_init_stdev    = 0.0005
plastic_decay_max_value     = 0.05
plastic_decay_min_value     = 0.0
plastic_decay_mutate_power  = 0.0005
plastic_decay_mutate_rate   = 0.4
plastic_decay_replace_rate  = 0.1

plastic_scale_init_mean     = 1.0
plastic_scale_init_stdev    = 0.2
plastic_scale_max_value     = 3.0
plastic_scale_min_value     = -3.0
plastic_scale_mutate_power  = 0.1
plastic_scale_mutate_rate   = 0.3
plastic_scale_replace_rate  = 0.05

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func    = max
max_stagnation          = 20
species_elitism         = 2

[DefaultReproduction]
elitism                 = 2
survival_threshold      = 0.2
min_species_size        = 2
"""


class PlasticConnectionGene(DefaultConnectionGene):
    _gene_attributes = DefaultConnectionGene._gene_attributes + [
        FloatAttribute("plastic_eta"),
        FloatAttribute("plastic_decay"),
        FloatAttribute("plastic_scale"),
    ]


class PlasticGenome(DefaultGenome):
    @classmethod
    def parse_config(cls, param_dict):
        param_dict["node_gene_type"] = DefaultNodeGene
        param_dict["connection_gene_type"] = PlasticConnectionGene
        return DefaultGenomeConfig(param_dict)


class PlasticEdgeState:
    def __init__(self, gene):
        self.base_weight = float(gene.weight)
        self.live_weight = float(gene.weight)
        self.eta = float(gene.plastic_eta)
        self.decay = max(0.0, float(gene.plastic_decay))
        self.scale = float(gene.plastic_scale)
        self.pre_mean = 0.0
        self.post_mean = 0.0

    def reset(self):
        self.live_weight = self.base_weight
        self.pre_mean = 0.0
        self.post_mean = 0.0

    def apply(self, rule, pre, post):
        if rule == "none":
            return

        if rule == "hebbian":
            dw = self.eta * self.scale * pre * post
        elif rule == "anti_hebbian":
            dw = -self.eta * self.scale * pre * post
        elif rule == "oja":
            dw = self.eta * self.scale * post * (pre - self.live_weight * post)
        elif rule == "covariance":
            alpha = 0.05
            self.pre_mean = (1.0 - alpha) * self.pre_mean + alpha * pre
            self.post_mean = (1.0 - alpha) * self.post_mean + alpha * post
            dw = self.eta * self.scale * (pre - self.pre_mean) * (post - self.post_mean)
        else:
            raise ValueError(f"Unknown learning rule: {rule}")

        self.live_weight += dw
        self.live_weight -= self.decay * self.live_weight
        self.live_weight = float(np.clip(self.live_weight, -8.0, 8.0))


class PlasticFeedForwardNetwork:
    def __init__(self, input_nodes, output_nodes, node_evals, edge_states):
        self.input_nodes = list(input_nodes)
        self.output_nodes = list(output_nodes)
        self.node_evals = node_evals
        self.edge_states = edge_states
        self.values = {}
        self.reset()

    @classmethod
    def create(cls, genome, config):
        base_net = neat.nn.FeedForwardNetwork.create(genome, config)

        edge_states = {}
        node_evals = []

        for node, act_func, agg_func, bias, response, links in base_net.node_evals:
            plastic_links = []

            for inode, _weight in links:
                conn_key = (inode, node)
                cg = genome.connections[conn_key]

                if conn_key not in edge_states:
                    edge_states[conn_key] = PlasticEdgeState(cg)

                plastic_links.append((inode, edge_states[conn_key]))

            node_evals.append(
                (node, act_func, agg_func, bias, response, plastic_links)
            )
    
        return cls(
            base_net.input_nodes,
            base_net.output_nodes,
            node_evals,
            edge_states,
        )

    def reset(self):
        self.values = {k: 0.0 for k in self.input_nodes + self.output_nodes}
        for node, _, _, _, _, _ in self.node_evals:
            self.values[node] = 0.0
        for edge_state in self.edge_states.values():
            edge_state.reset()

    def activate(self, inputs, rule="oja", adapt=True):
        if len(inputs) != len(self.input_nodes):
            raise RuntimeError(f"Expected {len(self.input_nodes)} inputs, got {len(inputs)}")

        for key, value in zip(self.input_nodes, inputs):
            self.values[key] = float(value)

        for node, act_func, agg_func, bias, response, incoming in self.node_evals:
            node_inputs = []
            pres = []

            for inode, edge_state in incoming:
                pre = self.values[inode]
                node_inputs.append(pre * edge_state.live_weight)
                pres.append((pre, edge_state))

            s = agg_func(node_inputs) if node_inputs else 0.0
            post = act_func(bias + response * s)
            self.values[node] = post

            if adapt:
                for pre, edge_state in pres:
                    edge_state.apply(rule, pre, post)

        return [self.values[i] for i in self.output_nodes]


def write_config(path="plastic_cartpole.cfg"):
    Path(path).write_text(CONFIG_TEXT.strip() + "\n")
    return path


def make_env(render_mode=None, max_steps=500):
    env = gym.make("CartPole-v1", render_mode=render_mode)
    if hasattr(env, "_max_episode_steps"):
        env._max_episode_steps = max_steps
    if getattr(env, "spec", None) is not None:
        env.spec.max_episode_steps = max_steps
    return env


def set_pole_length(env, step_idx, max_steps, start_len=0.5, end_len=1.5):
    frac = min(1.0, step_idx / max(1, max_steps - 1))
    internal_length = start_len + frac * (end_len - start_len)
    env.unwrapped.length = internal_length
    env.unwrapped.polemass_length = env.unwrapped.masspole * env.unwrapped.length
    return internal_length


def normalize_obs(obs):
    x, x_dot, theta, theta_dot = obs
    return np.array(
        [
            x / 2.4,
            np.tanh(x_dot / 2.0),
            theta / 0.2095,
            np.tanh(theta_dot / 2.0),
        ],
        dtype=np.float32,
    )


def run_episode(net, env, rule, seed=None, max_steps=500, render=False):
    obs, _ = env.reset(seed=seed)
    net.reset()
    total_reward = 0.0
    final_length = env.unwrapped.length

    if render:
        env.render()

    for t in range(max_steps):
        final_length = set_pole_length(env, t, max_steps)
        output = net.activate(normalize_obs(obs), rule=rule, adapt=(rule != "none"))[0]
        action = 1 if output > 0.0 else 0
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if render:
            env.render()

        if terminated or truncated:
            break

    return total_reward, final_length


class Evaluator:
    def __init__(self, rule, episodes_per_genome=3, max_steps=500):
        self.rule = rule
        self.episodes_per_genome = episodes_per_genome
        self.max_steps = max_steps

    def __call__(self, genomes, config):
        env = make_env(render_mode=None, max_steps=self.max_steps)
        seeds = [101, 202, 303, 404, 505]

        for genome_id, genome in genomes:
            net = PlasticFeedForwardNetwork.create(genome, config)
            scores = []

            for i in range(self.episodes_per_genome):
                score, _ = run_episode(
                    net,
                    env,
                    rule=self.rule,
                    seed=seeds[i % len(seeds)] + 10000 * genome_id,
                    max_steps=self.max_steps,
                    render=False,
                )
                scores.append(score)

            genome.fitness = float(np.mean(scores))

        env.close()


def visualize_winner(winner, config, rule, episodes=3, max_steps=500):
    env = make_env(render_mode="human", max_steps=max_steps)
    net = PlasticFeedForwardNetwork.create(winner, config)

    print(f"\nVisualising best genome with rule={rule!r}")
    for ep in range(episodes):
        score, final_length = run_episode(
            net,
            env,
            rule=rule,
            seed=1000 + ep,
            max_steps=max_steps,
            render=True,
        )
        print(f"visual episode {ep + 1}: reward={score:.1f}, final_internal_length={final_length:.3f}")

    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rule",
        choices=["none", "hebbian", "anti_hebbian", "oja", "covariance"],
        default="oja",
        help="Plasticity rule used during the episode.",
    )
    parser.add_argument("--generations", type=int, default=60)
    parser.add_argument("--episodes-per-genome", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--visual-episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--visual-only", action="store_true")
    parser.add_argument("--load", type=str, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    config_path = write_config()
    config = neat.Config(
        PlasticGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    if args.visual_only:
        if not args.load:
            raise ValueError("--visual-only requires --load winner_file.pkl")
        with open(args.load, "rb") as f:
            winner = pickle.load(f)
        visualize_winner(
            winner,
            config,
            rule=args.rule,
            episodes=args.visual_episodes,
            max_steps=args.max_steps,
        )
        return

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    population.add_reporter(
        neat.Checkpointer(10, filename_prefix=f"chk-{args.rule}-")
    )

    evaluator = Evaluator(
        rule=args.rule,
        episodes_per_genome=args.episodes_per_genome,
        max_steps=args.max_steps,
    )

    winner = population.run(evaluator, args.generations)

    print("\nBest genome:\n")
    print(winner)

    with open(f"winner_{args.rule}.pkl", "wb") as f:
        pickle.dump(winner, f)

    with open(f"winner_{args.rule}.txt", "w") as f:
        f.write(str(winner))

    visualize_winner(
        winner,
        config,
        rule=args.rule,
        episodes=args.visual_episodes,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()