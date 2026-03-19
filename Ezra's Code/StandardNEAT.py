"""
NEAT CartPole Implementation

This implementation uses neat-python and gymnasium.

========================================================================
CartPole-v1 environment:
========================================================================

- At each timestep, the environment returns a state vector containing:
    - Cart position (-1)
    - Cart velocity (-2)
    - Pole angle (-3)
    - Pole angular velocity (-4)
- Given this state, the agent must decide whether to push the cart left (0) or right (1).
- This code returns a policy network that tells the agent how to act based on the state variable
- In this script, the network starts as a simple feedforward network with no hidden layers (4 input neurons, and 2 output neurons).
- NEAT can add hidden layers and connections over generations, via mutations.
- The output neurons represent the action values for left and right.
- We take a majority vote from the output neurons to decide the action.
- In NEAT, a neural network is encoded as a "genome" that specifies the network's structure and weights.

========================================================================

The NEAT algorithm works as follows:
1. Create a population of random genomes (neural networks).
2. For each neural network, evaluate its fitness. Fitness is defined as how long the cart can balance the pole (number of steps before failure).
3. Select the best-performing genomes and mutate them to create a new generation of neural networks.
    - Possible mutations include:
        - Changing connection weights
        - Adding new connections
        - Removing connections
        - Adding new neurons
4. Repeat this process for a specified number of generations.

========================================================================
Execution:
========================================================================

Run the following line in the python terminal (with default parameters):
  python StandardNEAT.py

Optional arguments:
  - --gens: number of generations to evolve (default: 50)
  - --pop: population size (default: 150)
  - --episodes: number of episodes to evaluate each genome (default: 5)
  - --render: whether to render the environment when demoing the winner (default: False)
"""

import os
import argparse
import pickle

import neat
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import gymnasium as gym


def step_env(env, action):
    '''
    env.step(action) returns:
     - obs: the new state after taking the action
     - reward: the reward received for taking the action
     - terminated: returns True when the environment violates certain conditions (e.g., the pole fell)
     - truncated: returns True when the time limit is exceeded (e.g., 500 steps)
     - _info: debugging data

    done returns True when either terminated or truncated is True,
    indicating that the episode has concluded.
    '''
    obs, reward, terminated, truncated, _info = env.step(action)
    done = terminated or truncated
    return obs, reward, done


def eval_genomes(genomes, config, env_id, episodes_per_genome=5, max_steps=500):

    env = gym.make(env_id)

    for _gid, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        episode_returns = []
        for _ in range(episodes_per_genome):
            obs = env.reset()[0]

            total_steps = 0
            for _t in range(max_steps):
                action_values = net.activate(obs) # feed the state into the network to get action values
                action = int(np.argmax(action_values)) # the network will output 2 numbers, we pick the action with the higher output.

                obs, _reward, done = step_env(env, action) # take the action in the environment, and observe the new state and reward
                total_steps += 1
                if done:
                    break

            episode_returns.append(total_steps)

        genome.fitness = float(np.mean(episode_returns))

    env.close()


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def run_neat(env_id="CartPole-v1", gens=50, pop_size=150, episodes=5, seed=None):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    config_path = os.path.join(os.path.dirname(__file__), "config-neat.ini")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    p = neat.Population(config)

    # Print progress:
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(
        lambda genomes, cfg: eval_genomes(genomes, cfg, env_id, episodes_per_genome=episodes),
        gens,
    )

    # Save the winner:
    with open(os.path.join(RESULTS_DIR, "neat_cartpole_winner.pkl"), "wb") as f:
        pickle.dump(winner, f)

    print("\nBest genome:\n", winner)
    return winner, config


def demo_winner(winner, config, env_id="CartPole-v1", episodes=3, render=True, max_steps=500):
    env = gym.make(env_id, render_mode="human")
    net = neat.nn.FeedForwardNetwork.create(winner, config)

    for ep in range(episodes):
        obs = env.reset()[0]
        total_steps = 0

        for _t in range(max_steps):
            action = int(np.argmax(net.activate(obs)))
            obs, _reward, done = step_env(env, action)
            total_steps += 1

            if done:
                break

        print(f"Demo episode {ep+1}: survived {total_steps} steps")

    env.close()


def visualise_winner(winner, config, filename=None):
    if filename is None:
        filename = os.path.join(RESULTS_DIR, "neat_winner_network.png")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    """
    Renders the winning genome as a publication-quality neural network diagram.
    Nodes are arranged in columns by layer (input → hidden → output).
    Edge colour and thickness encodes connection weight.
    Saves a PNG suitable for slide decks.
    """

    # ── Palette ────────────────────────────────────────────────────────────────
    BG        = "#0d1117"
    INPUT_C   = "#58a6ff"   # blue
    HIDDEN_C  = "#3fb950"   # green
    OUTPUT_C  = "#f78166"   # coral
    NODE_EDGE  = "#ffffff"
    LABEL_C   = "#e6edf3"
    TITLE_C   = "#e6edf3"
    ANNOT_C   = "#8b949e"

    INPUT_LABELS  = ["Cart\nPosition", "Cart\nVelocity", "Pole\nAngle", "Pole\nAngular\nVelocity"]
    OUTPUT_LABELS = ["← Left", "Right →"]

    # ── Collect node & connection info from genome ─────────────────────────────
    connections = {k: v for k, v in winner.connections.items() if v.enabled}

    # node keys: inputs are negative, outputs are 0..n_out-1 in neat-python
    input_keys  = config.genome_config.input_keys   # e.g. [-1, -2, -3, -4]
    output_keys = config.genome_config.output_keys  # e.g. [0, 1]
    # winner.nodes only contains output + hidden nodes (not inputs)
    hidden_keys = [k for k in winner.nodes.keys() if k not in output_keys]
    n_total_nodes = len(input_keys) + len(hidden_keys) + len(output_keys)

    # ── Assign (x, y) positions ────────────────────────────────────────────────
    def column_positions(keys, x):
        n = len(keys)
        ys = np.linspace(1, 0, n)
        return {k: (x, y) for k, y in zip(keys, ys)}

    n_cols = 3 if hidden_keys else 2
    pos = {}
    pos.update(column_positions(input_keys,  0.0))
    if hidden_keys:
        pos.update(column_positions(hidden_keys, 0.5))
    pos.update(column_positions(output_keys, 1.0))

    node_colors = {}
    for k in input_keys:  node_colors[k] = INPUT_C
    for k in hidden_keys: node_colors[k] = HIDDEN_C
    for k in output_keys: node_colors[k] = OUTPUT_C

    # ── Weight normalisation for edge colouring ────────────────────────────────
    weights = [c.weight for c in connections.values()]
    if weights:
        wmax = max(abs(w) for w in weights) or 1.0
    else:
        wmax = 1.0

    cmap_pos = plt.cm.Blues
    cmap_neg = plt.cm.Reds

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(-0.12, 1.12)
    ax.axis("off")

    # ── Draw edges ─────────────────────────────────────────────────────────────
    for (src, tgt), conn in connections.items():
        if src not in pos or tgt not in pos:
            continue
        x0, y0 = pos[src]
        x1, y1 = pos[tgt]
        w  = conn.weight
        t  = 0.3 + 2.2 * abs(w) / wmax
        α  = 0.25 + 0.65 * abs(w) / wmax
        c  = cmap_pos(0.4 + 0.5 * abs(w) / wmax) if w >= 0 else cmap_neg(0.4 + 0.5 * abs(w) / wmax)
        ax.plot([x0, x1], [y0, y1], color=c, linewidth=t, alpha=α, zorder=1)
        # Weight label at 10% of the way from source to target
        mx, my = x0 + 0.1 * (x1 - x0), y0 + 0.1 * (y1 - y0)
        ax.text(mx, my, f"{w:+.2f}", ha="center", va="center",
                fontsize=9, color=LABEL_C, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.15", facecolor=BG, edgecolor="none", alpha=0.75),
                zorder=2)

    # ── Draw nodes ─────────────────────────────────────────────────────────────
    NODE_R = 0.045
    for key, (x, y) in pos.items():
        circle = plt.Circle((x, y), NODE_R, color=node_colors[key],
                             zorder=3, linewidth=1.8, ec=NODE_EDGE)
        ax.add_patch(circle)
        # node id label inside
        ax.text(x, y, str(key), ha="center", va="center",
                fontsize=14, color="white", fontweight="bold", zorder=4,
                fontfamily="monospace")

    # ── Input / output annotations ─────────────────────────────────────────────
    for i, k in enumerate(input_keys):
        x, y = pos[k]
        lbl = INPUT_LABELS[i] if i < len(INPUT_LABELS) else f"In {i}"
        ax.text(x - NODE_R - 0.015, y, lbl, ha="right", va="center",
                fontsize=12, color=LABEL_C, fontfamily="sans-serif")

    for i, k in enumerate(output_keys):
        x, y = pos[k]
        lbl = OUTPUT_LABELS[i] if i < len(OUTPUT_LABELS) else f"Out {i}"
        ax.text(x + NODE_R + 0.015, y, lbl, ha="left", va="center",
                fontsize=12, color=LABEL_C, fontfamily="sans-serif")

    # ── Column headers ─────────────────────────────────────────────────────────
    headers = {"Inputs": 0.0, "Outputs": 1.0}
    if hidden_keys:
        headers["Hidden"] = 0.5
    for lbl, x in headers.items():
        ax.text(x, 1.09, lbl, ha="center", va="bottom",
                fontsize=14, color=ANNOT_C, fontfamily="sans-serif",
                fontstyle="italic")

    # ── Title & subtitle ───────────────────────────────────────────────────────
    n_nodes = n_total_nodes
    n_conns = len(connections)
    fitness  = winner.fitness if winner.fitness is not None else 0

    ax.text(0.5, -0.09,
            f"NEAT Winner  ·  {n_nodes} nodes  ·  {n_conns} active connections  ·  fitness {fitness:.0f}",
            ha="center", va="top", fontsize=13, color=ANNOT_C,
            fontfamily="monospace", transform=ax.transData)

    ax.set_title("Winning Neural Network — CartPole-v1",
                 fontsize=19, color=TITLE_C, pad=14,
                 fontfamily="sans-serif", fontweight="bold")

    # ── Legend ─────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color=INPUT_C,  label="Input node"),
        mpatches.Patch(color=HIDDEN_C, label="Hidden node"),
        mpatches.Patch(color=OUTPUT_C, label="Output node"),
        mpatches.Patch(color=cmap_pos(0.75), label="Positive weight"),
        mpatches.Patch(color=cmap_neg(0.75), label="Negative weight"),
    ]
    leg = ax.legend(handles=legend_handles, loc="center right",
                    framealpha=0.15, labelcolor=LABEL_C,
                    facecolor=BG, edgecolor="#30363d",
                    fontsize=12, handlelength=1.2)

    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"\nNetwork visualisation saved → {filename}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--gens", type=int, default=50)
    parser.add_argument("--pop", type=int, default=150)
    parser.add_argument("--episodes", type=int, default=5, help="episodes per genome evaluation")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--render", action="store_true", help="render winner after training")
    args = parser.parse_args()

    winner, config = run_neat(
        env_id=args.env,
        gens=args.gens,
        pop_size=args.pop,
        episodes=args.episodes,
        seed=args.seed,
    )

    if args.render:
        demo_winner(winner, config, env_id=args.env, episodes=3, render=True)

    visualise_winner(winner, config)


if __name__ == "__main__":
    main()