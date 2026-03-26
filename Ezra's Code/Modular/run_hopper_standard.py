"""
run_hopper_standard.py
======================
Standard NEAT (no plasticity) on Hopper-v5.

Used to establish a fitness ceiling baseline against which PlasticNEAT
results can be judged. Uses neat-python's built-in DefaultGenome and
FeedForwardNetwork — no plasticity genes, no inner weight update loop.

Parallelism is identical to run_hopper.py: genome evaluations are
distributed across all available CPU cores via multiprocessing.

Usage:
    python run_hopper_standard.py
    python run_hopper_standard.py --gens 200 --pop 200 --episodes 3
    python run_hopper_standard.py --seed 42
"""

import argparse
import os
import sys
import pickle
import multiprocessing as mp
from typing import Optional

import neat
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CONFIG_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config-neat-hopper.ini")

MAX_STEPS = 1000

NODE_NAMES = {
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
    0:   "thigh torque",
    1:   "leg torque",
    2:   "foot torque",
}

# ---------------------------------------------------------------------------
# Single-genome evaluator (top-level for multiprocessing pickling)
# ---------------------------------------------------------------------------

def _eval_single_genome(args):
    """Evaluate one genome; returns fitness. Must be top-level for pickle."""
    genome, config, episodes, seed = args

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = gym.make("Hopper-v5")

    episode_returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        total_reward = 0.0

        for _t in range(MAX_STEPS):
            action_values = net.activate(list(obs))
            action = np.clip(action_values, -1.0, 1.0)

            obs, reward, terminated, truncated, _ = env.step(action)

            # Same reward shaping as PlasticNEAT for a fair comparison
            x_vel   = obs[5]
            t_angle = obs[1]
            reward += 0.5 * max(x_vel, 0.0) - 0.05 * abs(t_angle)

            total_reward += reward
            if terminated or truncated:
                break

        episode_returns.append(total_reward)

    env.close()
    return float(np.mean(episode_returns))


# ---------------------------------------------------------------------------
# Parallel eval_genomes
# ---------------------------------------------------------------------------

def eval_genomes(genomes, config, episodes=3, seed=None):
    """Distribute genome evaluation across all available cores."""
    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", mp.cpu_count()))
    n_workers = min(n_workers, len(genomes))

    eval_args = [
        (genome, config, episodes, seed)
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


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_neat(gens=200, episodes=3, seed=None):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH,
    )

    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    print(f"\n{'='*60}")
    print(f"  Standard NEAT  |  Hopper-v5  |  stationary baseline")
    print(f"{'='*60}\n")

    winner = p.run(
        lambda genomes, cfg: eval_genomes(genomes, cfg, episodes=episodes, seed=seed),
        gens,
    )

    save_path = os.path.join(RESULTS_DIR, "standard_neat_winner_hopper.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(winner, f)
    print(f"\nWinner saved → {save_path}")

    return winner, config, stats


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualise_winner(winner, config, filename=None):
    if filename is None:
        filename = os.path.join(RESULTS_DIR, "standard_neat_winner_network_hopper.png")
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    BG       = "#0d1117"
    INPUT_C  = "#58a6ff"
    HIDDEN_C = "#3fb950"
    OUTPUT_C = "#f78166"
    LABEL_C  = "#e6edf3"
    ANNOT_C  = "#8b949e"

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
        w = conn.weight
        t = 0.3 + 2.2 * abs(w) / wmax
        α = 0.25 + 0.65 * abs(w) / wmax
        c = (cmap_pos if w >= 0 else cmap_neg)(0.4 + 0.5 * abs(w) / wmax)
        ax.plot([x0, x1], [y0, y1], color=c, linewidth=t, alpha=α, zorder=1)
        mx = x0 + 0.12 * (x1 - x0)
        my = y0 + 0.12 * (y1 - y0)
        ax.text(mx, my, f"{w:+.2f}", ha="center", va="center",
                fontsize=7, color=LABEL_C, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.12", facecolor=BG,
                          edgecolor="none", alpha=0.8), zorder=2)

    NODE_R = 0.045
    for key, (x, y) in pos.items():
        ax.add_patch(plt.Circle((x, y), NODE_R, color=node_colors[key],
                                zorder=3, linewidth=1.8, ec="#ffffff"))
        ax.text(x, y, str(key), ha="center", va="center",
                fontsize=13, color="white", fontweight="bold",
                fontfamily="monospace", zorder=4)

    for k in input_keys:
        x, y = pos[k]
        ax.text(x - NODE_R - 0.015, y, NODE_NAMES.get(k, f"In {k}"),
                ha="right", va="center", fontsize=9, color=LABEL_C)

    for k in output_keys:
        x, y = pos[k]
        ax.text(x + NODE_R + 0.015, y, NODE_NAMES.get(k, f"Out {k}"),
                ha="left", va="center", fontsize=9, color=LABEL_C)

    headers = {"Inputs": 0.0, "Outputs": 1.0}
    if hidden_keys:
        headers["Hidden"] = 0.5
    for lbl, x in headers.items():
        ax.text(x, 1.11, lbl, ha="center", fontsize=13, color=ANNOT_C,
                fontstyle="italic")

    n_nodes = len(input_keys) + len(hidden_keys) + len(output_keys)
    n_conns = len(connections)
    fitness = winner.fitness or 0
    ax.text(0.5, -0.12,
            f"Standard NEAT  ·  {n_nodes} nodes  ·  {n_conns} connections  ·  fitness {fitness:.0f}",
            ha="center", fontsize=11, color=ANNOT_C,
            fontfamily="monospace", transform=ax.transData)

    ax.set_title("Winning Standard Neural Network — Hopper-v5 (stationary baseline)",
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Standard NEAT (no plasticity) on Hopper-v5 — baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gens",     type=int, default=200)
    parser.add_argument("--pop",      type=int, default=None,
                        help="Override population size from config.")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed",     type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.pop is not None:
        import configparser, tempfile
        global CONFIG_PATH
        raw = configparser.ConfigParser()
        raw.read(CONFIG_PATH)
        raw.set("NEAT", "pop_size", str(args.pop))
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False)
        raw.write(tmp); tmp.flush(); tmp.close()
        CONFIG_PATH = tmp.name

    winner, config, stats = run_neat(
        gens=args.gens,
        episodes=args.episodes,
        seed=args.seed,
    )

    visualise_winner(winner, config)


if __name__ == "__main__":
    main()