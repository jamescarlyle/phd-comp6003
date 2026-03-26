"""
run_hopper.py
=============
Entry point for running PlasticNEAT on Hopper-v5.

Usage examples:
    python run_hopper.py
    python run_hopper.py --gens 200 --pop 200 --episodes 3
    python run_hopper.py --nonstat sudden --magnitude 0.4
    python run_hopper.py --rule hebbian --nonstat gradual
    python run_hopper.py --render
"""

import argparse
import os
import sys

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import env_hopper
import plastic_neat_core as core


def parse_args():
    parser = argparse.ArgumentParser(
        description="PlasticNEAT on Hopper-v5 (MuJoCo).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gens",     type=int,   default=200,
                        help="Number of NEAT generations.")
    parser.add_argument("--pop",      type=int,   default=None,
                        help="Override population size from config.")
    parser.add_argument("--episodes", type=int,   default=3,
                        help="Evaluation episodes per genome per generation.")
    parser.add_argument("--seed",     type=int,   default=None,
                        help="RNG seed for reproducibility.")
    parser.add_argument("--rule",     type=str,   default=None,
                        choices=core.RULES,
                        help="Force all synapses to use one plasticity rule "
                             "(ablation). Omit to evolve rules freely.")
    parser.add_argument("--nonstat",  type=str,   default=None,
                        choices=["sudden", "gradual", "cyclic", "degradation"],
                        help="Non-stationarity type. Omit for stationary baseline.")
    parser.add_argument("--magnitude", type=float, default=0.5,
                        help="Non-stationarity strength: 0=none, 1=full range.")
    parser.add_argument("--shift-interval", type=int, default=200,
                        help="Steps between sudden parameter jumps.")
    parser.add_argument("--render",   action="store_true",
                        help="Render the winner after training.")
    return parser.parse_args()


def main():
    args = parse_args()

    env_type = args.nonstat if args.nonstat else "stationary"

    # Optionally override population size from config
    if args.pop is not None:
        import configparser
        cfg_path = env_hopper.ENV_CONFIG["config_path"]
        raw = configparser.ConfigParser()
        raw.read(cfg_path)
        raw.set("NEAT", "pop_size", str(args.pop))
        import tempfile
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False)
        raw.write(tmp)
        tmp.flush()
        tmp.close()
        env_hopper.ENV_CONFIG["config_path"] = tmp.name

    winner, config, stats = core.run_neat(
        env_module=env_hopper,
        env_type=env_type,
        gens=args.gens,
        episodes=args.episodes,
        seed=args.seed,
        forced_rule=args.rule,
        magnitude=args.magnitude,
        shift_interval=args.shift_interval,
    )

    # Visualise the winning network
    rule_tag = f"_{args.rule}" if args.rule else ""
    env_tag  = f"_{env_type}" if env_type != "stationary" else ""
    seed_tag = f"_s{args.seed}" if args.seed is not None else ""
    net_path = os.path.join(
        core.RESULTS_DIR,
        f"plastic_neat_winner_network_hopper{rule_tag}{env_tag}{seed_tag}.png",
    )
    core.visualise_winner(
        winner, config,
        forced_rule=args.rule,
        env_type=env_type,
        env_name=env_hopper.ENV_CONFIG["env_name"],
        node_names=env_hopper.NODE_NAMES,
        filename=net_path,
    )

    # Render winner in the environment
    if args.render:
        import gymnasium as gym
        import numpy as np
        from plastic_neat_core import PlasticFeedForwardNetwork

        render_env = gym.make("Hopper-v5", render_mode="human")
        net = PlasticFeedForwardNetwork.create(winner, config,
                                               forced_rule=args.rule)
        max_steps = env_hopper.ENV_CONFIG["max_steps"]

        print("\nRendering winner — close the window to exit.")
        for ep in range(3):
            net.reset()
            obs, _ = render_env.reset()
            total_reward = 0.0
            for _ in range(max_steps):
                action = np.clip(net.activate(list(obs)), -1.0, 1.0)
                obs, r, terminated, truncated, _ = render_env.step(action)
                total_reward += r
                if terminated or truncated:
                    obs, _ = render_env.reset()
                    break
            print(f"  Episode {ep+1}: reward = {total_reward:.1f}")
        render_env.close()


if __name__ == "__main__":
    main()
