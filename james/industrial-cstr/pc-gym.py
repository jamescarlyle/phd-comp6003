import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    # =========================
    # Cell 1: imports + setup
    # =========================
    import pcgym
    import numpy as np
    import gymnasium as gym
    from gymnasium import spaces
    import neat
    import os
    import visualize
    import pickle
    import matplotlib.pyplot as plt
    import random

    return neat, np, os, pcgym, pickle, plt, random


@app.cell
def _(np, random):
    SEED = 43
    random.seed(SEED)
    np.random.seed(SEED)
    return (SEED,)


@app.cell
def _(np, os):
    # -------------------------
    # pc-gym CSTR configuration
    # -------------------------
    N_STEPS = 100
    TSIM = 25

    SP = {
        "Ca": [0.85 for _ in range(N_STEPS // 2)] + [0.90 for _ in range(N_STEPS - N_STEPS // 2)]
    }

    OBS_LOW = np.array([0.7, 300.0, 0.8], dtype=np.float32)
    OBS_HIGH = np.array([1.0, 350.0, 0.9], dtype=np.float32)

    ACT_LOW = np.array([295.0], dtype=np.float32)
    ACT_HIGH = np.array([302.0], dtype=np.float32)

    env_params = {
        "N": N_STEPS,
        "tsim": TSIM,
        "SP": SP,
        "o_space": {"low": OBS_LOW, "high": OBS_HIGH},
        "a_space": {"low": ACT_LOW, "high": ACT_HIGH},
        "x0": np.array([0.8, 330.0, 0.8], dtype=np.float32),
        "model": "cstr",
    }

    # -------------------------
    # NEAT hyperparameters
    # -------------------------
    N_GENERATIONS = 100
    EVAL_EPISODES = 5

    DU_WEIGHT = 1e-3
    REWARD_WEIGHT = 0.01

    OUTDIR = "neat_cstr_outputs"
    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR)
    return ACT_HIGH, ACT_LOW, N_GENERATIONS, N_STEPS, OUTDIR, env_params


@app.cell
def _(ACT_HIGH, ACT_LOW, N_STEPS, SEED, env_params, np, pcgym):
    # ==================================
    # Cell 2: helper functions
    # ==================================
    def make_env():
        trial_params = dict(env_params)
        return pcgym.make_env(trial_params)

    def reset_env(env, seed=None):
        try:
            out = env.reset(seed=seed)
        except TypeError:
            out = env.reset()

        if isinstance(out, tuple):
            obs = np.asarray(out[0], dtype=np.float64)
            info = out[1] if len(out) > 1 else {}
        else:
            obs = np.asarray(out, dtype=np.float64)
            info = {}

        return obs, info

    def step_env(env, action):
        out = env.step(np.asarray(action, dtype=np.float32))

        if len(out) == 5:
            obs, rew, done_a, done_b, info = out
            done = bool(done_a or done_b)
        elif len(out) == 4:
            obs, rew, done, info = out
        else:
            raise ValueError(f"Unexpected env.step output length: {len(out)}")

        return np.asarray(obs, dtype=np.float64), float(rew), done, info


    def scale_action(raw_output):
        # NEAT tanh output expected roughly in [-1, 1]
        raw_output = float(np.clip(raw_output, -1.0, 1.0))
        scaled = ACT_LOW + 0.5 * (raw_output + 1.0) * (ACT_HIGH - ACT_LOW)
        return np.asarray([scaled.item()], dtype=np.float32)

    def run_episode(net, seed_offset=0, return_trace=False):
        env = make_env()
        obs, _ = reset_env(env, seed=SEED + seed_offset)

        total_reward = 0.0
        done = False
        step_count = 0

        obs_trace = []
        action_trace = []
        reward_trace = []

        while not done and step_count < N_STEPS:
            # IMPORTANT: do not normalize again
            net_input = np.asarray(obs, dtype=np.float64)

            raw_action = net.activate(net_input)[0]
            action = scale_action(raw_action)

            obs, rew, done, info = step_env(env, action)
            total_reward += rew

            if return_trace:
                obs_trace.append(np.array(obs, copy=True))
                action_trace.append(float(action[0]))
                reward_trace.append(float(rew))

            step_count += 1

        try:
            env.close()
        except Exception:
            pass

        result = {
            "fitness": float(total_reward),
            "total_reward": float(total_reward),
        }

        if return_trace:
            result["trace"] = {
                "obs": np.array(obs_trace),
                "u": np.array(action_trace),
                "r": np.array(reward_trace),
            }

        return result

    return (run_episode,)


@app.cell
def _():
    # ==================================
    # Cell 3: write NEAT config file
    # ==================================
    NEAT_CONFIG = """
    [NEAT]
    fitness_criterion     = max
    fitness_threshold     = -0.01
    pop_size              = 100
    reset_on_extinction   = False
    no_fitness_termination = False


    [DefaultGenome]
    activation_default      = tanh
    activation_mutate_rate  = 0.0
    activation_options      = tanh

    aggregation_default     = sum
    aggregation_mutate_rate = 0.0
    aggregation_options     = sum

    bias_init_mean          = 0.0
    bias_init_stdev         = 1.0
    bias_init_type          = gaussian
    bias_max_value          = 30.0
    bias_min_value          = -30.0
    bias_mutate_power       = 0.5
    bias_mutate_rate        = 0.7
    bias_replace_rate       = 0.1

    compatibility_disjoint_coefficient = 1.0
    compatibility_weight_coefficient   = 0.5

    conn_add_prob           = 0.5
    conn_delete_prob        = 0.3

    enabled_default         = True
    enabled_mutate_rate     = 0.01

    feed_forward            = True
    initial_connection      = full_direct

    node_add_prob           = 0.2
    node_delete_prob        = 0.2

    num_hidden              = 0
    num_inputs              = 3
    num_outputs             = 1

    response_init_mean      = 1.0
    response_init_stdev     = 0.0
    response_init_type      = gaussian
    response_max_value      = 30.0
    response_min_value      = -30.0
    response_mutate_power   = 0.0
    response_mutate_rate    = 0.0
    response_replace_rate   = 0.0

    weight_init_mean        = 0.0
    weight_init_stdev       = 1.5
    weight_init_type        = gaussian
    weight_max_value        = 30
    weight_min_value        = -30
    weight_mutate_power     = 0.5
    weight_mutate_rate      = 0.8
    weight_replace_rate     = 0.1

    single_structural_mutation = False
    structural_mutation_surer  = default

    [DefaultSpeciesSet]
    compatibility_threshold = 3.0

    [DefaultStagnation]
    species_fitness_func = max
    max_stagnation       = 20
    species_elitism      = 2

    [DefaultReproduction]
    elitism            = 2
    survival_threshold = 0.2
    """

    with open('neat_cstr_config.cfg', 'w') as _file:
        _file.write(NEAT_CONFIG)
    return


@app.cell
def _(neat, np, run_episode):
    # ==================================
    # Cell 4: genome evaluation
    # ==================================
    def eval_genomes(genomes, config):
        vals = []
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)

            fitnesses = []
            for ep in range(3):
                res = run_episode(net, seed_offset=1000 * genome_id + ep, return_trace=False)
                fitnesses.append(res["fitness"])

            genome.fitness = float(np.mean(fitnesses))
            vals.append(genome.fitness)

        vals = np.array(vals)
        print("unique fitnesses (6dp):", len(np.unique(np.round(vals, 6))))
        print("min/mean/max:", vals.min(), vals.mean(), vals.max())

    return (eval_genomes,)


@app.cell
def _(N_GENERATIONS, OUTDIR, eval_genomes, neat, pickle):
    # ==================================
    # Cell 5: train NEAT
    # ==================================
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str('neat_cstr_config.cfg'),
    )

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))

    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    checkpointer = neat.Checkpointer(
        generation_interval=10,
        filename_prefix=str(OUTDIR + 'neat-checkpoint-')
    )
    population.add_reporter(checkpointer)

    winner = population.run(eval_genomes, N_GENERATIONS)

    with open(OUTDIR / "winner.pkl", "wb") as _file:
        pickle.dump(winner, _file)

    print("Training complete.")
    print("Best genome:", winner)
    return config, winner


@app.cell
def _(config, neat, run_episode, winner):
    # ==================================
    # Cell 6: evaluate best genome
    # ==================================
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    best_result = run_episode(winner_net, seed_offset=999, return_trace=True)

    print("Winner fitness:", best_result["fitness"])
    print("Winner total reward:", best_result["total_reward"])
    return (best_result,)


@app.cell
def _(best_result, np, plt):
    # ==================================
    # Cell 7: plotting
    # ==================================
    trace = best_result["trace"]
    t = np.arange(len(trace["Ca"]))

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(t, trace["Ca"], label="Ca", linewidth=2)
    axes[0].plot(t, trace["Ca_sp"], "--", label="Ca setpoint", linewidth=2)
    axes[0].set_ylabel("Concentration")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, trace["u"], color="tab:orange", linewidth=2)
    axes[1].set_ylabel("Action")
    axes[1].set_title("Manipulated temperature input")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, trace["r"], color="tab:green", linewidth=2)
    axes[2].set_ylabel("Reward")
    axes[2].set_xlabel("Step")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(OUTDIR, config, neat, pickle, run_episode):
    # ==================================
    # Cell 8: optional reload later
    # ==================================
    with open(OUTDIR / "winner.pkl", "rb") as f:
        loaded_winner = pickle.load(f)

    loaded_net = neat.nn.FeedForwardNetwork.create(loaded_winner, config)
    loaded_result = run_episode(loaded_net, seed_offset=1234, return_trace=False)
    print("Reloaded winner fitness:", loaded_result["fitness"])
    return


if __name__ == "__main__":
    app.run()
