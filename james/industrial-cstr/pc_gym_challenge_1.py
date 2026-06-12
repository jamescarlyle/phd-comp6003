# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo",
#   "numpy",
#   "pandas",
#   "matplotlib",
#   "plotly",
#   "neat-python",
#   "pcgym",
# ]
# ///

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    import tempfile
    import numpy as np
    import pandas as pd
    import neat
    import plotly.graph_objects as go
    import plotly.subplots as sp
    from neat.nn import FeedForwardNetwork
    import pcgym

    return FeedForwardNetwork, go, neat, np, os, pcgym, pd, sp, tempfile


@app.cell
def _(np):
    N_STEPS = 100
    TSIM = 25

    SP = {"Ca": [0.85] * (N_STEPS // 2) + [0.90] * (N_STEPS - N_STEPS // 2)}
    A_SPACE = {"low": np.array([295.0]), "high": np.array([302.0])}
    O_SPACE = {"low": np.array([0.7, 300.0, 0.8]), "high": np.array([1.0, 350.0, 0.9])}

    ENV_PARAMS = {
        "N": N_STEPS,
        "tsim": TSIM,
        "SP": SP,
        "o_space": O_SPACE,
        "a_space": A_SPACE,
        "x0": np.array([0.8, 330.0, 0.8]),
        "model": "cstr",
    }
    return (ENV_PARAMS,)


@app.cell
def _():
    CONFIG_TEXT = """
    [NEAT]
    fitness_criterion     = max
    fitness_threshold     = 0.0
    pop_size              = 80
    reset_on_extinction   = False
    no_fitness_termination = False

    [DefaultGenome]
    num_inputs            = 3
    num_outputs           = 1
    num_hidden            = 0
    feed_forward          = True
    initial_connection    = full_direct

    activation_default    = tanh
    activation_options    = tanh
    activation_mutate_rate = 0.0

    aggregation_default   = sum
    aggregation_options   = sum
    aggregation_mutate_rate = 0.0

    bias_init_mean        = 0.0
    bias_init_stdev       = 1.0
    bias_max_value        = 30.0
    bias_min_value        = -30.0
    bias_mutate_power     = 0.5
    bias_mutate_rate      = 0.7
    bias_replace_rate     = 0.1

    compatibility_disjoint_coefficient = 1.0
    compatibility_weight_coefficient   = 0.5

    response_init_mean    = 1.0
    response_init_stdev   = 0.0
    response_max_value    = 30.0
    response_min_value    = -30.0
    response_mutate_power = 0.0
    response_mutate_rate  = 0.0
    response_replace_rate = 0.0

    weight_init_mean      = 0.0
    weight_init_stdev     = 1.0
    weight_max_value      = 30.0
    weight_min_value      = -30.0
    weight_mutate_power   = 0.5
    weight_mutate_rate    = 0.8
    weight_replace_rate   = 0.1

    enabled_default       = True
    enabled_mutate_rate   = 0.01

    conn_add_prob         = 0.5
    conn_delete_prob      = 0.3
    node_add_prob         = 0.2
    node_delete_prob      = 0.1

    [DefaultSpeciesSet]
    compatibility_threshold = 3.0

    [DefaultStagnation]
    species_fitness_func = max
    max_stagnation       = 15
    species_elitism      = 2

    [DefaultReproduction]
    elitism               = 2
    survival_threshold    = 0.2
    """
    return (CONFIG_TEXT,)


@app.cell
def _(CONFIG_TEXT, neat, os, tempfile):
    cfg_dir = tempfile.mkdtemp()
    cfg_path = os.path.join(cfg_dir, "neat_cstr.ini")
    with open(cfg_path, "w") as f:
        f.write(CONFIG_TEXT)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        cfg_path,
    )
    return (config,)


@app.cell
def _(ENV_PARAMS, np, pcgym, pd):
    def reset_env():
        env = pcgym.make_env(ENV_PARAMS)
        out = env.reset()
        if isinstance(out, tuple) and len(out) == 2:
            obs, state = out
        else:
            obs, state = out, None
        return env, np.asarray(obs, dtype=float), state

    def step_env(env, action):
        out = env.step(np.array([action], dtype=float))
        if len(out) == 5:
            obs, rew, done, term, info = out
            done = bool(done) or bool(term)
        elif len(out) == 6:
            obs, rew, done, term, trunc, info = out
            done = bool(done) or bool(term) or bool(trunc)
        else:
            obs, rew, done, info = out
            done = bool(done)
        return np.asarray(obs, dtype=float), float(rew), done, info

    def scale_action(raw, low=295.0, high=302.0):
        y = np.tanh(raw)
        return float(low + (y + 1.0) * 0.5 * (high - low))

    def rollout_policy(policy_fn, seed=1):
        env, obs, _ = reset_env()
        rows = []
        total_reward = 0.0

        for t in range(ENV_PARAMS["N"]):
            action = policy_fn(obs, t)
            obs, rew, done, info = step_env(env, action)
            total_reward += rew
            state = info.get("state", None) if isinstance(info, dict) else None
            Ca = float(state[0]) if state is not None else float(obs[0])
            T = float(state[1]) if state is not None else float(obs[1])
            sp = float(obs[2]) if len(obs) > 2 else float(ENV_PARAMS["SP"]["Ca"][t])

            rows.append(
                {
                    "t": t,
                    "Ca": Ca,
                    "T": T,
                    "SP": sp,
                    "u": float(action),
                    "reward": rew,
                    "cum_reward": total_reward,
                }
            )
            if done:
                break

        return pd.DataFrame(rows)

    def rollout_baseline(u_value=298.0):
        return rollout_policy(lambda obs, t: u_value, seed=1)

    return rollout_baseline, rollout_policy, scale_action


@app.cell
def _(FeedForwardNetwork, config, neat, np, rollout_policy, scale_action):
    def make_neat_policy(genome):
        net = FeedForwardNetwork.create(genome, config)

        def policy(obs, t):
            raw = float(net.activate(np.asarray(obs, dtype=float))[0])
            return scale_action(raw)

        return policy, net

    def evaluate_genomes(genomes, cfg):
        for _, genome in genomes:
            policy, _ = make_neat_policy(genome)
            df = rollout_policy(policy, seed=0)
            # mse = float(((df["Ca"] - df["SP"]) ** 2).mean())
            # smooth = float(((df["u"].diff().fillna(0.0)) ** 2).mean())
            # genome.fitness = -(mse + 0.01 * smooth)
            mse = ((df["Ca"] - df["SP"]) ** 2).mean()
            smooth = ((df["u"].diff().fillna(0.0)) ** 2).mean()
            range_penalty = float((df["Ca"].lt(0.7) | df["Ca"].gt(1.0)).mean())
            genome.fitness = -(mse + 0.01 * smooth + 10.0 * range_penalty)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(False))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    winner = p.run(evaluate_genomes, 40)

    policy_winner, winner_net = make_neat_policy(winner)
    df_neat = rollout_policy(policy_winner, seed=1)
    return df_neat, winner_net


@app.cell
def _(rollout_baseline):
    df_base = rollout_baseline(298.0)
    return (df_base,)


@app.cell
def _(df_base, df_neat, go, sp):
    fig = sp.make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("Concentration vs setpoint", "Reactor temperature", "Coolant input", "Reward"),
    )

    fig.add_trace(go.Scatter(x=df_base["t"], y=df_base["Ca"], name="Ca baseline", line=dict(width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_neat["t"], y=df_neat["Ca"], name="Ca NEAT", line=dict(width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_neat["t"], y=df_neat["SP"], name="Setpoint", line=dict(width=2, dash="dash")), row=1, col=1)

    fig.add_trace(go.Scatter(x=df_base["t"], y=df_base["T"], name="T baseline", line=dict(width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_neat["t"], y=df_neat["T"], name="T NEAT", line=dict(width=2)), row=2, col=1)

    fig.add_trace(go.Scatter(x=df_base["t"], y=df_base["u"], name="u baseline", line=dict(width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_neat["t"], y=df_neat["u"], name="u NEAT", line=dict(width=2)), row=3, col=1)

    fig.add_trace(go.Scatter(x=df_base["t"], y=df_base["cum_reward"], name="Cumulative reward baseline", line=dict(width=2)), row=4, col=1)
    fig.add_trace(go.Scatter(x=df_neat["t"], y=df_neat["cum_reward"], name="Cumulative reward NEAT", line=dict(width=2)), row=4, col=1)

    fig.update_layout(height=1100, width=1100, legend=dict(orientation="h"))
    fig.update_xaxes(title_text="Time step", row=4, col=1)
    fig.update_yaxes(title_text="Concentration", row=1, col=1)
    fig.update_yaxes(title_text="Temperature", row=2, col=1)
    fig.update_yaxes(title_text="Coolant u", row=3, col=1)
    fig.update_yaxes(title_text="Reward", row=4, col=1)
    fig
    return


@app.cell
def _(df_base, df_neat, go):
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df_base["t"], y=df_base["Ca"], name="Ca baseline", line=dict(width=2)))
    fig2.add_trace(go.Scatter(x=df_neat["t"], y=df_neat["Ca"], name="Ca NEAT", line=dict(width=2)))
    fig2.add_trace(go.Scatter(x=df_neat["t"], y=df_neat["SP"], name="Setpoint", line=dict(width=2, dash="dash")))
    fig2.add_trace(go.Scatter(x=df_base["t"], y=df_base["u"], name="u baseline", yaxis="y2", line=dict(width=1, color="green", dash="dot")))
    fig2.add_trace(go.Scatter(x=df_neat["t"], y=df_neat["u"], name="u NEAT", yaxis="y2", line=dict(width=2, color="red")))

    fig2.update_layout(
        height=600,
        width=1100,
        xaxis=dict(title="Time step"),
        yaxis=dict(title="Concentration"),
        yaxis2=dict(title="Coolant input", overlaying="y", side="right"),
        legend=dict(orientation="h"),
    )
    fig2
    return


@app.cell
def _(df_base, df_neat):
    import marimo
    ne = df_neat
    ba = df_base
    marimo.md(
        f"""
    ### Performance summary

    | Metric | NEAT | Baseline |
    |---|---:|---:|
    | Total Reward | {ne["cum_reward"].iloc[-1]:.3f} | {ba["cum_reward"].iloc[-1]:.3f} |
    | MSE | {((ne["Ca"] - ne["SP"]) ** 2).mean():.6f} | {((ba["Ca"] - ba["SP"]) ** 2).mean():.6f} |
    | Final Ca | {ne["Ca"].iloc[-1]:.4f} | {ba["Ca"].iloc[-1]:.4f} |
    | Final u | {ne["u"].iloc[-1]:.4f} | {ba["u"].iloc[-1]:.4f} |
    """
    )
    return (marimo,)


@app.cell
def _(df_base, df_neat, marimo, np, scale_action, winner_net):
    probe_obs = np.array([0.82, 331.0, 0.85], dtype=float)
    raw_out = float(winner_net.activate(probe_obs)[0])
    scaled_u = scale_action(raw_out)

    marimo.md(
        f"""
    ### Debug panel

    - Probe observation: `{probe_obs.tolist()}`
    - Raw NEAT output: `{raw_out:.6f}`
    - Scaled action: `{scaled_u:.6f}`
    - NEAT first 5 actions: `{df_neat["u"].head(5).round(4).tolist()}`
    - Baseline first 5 actions: `{df_base["u"].head(5).round(4).tolist()}`
    """
    )
    return


@app.cell
def _(df_base, df_neat):
    debug_neat = df_neat.head(5).copy()
    debug_base = df_base.head(5).copy()
    return debug_base, debug_neat


@app.cell
def _(debug_neat):
    debug_neat
    return


@app.cell
def _(debug_base):
    debug_base
    return


@app.cell
def _(df_base, df_neat):
    df_neat.to_csv("cstr_neat_rollout.csv", index=False)
    df_base.to_csv("cstr_baseline_rollout.csv", index=False)
    return


if __name__ == "__main__":
    app.run()
