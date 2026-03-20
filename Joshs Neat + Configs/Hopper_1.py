# Importing necessary modules
import neat
import gymnasium as gym
import pickle
import numpy as np
import visualize
from network_report import save_network_report

# A series of hyperparameters
GENERATIONS = 600
MAX_STEPS   = 1000  # Hopper episodes are shorter than walker
NUM_RUNS    = 3     # MuJoCo is fast enough for multiple runs

# Hopper-v5: 11 inputs, 3 outputs
# Observations: z (height), angle, thigh joint, leg joint, foot joint,
#               + velocities of all 5 above, plus x velocity
NODE_NAMES = {
    -1:  "z (height)",
    -2:  "torso angle",
    -3:  "thigh joint angle",
    -4:  "leg joint angle",
    -5:  "foot joint angle",
    -6:  "x velocity",
    -7:  "z velocity",
    -8:  "torso ang vel",
    -9:  "thigh ang vel",
    -10: "leg ang vel",
    -11: "foot ang vel",
    0:   "thigh torque",
    1:   "leg torque",
    2:   "foot torque",
}

env = gym.make("Hopper-v5")

# --- Best genome tracking ---
best_genome  = None
best_fitness = float("-inf")

def eval_genomes(genomes, config):
    global best_genome, best_fitness

    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        total_reward = 0.0
        for run in range(NUM_RUNS):
            obs, _ = env.reset()
            for step in range(MAX_STEPS):
                action = net.activate(obs)
                action = np.clip(action, -1, 1)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
        genome.fitness = total_reward / NUM_RUNS

        # Save the best genome seen across ALL generations
        if genome.fitness > best_fitness:
            best_fitness = genome.fitness
            best_genome  = pickle.loads(pickle.dumps(genome))  # deep copy
            print(f"  >> New best genome! Fitness: {best_fitness:.2f}")

            # Persist immediately so a crash never loses it
            with open("neat_best.pkl", "wb") as f:
                pickle.dump(best_genome, f)

            # Save CSV report for the new best
            save_network_report(
                best_genome, config,
                filename="network_report_best",
                node_names=NODE_NAMES,
            )


config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    "config_hopper",
)

pop = neat.Population(config)
pop.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
pop.add_reporter(stats)

# Save population checkpoints every 25 generations so you can resume
pop.add_reporter(neat.Checkpointer(25, filename_prefix="neat-checkpoint-"))

final_winner = pop.run(eval_genomes, GENERATIONS)

# Save the final generation winner too (for comparison)
with open("neat_winner_final.pkl", "wb") as f:
    pickle.dump(final_winner, f)

print(f"\nFinal generation winner fitness: {final_winner.fitness:.2f}")
print(f"All-time best genome fitness:    {best_fitness:.2f}")

# Use the all-time best for visualisation and rendering
winner = best_genome

# ── Network report ─────────────────────────────────────────────────────────────
save_network_report(winner, config,
                    filename="network_report_final",
                    node_names=NODE_NAMES)

# ── Visualisations ─────────────────────────────────────────────────────────────
visualize.plot_stats(stats, ylog=False, view=True, filename="fitness.png")
visualize.plot_species(stats, view=True, filename="speciation.png")
visualize.draw_net(config, winner, view=True, filename="network",
                   node_names=NODE_NAMES, prune_unused=True)

# ── Render the winner ──────────────────────────────────────────────────────────
env = gym.make("Hopper-v5", render_mode="human")
net = neat.nn.FeedForwardNetwork.create(winner, config)
obs, _ = env.reset()

for _ in range(MAX_STEPS):
    action = net.activate(obs)
    action = np.clip(action, -1, 1)
    obs, _, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()  # keep rendering, reset on fall

env.close()