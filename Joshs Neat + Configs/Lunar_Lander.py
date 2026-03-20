# Importing nessesary modules
import neat
import gymnasium as gym
import pickle
import visualize

# A series of hyperparameters
GENERATIONS = 600
MAX_STEPS   = 600
NUM_RUNS    = 10

NODE_NAMES = {
    -1: "x coord",
    -2: "y coord",
    -3: "x vel",
    -4: "y vel",
    -5: "angle",
    -6: "ang acc",
    -7: "L down",
    -8: "R down",
    0: "Do Nothing",
    1: "Left Engine",
    2: "Main Engine",
    3: "Right Engine",
}



def eval_genomes(genomes, config):
    env = gym.make("LunarLander-v3")
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        total_reward = 0.0
        for run in range(NUM_RUNS):
            obs, info = env.reset()
            for step in range(MAX_STEPS):
                action = net.activate(obs)
                action = action.index(max(action))
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
        genome.fitness = total_reward / NUM_RUNS
    env.close()


config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    "config_lander",
)

pop = neat.Population(config)
pop.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
pop.add_reporter(stats)

winner = pop.run(eval_genomes, GENERATIONS)

with open("neat_winner.pkl", "wb") as f:
    pickle.dump(winner, f)

# ── Visualisations ─────────────────────────────────────────────────────────────
visualize.plot_stats(stats,  ylog=False, view=True, filename="fitness.png")
visualize.plot_species(stats, view=True, filename="speciation.png")
visualize.draw_net(config, winner, view=True, filename="network",
                   node_names=NODE_NAMES, prune_unused=True)

# ── Render the winner ──────────────────────────────────────────────────────────
env = gym.make("LunarLander-v3", render_mode="human")
net = neat.nn.FeedForwardNetwork.create(winner, config)
obs, _ = env.reset()

for _ in range(MAX_STEPS):
    action = net.activate(obs)
    action = action.index(max(action))
    obs, _, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        break

env.close()
