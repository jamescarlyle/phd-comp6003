import neat
import gymnasium as gym
import pickle
import visualize

GENERATIONS = 100
MAX_STEPS   = 500
NUM_RUNS    = 1

NODE_NAMES = {
    -1: "cart pos",
    -2: "cart vel",
    -3: "pole ang",
    -4: "pole vel",
     0: "action",
}



def eval_genomes(genomes, config):
    env = gym.make("CartPole-v1")
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        total_reward = 0.0
        for _ in range(NUM_RUNS):
            obs, _ = env.reset()
            for _ in range(MAX_STEPS):
                action = 1 if net.activate(obs)[0] > 0.5 else 0
                obs, reward, terminated, truncated, _ = env.step(action)
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
    "config_cartpole",
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
env = gym.make("CartPole-v1", render_mode="human")
net = neat.nn.FeedForwardNetwork.create(winner, config)
obs, _ = env.reset()

for _ in range(MAX_STEPS):
    action = 1 if net.activate(obs)[0] > 0.5 else 0
    obs, _, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        break

env.close()