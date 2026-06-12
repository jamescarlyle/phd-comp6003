#!/usr/bin/env python3
import os
import sys
import glob
import shutil
import pickle
import multiprocessing as mp

import gymnasium as gym
import neat
import numpy as np

GENERATIONS = 500
EPISODES_PER_GENOME = 4
MAX_STEPS = 1000
STALL_STEP_LIMIT = 120
MIN_PROGRESS_DELTA = 0.15
VIDEO_OUTPUT = 'hopper_winner_episode.mp4'
GENOME_OUTPUT = 'winner_hopper_genome.pkl'
CONFIG_SNAPSHOT_OUTPUT = 'winner_hopper_config.pkl'
STATS_OUTPUT = 'evolution_statistics.pkl'


def make_env(render_mode=None):
    return gym.make('Hopper-v5', render_mode=render_mode)


def evaluate_single_episode(net, seed=None):
    env = make_env()
    obs, info = env.reset(seed=seed)

    start_x = info.get('x_position', 0.0)
    last_x = start_x
    best_x = start_x
    total_reward = 0.0
    progress_stall_steps = 0
    joint_motion_sum = 0.0
    action_energy_sum = 0.0
    steps = 0

    for step in range(MAX_STEPS):
        raw_action = np.asarray(net.activate(obs), dtype=np.float32)
        action = np.tanh(raw_action).astype(np.float32)
        prev_joint_angles = obs[1:4].copy()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps = step + 1

        current_x = info.get('x_position', last_x)
        best_x = max(best_x, current_x)

        joint_delta = np.abs(obs[1:4] - prev_joint_angles)
        joint_motion_sum += float(np.sum(joint_delta))
        action_energy_sum += float(np.sum(np.abs(action)))

        if current_x > last_x + 1e-4:
            progress_stall_steps = 0
        else:
            progress_stall_steps += 1

        last_x = current_x

        if progress_stall_steps >= STALL_STEP_LIMIT and (best_x - start_x) < MIN_PROGRESS_DELTA:
            break

        if terminated or truncated:
            break

    env.close()

    distance = best_x - start_x
    avg_joint_motion = joint_motion_sum / max(steps, 1)
    avg_action_energy = action_energy_sum / max(steps, 1)
    survived_ratio = steps / MAX_STEPS

    distance_score = max(distance, -1.0) * 150.0
    reward_score = total_reward * 0.15
    articulation_bonus = min(avg_joint_motion, 0.35) * 80.0
    active_control_bonus = min(avg_action_energy, 1.8) * 12.0

    if distance < 0.25:
        no_progress_penalty = 90.0
    elif distance < 0.75:
        no_progress_penalty = 35.0
    else:
        no_progress_penalty = 0.0

    if avg_joint_motion < 0.03:
        articulation_penalty = 60.0
    elif avg_joint_motion < 0.06:
        articulation_penalty = 20.0
    else:
        articulation_penalty = 0.0

    fall_penalty = 0.0 if survived_ratio > 0.95 else (1.0 - survived_ratio) * 40.0

    fitness = (distance_score + reward_score + articulation_bonus + active_control_bonus
        - no_progress_penalty - articulation_penalty - fall_penalty)

    return fitness


def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    episode_scores = []
    for seed in range(EPISODES_PER_GENOME):
        episode_scores.append(evaluate_single_episode(net, seed=seed))
    return float(np.mean(episode_scores))


def run(config_path):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    workers = max(1, mp.cpu_count() - 1)
    pe = neat.ParallelEvaluator(workers, eval_genome)
    print(f'Using {workers} worker processes for genome evaluation.')

    winner = population.run(pe.evaluate, GENERATIONS)

    with open(GENOME_OUTPUT, 'wb') as f:
        pickle.dump(winner, f)
    with open(CONFIG_SNAPSHOT_OUTPUT, 'wb') as f:
        pickle.dump(config, f)
    with open(STATS_OUTPUT, 'wb') as f:
        pickle.dump(stats, f)

    print('\n' + '=' * 60)
    print('EVOLUTION COMPLETE')
    print('=' * 60)
    print(f'Best genome fitness: {winner.fitness:.4f}')
    print(f'Best genome complexity: {len(winner.nodes)} nodes, {len(winner.connections)} connections')
    print(f'Winner genome saved to: {GENOME_OUTPUT}')
    print(f'Config snapshot saved to: {CONFIG_SNAPSHOT_OUTPUT}')
    print(f'Statistics saved to: {STATS_OUTPUT}')


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    if not os.path.exists(config_path):
        print(f'Error: missing config file at {config_path}')
        sys.exit(1)
    run(config_path)