#!/usr/bin/env python3
import os
import sys
import csv
import glob
import math
import shutil
import pickle
import random
import multiprocessing as mp
from itertools import count

import gymnasium as gym
import neat
import numpy as np
from neat.math_util import mean
from neat.config import ConfigParameter, DefaultClassConfig

GENERATIONS = 500
EPISODES_PER_GENOME = 4
MAX_STEPS = 1000
STALL_STEP_LIMIT = 120
MIN_PROGRESS_DELTA = 0.15
VIDEO_OUTPUT = 'hopper_winner_episode.mp4'
GENOME_OUTPUT = 'winner_hopper_genome.pkl'
CONFIG_SNAPSHOT_OUTPUT = 'winner_hopper_config.pkl'
STATS_OUTPUT = 'evolution_statistics.pkl'
GENEALOGY_OUTPUT = 'genealogy_events.csv'
GENEALOGY_EVENTS = []

class SimpleInnovationTracker:
    def __init__(self):
        self.connection_innovations = {}
        self.node_innovations = {}
        self.next_connection_innovation = 0
        self.next_node_id = None

    def initialize(self, genome_config):
        if self.next_node_id is None:
            node_ids = list(genome_config.output_keys)
            self.next_node_id = (max(node_ids) + 1) if node_ids else 0

    def get_connection_innovation(self, input_key, output_key):
        return self.get_innovation_number(input_key, output_key)

    def get_innovation_number(self, input_key, output_key, mutation_type='add_connection'):
        key = (input_key, output_key, mutation_type)
        if key not in self.connection_innovations:
            self.connection_innovations[key] = self.next_connection_innovation
            self.next_connection_innovation += 1
        return self.connection_innovations[key]

    def get_node_innovation(self, input_key, output_key, existing_connections=None):
        key = (input_key, output_key)
        if key not in self.node_innovations:
            node_id = self.next_node_id
            self.next_node_id += 1
            self.node_innovations[key] = (node_id, None, None)
        return self.node_innovations[key]

class GenealogyReproduction:
    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(param_dict, [
            ConfigParameter('elitism', int, 0),
            ConfigParameter('survival_threshold', float, 0.2),
            ConfigParameter('min_species_size', int, 1),
        ])

    def __init__(self, config, reporters, stagnation):
        self.reproduction_config = config
        self.reporters = reporters
        self.genome_indexer = count(1)
        self.stagnation = stagnation
        self.ancestors = {}
        self.innovation_tracker = SimpleInnovationTracker()

    def _ensure_tracker(self, config):
        self.innovation_tracker.initialize(config.genome_config)
        config.genome_config.innovation_tracker = self.innovation_tracker

    @staticmethod
    def _record_event(generation, event, genome_id, species_id='', parent1_id='', parent2_id='', birth_type=''):
        GENEALOGY_EVENTS.append({
            'generation': generation,
            'event': event,
            'genome_id': genome_id,
            'species_id': species_id,
            'parent1_id': parent1_id,
            'parent2_id': parent2_id,
            'birth_type': birth_type,
        })

    def create_new(self, genome_type, genome_config, num_genomes):
        self.innovation_tracker.initialize(genome_config)
        genome_config.innovation_tracker = self.innovation_tracker
        new_genomes = {}
        for _ in range(num_genomes):
            key = next(self.genome_indexer)
            g = genome_type(key)
            g.configure_new(genome_config)
            new_genomes[key] = g
            self.ancestors[key] = tuple()
            self._record_event(0, 'birth', key, birth_type='initial')
        return new_genomes

    @staticmethod
    def compute_spawn(adjusted_fitness, previous_sizes, pop_size, min_species_size):
        af_sum = sum(adjusted_fitness)
        spawn_amounts = []
        for af, ps in zip(adjusted_fitness, previous_sizes):
            if af_sum > 0:
                s = max(min_species_size, af / af_sum * pop_size)
            else:
                s = min_species_size
            d = (s - ps) * 0.5
            c = int(round(d))
            spawn = ps
            if abs(c) > 0:
                spawn += c
            elif d > 0:
                spawn += 1
            elif d < 0:
                spawn -= 1
            spawn_amounts.append(spawn)
        total_spawn = sum(spawn_amounts)
        norm = pop_size / total_spawn
        return [max(min_species_size, int(round(n * norm))) for n in spawn_amounts]

    def reproduce(self, config, species, pop_size, generation):
        self._ensure_tracker(config)
        all_fitnesses = []
        remaining_species = []
        for stag_sid, stag_s, stagnant in self.stagnation.update(species, generation):
            if stagnant:
                self.reporters.species_stagnant(stag_sid, stag_s)
            else:
                all_fitnesses.extend(m.fitness for m in stag_s.members.values())
                remaining_species.append(stag_s)

        if not remaining_species:
            species.species = {}
            return {}

        min_fitness = min(all_fitnesses)
        max_fitness = max(all_fitnesses)
        fitness_range = max(1.0, max_fitness - min_fitness)
        for afs in remaining_species:
            msf = mean([m.fitness for m in afs.members.values()])
            afs.adjusted_fitness = (msf - min_fitness) / fitness_range

        adjusted_fitnesses = [s.adjusted_fitness for s in remaining_species]
        self.reporters.info('Average adjusted fitness: {:.3f}'.format(mean(adjusted_fitnesses)))
        previous_sizes = [len(s.members) for s in remaining_species]
        min_species_size = max(self.reproduction_config.min_species_size, self.reproduction_config.elitism)
        spawn_amounts = self.compute_spawn(adjusted_fitnesses, previous_sizes, pop_size, min_species_size)

        old_population_ids = set()
        for s in remaining_species:
            old_population_ids.update(s.members.keys())

        new_population = {}
        species.species = {}
        next_population_ids = set()

        for spawn, s in zip(spawn_amounts, remaining_species):
            spawn = max(spawn, self.reproduction_config.elitism)
            old_members = list(s.members.items())
            s.members = {}
            species.species[s.key] = s
            old_members.sort(reverse=True, key=lambda x: x[1].fitness)

            elites_to_copy = min(self.reproduction_config.elitism, len(old_members))
            for gid, genome in old_members[:elites_to_copy]:
                new_population[gid] = genome
                next_population_ids.add(gid)
                self._record_event(generation, 'survive', gid, species_id=s.key, parent1_id=gid, birth_type='elite_copy')
            spawn -= elites_to_copy

            if spawn <= 0:
                continue

            repro_cutoff = int(math.ceil(self.reproduction_config.survival_threshold * len(old_members)))
            repro_cutoff = max(repro_cutoff, 2)
            old_members = old_members[:repro_cutoff]

            while spawn > 0:
                spawn -= 1
                parent1_id, parent1 = random.choice(old_members)
                parent2_id, parent2 = random.choice(old_members)
                gid = next(self.genome_indexer)
                child = config.genome_type(gid)
                child.configure_crossover(parent1, parent2, config.genome_config)
                config.genome_config.innovation_tracker = self.innovation_tracker
                child.mutate(config.genome_config)
                new_population[gid] = child
                next_population_ids.add(gid)
                self.ancestors[gid] = (parent1_id, parent2_id)
                self._record_event(generation, 'birth', gid, species_id=s.key, parent1_id=parent1_id, parent2_id=parent2_id, birth_type='crossover_mutation')

        for gid in sorted(old_population_ids - next_population_ids):
            self._record_event(generation, 'loss', gid)
        return new_population


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
    no_progress_penalty = 90.0 if distance < 0.25 else (35.0 if distance < 0.75 else 0.0)
    articulation_penalty = 60.0 if avg_joint_motion < 0.03 else (20.0 if avg_joint_motion < 0.06 else 0.0)
    fall_penalty = 0.0 if survived_ratio > 0.95 else (1.0 - survived_ratio) * 40.0
    return distance_score + reward_score + articulation_bonus + active_control_bonus - no_progress_penalty - articulation_penalty - fall_penalty

def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    return float(np.mean([evaluate_single_episode(net, seed=s) for s in range(EPISODES_PER_GENOME)]))

def save_winner_video(winner, config, output_path=VIDEO_OUTPUT):
    temp_dir = 'hopper_video_tmp'
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    env = gym.make('Hopper-v5', render_mode='rgb_array')
    env = gym.wrappers.RecordVideo(env, video_folder=temp_dir, name_prefix='hopper-winner', episode_trigger=lambda _: True, disable_logger=True)
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    obs, info = env.reset(seed=999)
    for _ in range(MAX_STEPS):
        raw_action = np.asarray(net.activate(obs), dtype=np.float32)
        action = np.tanh(raw_action).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    env.close()
    mp4s = sorted(glob.glob(os.path.join(temp_dir, '*.mp4')))
    if not mp4s:
        raise RuntimeError('RecordVideo did not create an MP4 file.')
    shutil.copy2(mp4s[0], output_path)
    return output_path

def write_genealogy_events(path):
    fieldnames = ['generation', 'event', 'genome_id', 'species_id', 'parent1_id', 'parent2_id', 'birth_type']
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(GENEALOGY_EVENTS)

def run(config_path):
    config = neat.Config(neat.DefaultGenome, GenealogyReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
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
    write_genealogy_events(GENEALOGY_OUTPUT)
    video_path = save_winner_video(winner, config, VIDEO_OUTPUT)
    print(f'Winner genome saved to: {GENOME_OUTPUT}')
    print(f'Config snapshot saved to: {CONFIG_SNAPSHOT_OUTPUT}')
    print(f'Statistics saved to: {STATS_OUTPUT}')
    print(f'Genealogy events saved to: {GENEALOGY_OUTPUT}')
    print(f'Winner episode video saved to: {video_path}')

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'hopper_neat_config.txt')
    if not os.path.exists(config_path):
        print(f'Error: missing config file at {config_path}')
        sys.exit(1)
    run(config_path)