#!/usr/bin/env python3
"""
Render the saved Hopper winner genome to an MP4 video.
"""

import os
import sys
import glob
import shutil
import pickle
import gymnasium as gym
import numpy as np
import neat


def load_genome_and_config():
    try:
        with open('winner_hopper_genome.pkl', 'rb') as f:
            genome = pickle.load(f)
    except FileNotFoundError:
        print('Error: winner_hopper_genome.pkl not found. Run training first.')
        sys.exit(1)

    try:
        with open('winner_hopper_config.pkl', 'rb') as f:
            config = pickle.load(f)
    except FileNotFoundError:
        print('Error: winner_hopper_config.pkl not found. Run training first.')
        sys.exit(1)

    return genome, config


def record_episode_mp4(genome, config, output_dir='hopper_videos', max_steps=3000):
    os.makedirs(output_dir, exist_ok=True)

    env = gym.make('Hopper-v5', render_mode='rgb_array', healthy_z_range=(0.5, 2.0), healthy_angle_range=(-0.4, 0.4), max_episode_steps=max_steps)
    env.unwrapped.model.geom_size[0][:2] = [40, 40]
    env.metadata['render_fps'] = 50

    env = gym.wrappers.RecordVideo(
        env,
        video_folder=output_dir,
        name_prefix='hopper-winner',
        episode_trigger=lambda episode_id: True,
        disable_logger=True,
    )

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    obs, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0.0
    step_count = 0
    final_x = 0.0

    while not (terminated or truncated) and step_count < max_steps:
        action = np.array(np.clip(net.activate(obs), -1.0, 1.0), dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        final_x = info.get('x_position', final_x)
        step_count += 1

    env.close()

    print(f'Steps: {step_count}')
    print(f'Total reward: {total_reward:.2f}')
    print(f'Final x position: {final_x:.2f}')


if __name__ == '__main__':
    genome, config = load_genome_and_config()
    record_episode_mp4(genome, config)