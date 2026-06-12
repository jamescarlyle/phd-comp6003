#!/usr/bin/env python3
import os
import glob
import shutil
import pickle
import gymnasium as gym
import numpy as np
import neat

def main():
    with open('winner_hopper_genome.pkl', 'rb') as f:
        genome = pickle.load(f)
    with open('winner_hopper_config.pkl', 'rb') as f:
        config = pickle.load(f)
    temp_dir = 'hopper_video_tmp'
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    env = gym.make('Hopper-v5', render_mode='rgb_array')
    env = gym.wrappers.RecordVideo(env, video_folder=temp_dir, name_prefix='hopper-winner', episode_trigger=lambda _: True, disable_logger=True)
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    obs, info = env.reset(seed=999)
    for _ in range(1000):
        action = np.tanh(np.asarray(net.activate(obs), dtype=np.float32)).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    env.close()
    mp4s = sorted(glob.glob(os.path.join(temp_dir, '*.mp4')))
    if not mp4s:
        raise RuntimeError('No MP4 produced')
    shutil.copy2(mp4s[0], 'hopper_winner_episode.mp4')
    print('Saved hopper_winner_episode.mp4')

if __name__ == '__main__':
    main()
