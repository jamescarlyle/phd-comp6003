#!/usr/bin/env python3
import argparse
import pickle
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stats', default='evolution_statistics.pkl')
    p.add_argument('--outdir', default='neat_stats_output')
    args = p.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(args.stats, 'rb') as f:
        stats = pickle.load(f)
    means = stats.get_fitness_mean()
    stdev = stats.get_fitness_stdev()
    medians = stats.get_fitness_median()
    best = [g.fitness for g in stats.most_fit_genomes]
    gens = list(range(len(means)))
    with open(outdir / 'fitness_summary.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['generation', 'mean', 'stdev', 'median', 'best'])
        for row in zip(gens, means, stdev, medians, best):
            w.writerow(row)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(gens, means, label='Mean')
    ax.plot(gens, medians, label='Median')
    ax.plot(gens, best, label='Best')
    ax.fill_between(gens, np.array(means) - np.array(stdev), np.array(means) + np.array(stdev), alpha=0.2, label='±1 stdev')
    ax.legend(); ax.grid(True, alpha=0.2); ax.set_title('Fitness over generations'); ax.set_xlabel('Generation'); ax.set_ylabel('Fitness')
    fig.tight_layout(); fig.savefig(outdir / 'fitness_lines.png', dpi=180)

if __name__ == '__main__':
    main()
