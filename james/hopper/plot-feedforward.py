#!/usr/bin/env python3
import argparse
import csv
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def load_stats(stats_path):
    with open(stats_path, 'rb') as f:
        return pickle.load(f)


def generation_fitness_arrays(stats):
    arrays = []
    for gen_data in stats.generation_statistics:
        scores = []
        for species_stats in gen_data.values():
            scores.extend(v for v in species_stats.values() if v is not None)
        arrays.append(np.array(scores, dtype=float))
    return arrays


def summarize_stats(stats):
    arrays = generation_fitness_arrays(stats)
    best = [g.fitness for g in stats.most_fit_genomes]
    mean = stats.get_fitness_mean()
    stdev = stats.get_fitness_stdev()
    median = stats.get_fitness_median()
    species_sizes = stats.get_species_sizes()
    species_fitness = stats.get_species_fitness(null_value=np.nan)

    rows = []
    for gen, arr in enumerate(arrays):
        if arr.size == 0:
            continue
        rows.append({
            'generation': gen,
            'population_size': int(arr.size),
            'species_count': int(sum(1 for x in species_sizes[gen] if x > 0)) if gen < len(species_sizes) else 0,
            'min_fitness': float(np.min(arr)),
            'q1_fitness': float(np.percentile(arr, 25)),
            'median_fitness': float(np.median(arr)),
            'mean_fitness': float(mean[gen]),
            'q3_fitness': float(np.percentile(arr, 75)),
            'max_fitness': float(np.max(arr)),
            'best_fitness': float(best[gen]),
            'stdev_fitness': float(stdev[gen]),
        })
    return rows, arrays, species_sizes, species_fitness


def write_csv_summary(rows, out_csv):
    fieldnames = list(rows[0].keys()) if rows else []
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_fitness_lines(rows, out_path):
    gens = [r['generation'] for r in rows]
    best = [r['best_fitness'] for r in rows]
    mean = [r['mean_fitness'] for r in rows]
    median = [r['median_fitness'] for r in rows]

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(gens, best, label='Best fitness', linewidth=2.2, color='#a12c7b')
    ax.plot(gens, mean, label='Mean fitness', linewidth=2.0, color='#01696f')
    ax.plot(gens, median, label='Median fitness', linewidth=1.8, color='#28251d')
    ax.set_title('Best, mean, and median fitness by generation')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_fitness_dispersion(rows, out_path):
    gens = [r['generation'] for r in rows]
    stdev = [r['stdev_fitness'] for r in rows]
    q1 = [r['q1_fitness'] for r in rows]
    q3 = [r['q3_fitness'] for r in rows]

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(gens, stdev, label='Fitness stdev', linewidth=2.0, color='#d19900')
    ax.fill_between(gens, q1, q3, color='#01696f', alpha=0.18, label='Interquartile range')
    ax.set_title('Population dispersion by generation')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_species_sizes(species_sizes, out_path):
    arr = np.array(species_sizes, dtype=float)
    gens = np.arange(arr.shape[0])

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.stackplot(gens, arr.T, alpha=0.85)
    ax.set_title('Species sizes over time')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Genomes in species')
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_species_fitness(species_fitness, out_path):
    arr = np.array(species_fitness, dtype=float)
    gens = np.arange(arr.shape[0])

    fig, ax = plt.subplots(figsize=(13, 6))
    for i in range(arr.shape[1]):
        ax.plot(gens, arr[:, i], linewidth=1.5, alpha=0.85, label=f'Species {i+1}')
    ax.set_title('Mean species fitness over time')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Mean species fitness')
    ax.grid(True, alpha=0.25)
    if arr.shape[1] <= 12:
        ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_candlestick(rows, out_path):
    fig, ax = plt.subplots(figsize=(15, 7))
    width = 0.65

    for r in rows:
        x = r['generation']
        low = r['min_fitness']
        high = r['max_fitness']
        open_ = r['q1_fitness']
        close = r['q3_fitness']
        median = r['median_fitness']
        color = '#01696f' if close >= open_ else '#964219'

        ax.vlines(x, low, high, color=color, linewidth=1.0, alpha=0.9)
        rect_y = min(open_, close)
        rect_h = max(abs(close - open_), 1e-9)
        rect = patches.Rectangle((x - width / 2, rect_y), width, rect_h, facecolor=color, edgecolor=color, alpha=0.35)
        ax.add_patch(rect)
        ax.hlines(median, x - width / 2, x + width / 2, color='#28251d', linewidth=1.2)

    ax.plot([r['generation'] for r in rows], [r['best_fitness'] for r in rows], color='#a12c7b', linewidth=1.8, label='Best fitness')
    ax.plot([r['generation'] for r in rows], [r['mean_fitness'] for r in rows], color='#006494', linewidth=1.4, alpha=0.9, label='Mean fitness')
    ax.set_title('Fitness distribution by generation (candlestick style)')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_boxplot_snapshots(arrays, out_path, max_boxes=25):
    total = len(arrays)
    if total == 0:
        return
    if total <= max_boxes:
        idx = list(range(total))
    else:
        idx = np.unique(np.linspace(0, total - 1, max_boxes, dtype=int)).tolist()

    data = [arrays[i] for i in idx if arrays[i].size > 0]
    labels = [str(i) for i in idx if arrays[i].size > 0]

    fig, ax = plt.subplots(figsize=(15, 7))
    bp = ax.boxplot(data, patch_artist=True, labels=labels, showfliers=False)
    for box in bp['boxes']:
        box.set(facecolor='#cedcd8', edgecolor='#01696f', alpha=0.85)
    for med in bp['medians']:
        med.set(color='#28251d', linewidth=1.6)
    for whisk in bp['whiskers']:
        whisk.set(color='#01696f', linewidth=1.1)
    for cap in bp['caps']:
        cap.set(color='#01696f', linewidth=1.1)

    ax.set_title('Fitness boxplots for sampled generations')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    ax.grid(True, axis='y', alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_dashboard(rows, species_sizes, out_path):
    gens = [r['generation'] for r in rows]
    best = [r['best_fitness'] for r in rows]
    mean = [r['mean_fitness'] for r in rows]
    median = [r['median_fitness'] for r in rows]
    stdev = [r['stdev_fitness'] for r in rows]
    species_count = [r['species_count'] for r in rows]
    q1 = [r['q1_fitness'] for r in rows]
    q3 = [r['q3_fitness'] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    axes[0, 0].plot(gens, best, label='Best', color='#a12c7b', linewidth=2)
    axes[0, 0].plot(gens, mean, label='Mean', color='#01696f', linewidth=1.8)
    axes[0, 0].plot(gens, median, label='Median', color='#28251d', linewidth=1.6)
    axes[0, 0].set_title('Fitness trend')
    axes[0, 0].set_xlabel('Generation')
    axes[0, 0].set_ylabel('Fitness')
    axes[0, 0].grid(True, alpha=0.25)
    axes[0, 0].legend()

    axes[0, 1].fill_between(gens, q1, q3, color='#01696f', alpha=0.2, label='IQR')
    axes[0, 1].plot(gens, stdev, color='#d19900', linewidth=2, label='Stdev')
    axes[0, 1].set_title('Population spread')
    axes[0, 1].set_xlabel('Generation')
    axes[0, 1].set_ylabel('Fitness / spread')
    axes[0, 1].grid(True, alpha=0.25)
    axes[0, 1].legend()

    axes[1, 0].plot(gens, species_count, color='#006494', linewidth=2)
    axes[1, 0].set_title('Species count')
    axes[1, 0].set_xlabel('Generation')
    axes[1, 0].set_ylabel('Active species')
    axes[1, 0].grid(True, alpha=0.25)

    arr = np.array(species_sizes, dtype=float)
    if arr.size:
        axes[1, 1].stackplot(np.arange(arr.shape[0]), arr.T, alpha=0.85)
    axes[1, 1].set_title('Species sizes')
    axes[1, 1].set_xlabel('Generation')
    axes[1, 1].set_ylabel('Genomes')
    axes[1, 1].grid(True, alpha=0.2)

    fig.suptitle('NEAT evolution diagnostics', fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Analyze a NEAT-Python StatisticsReporter pickle and export graphs.')
    parser.add_argument('--stats', default='evolution_statistics.pkl', help='Path to evolution_statistics.pkl')
    parser.add_argument('--outdir', default='stats-feedforward', help='Directory for generated charts and CSV')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    stats = load_stats(args.stats)
    rows, arrays, species_sizes, species_fitness = summarize_stats(stats)
    if not rows:
        raise RuntimeError('No generation statistics found in the pickle.')

    write_csv_summary(rows, outdir / 'fitness_summary.csv')
    plot_fitness_lines(rows, outdir / 'fitness_lines.png')
    plot_fitness_dispersion(rows, outdir / 'fitness_dispersion.png')
    plot_species_sizes(species_sizes, outdir / 'species_sizes.png')
    plot_species_fitness(species_fitness, outdir / 'species_fitness.png')
    plot_candlestick(rows, outdir / 'fitness_candlestick.png')
    plot_boxplot_snapshots(arrays, outdir / 'fitness_boxplot_snapshots.png')
    plot_dashboard(rows, species_sizes, outdir / 'neat_dashboard.png')

    with open(outdir / 'README.txt', 'w') as f:
        f.write(
            'Generated files:\n'
            '- fitness_summary.csv\n'
            '- fitness_lines.png\n'
            '- fitness_dispersion.png\n'
            '- species_sizes.png\n'
            '- species_fitness.png\n'
            '- fitness_candlestick.png\n'
            '- fitness_boxplot_snapshots.png\n'
            '- neat_dashboard.png\n\n'
            'Run with:\n'
            'python analyze_neat_stats.py --stats evolution_statistics.pkl --outdir neat_stats_output\n'
        )

    print(f'Wrote charts and CSV to {outdir}')


if __name__ == '__main__':
    main()