#!/usr/bin/env python3
import argparse
import csv
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

def build_rows(stats):
    rows = {}
    for gen_idx, species_map in enumerate(stats.generation_statistics):
        for sid, members in species_map.items():
            for gid, fitness in members.items():
                if gid not in rows:
                    rows[gid] = {'genome_id': gid, 'species_id_first': sid, 'species_id_last': sid, 'birth_generation_est': gen_idx, 'death_generation_est': gen_idx, 'best_fitness_seen': fitness}
                rows[gid]['species_id_last'] = sid
                rows[gid]['death_generation_est'] = gen_idx
                rows[gid]['best_fitness_seen'] = max(rows[gid]['best_fitness_seen'], fitness)
    return list(rows.values())

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stats', default='evolution_statistics.pkl')
    p.add_argument('--outdir', default='neat_genealogy_output')
    args = p.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(args.stats, 'rb') as f:
        stats = pickle.load(f)
    rows = build_rows(stats)
    with open(outdir / 'genome_lifespans_estimated.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    rows = sorted(rows, key=lambda r: (r['birth_generation_est'], -r['best_fitness_seen']))[:400]
    fig, ax = plt.subplots(figsize=(16, max(7, len(rows) * 0.04)))
    cmap = plt.get_cmap('tab20')
    for y, row in enumerate(rows):
        color = cmap(((row['species_id_first'] or 1) - 1) % 20)
        ax.hlines(y, row['birth_generation_est'], row['death_generation_est'], color=color, linewidth=2)
        ax.plot(row['birth_generation_est'], y, 'o', color=color, markersize=3)
        ax.plot(row['death_generation_est'], y, 'x', color=color, markersize=4)
    ax.set_title('Estimated genome birth and loss timeline')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Genome index')
    ax.grid(True, alpha=0.2)
    fig.tight_layout(); fig.savefig(outdir / 'genome_lifespans_timeline.png', dpi=180)

if __name__ == '__main__':
    main()
