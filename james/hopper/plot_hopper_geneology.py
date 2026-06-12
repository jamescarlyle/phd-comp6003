#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

def read_events(path):
    rows = []
    with open(path, 'r', newline='') as f:
        for row in csv.DictReader(f):
            row['generation'] = int(row['generation'])
            for key in ('genome_id', 'parent1_id', 'parent2_id', 'species_id'):
                row[key] = None if row[key] == '' else int(row[key])
            rows.append(row)
    return rows

def build_genomes(events):
    genomes = {}
    for e in events:
        gid = e['genome_id']
        if gid is not None and gid not in genomes:
            genomes[gid] = {'genome_id': gid, 'birth_generation': None, 'loss_generation': None, 'species_id': e['species_id'], 'parent1_id': None, 'parent2_id': None, 'birth_type': None}
        if e['event'] == 'birth':
            genomes[gid]['birth_generation'] = e['generation']
            genomes[gid]['species_id'] = e['species_id']
            genomes[gid]['parent1_id'] = e['parent1_id']
            genomes[gid]['parent2_id'] = e['parent2_id']
            genomes[gid]['birth_type'] = e['birth_type']
        elif e['event'] == 'loss' and gid is not None:
            genomes[gid]['loss_generation'] = e['generation']
    max_gen = max((e['generation'] for e in events), default=0)
    for g in genomes.values():
        if g['birth_generation'] is None:
            g['birth_generation'] = 0
        if g['loss_generation'] is None:
            g['loss_generation'] = max_gen
    return genomes

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--events', default='genealogy_events.csv')
    p.add_argument('--outdir', default='genealogy_tree_output')
    p.add_argument('--max-genomes', type=int, default=300)
    args = p.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    events = read_events(args.events)
    genomes = build_genomes(events)
    selected = sorted(genomes.values(), key=lambda g: (g['birth_generation'], g['genome_id']))[:args.max_genomes]
    selected_ids = {g['genome_id'] for g in selected}
    ypos = {g['genome_id']: i for i, g in enumerate(selected)}
    fig, ax = plt.subplots(figsize=(18, max(8, len(selected) * 0.05)))
    cmap = plt.get_cmap('tab20')
    for g in selected:
        color = cmap((((g['species_id'] or 1) - 1) % 20))
        gid = g['genome_id']
        ax.hlines(ypos[gid], g['birth_generation'], g['loss_generation'], color=color, linewidth=1.8, alpha=0.9)
        ax.plot(g['birth_generation'], ypos[gid], 'o', color=color, markersize=3)
        ax.plot(g['loss_generation'], ypos[gid], 'x', color=color, markersize=4)
    edges = 0
    for g in selected:
        for parent_key in ('parent1_id', 'parent2_id'):
            pid = g.get(parent_key)
            if pid is None or pid not in selected_ids or pid == g['genome_id'] or edges >= 900:
                continue
            ax.add_patch(FancyArrowPatch((genomes[pid]['birth_generation'], ypos[pid]), (g['birth_generation'], ypos[g['genome_id']]), arrowstyle='-', linewidth=0.6, color='0.35', alpha=0.25))
            edges += 1
    ax.set_title('NEAT genealogy lifelines with parent-child links')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Genome order (subset)')
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    fig.savefig(outdir / 'genealogy_tree.png', dpi=180)
    plt.close(fig)
    with open(outdir / 'README.txt', 'w') as f:
        f.write('Circle = birth, X = loss, gray links = parent-child.\n')

if __name__ == '__main__':
    main()