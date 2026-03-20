"""
network_report.py  –  Export a NEAT genome as CSV files.
Produces two files:
  - {prefix}_nodes.csv       : key, type, label, activation, bias, response
  - {prefix}_connections.csv : from_key, from_label, to_key, to_label, weight, enabled
"""

import csv


def save_network_report(genome, config, filename="network_report", node_names=None):
    node_names = node_names or {}
    cfg_g = config.genome_config

    input_keys  = set(cfg_g.input_keys)
    output_keys = set(cfg_g.output_keys)

    def label(k):
        return node_names.get(k, str(k))

    def node_type(k):
        if k in input_keys:  return "input"
        if k in output_keys: return "output"
        return "hidden"

    # Strip extension if user passed one, we'll add our own
    prefix = filename.replace(".csv", "").replace(".html", "")

    # ── nodes ──────────────────────────────────────────────────────────────────
    nodes_file = f"{prefix}_nodes.csv"
    with open(nodes_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "type", "label", "activation", "bias", "response"])

        for k in sorted(input_keys):
            w.writerow([k, "input", label(k), "", "", ""])

        for k in sorted(genome.nodes):
            ntype = node_type(k)
            nd = genome.nodes[k]
            w.writerow([k, ntype, label(k), nd.activation, f"{nd.bias:.6f}", f"{nd.response:.6f}"])

    # ── connections ────────────────────────────────────────────────────────────
    conns_file = f"{prefix}_connections.csv"
    with open(conns_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["from_key", "from_label", "to_key", "to_label", "weight", "enabled"])

        for (i, o), conn in sorted(genome.connections.items(), key=lambda x: -abs(x[1].weight)):
            w.writerow([i, label(i), o, label(o), f"{conn.weight:.6f}", conn.enabled])

    print(f"[report] Saved → {nodes_file}, {conns_file}")