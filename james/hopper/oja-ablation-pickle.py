import os
import neat
import numpy as np
import gymnasium as gym
import csv
import pickle
from scipy.stats import wilcoxon
from contextlib import contextmanager

# ==========================================
# 1. OJA NETWORK WITH CORRECTED MATH
# ==========================================
class OjaNetwork:
    """
    Manual Oja network implementation with corrected learning rule.
    Oja's rule: dw = eta * y * (x - w * y)
    This maintains weight normalization naturally while performing PCA.
    """
    def __init__(self, genome, config, eta=0.05):
        self.eta = eta
        self.input_keys = config.genome_config.input_keys
        self.output_keys = config.genome_config.output_keys

        # Extract enabled connections
        self.weights = {}
        for conn_key, conn in genome.connections.items():
            if conn.enabled:
                self.weights[conn_key] = conn.weight

        # Store node properties
        self.nodes = {}
        for node_key, node in genome.nodes.items():
            self.nodes[node_key] = {
                'bias': node.bias,
                'response': node.response,
                'act_func': config.genome_config.activation_defs.get(node.activation)
            }

        # Initialize neural activity (double buffer for recurrency)
        self.values = {k: 0.0 for k in self.input_keys}
        self.values.update({k: 0.0 for k in self.nodes.keys()})

    def activate(self, inputs):
        """
        Forward pass with recurrent dynamics and Oja's rule learning.
        Uses double-buffering to properly handle recurrent connections.
        """
        # Update input nodes with current step's inputs
        for i, val in zip(self.input_keys, inputs):
            self.values[i] = val

        # Standard recurrent forward pass with proper ordering
        new_values = self.values.copy()

        # Compute all node outputs (allow 2 passes for full recurrence)
        for node_key, node_data in self.nodes.items():
            node_input = node_data['bias']

            # Sum weighted inputs from all sources
            for (in_k, out_k), weight in self.weights.items():
                if out_k == node_key:
                    node_input += weight * self.values[in_k]

            # Apply activation with response coefficient
            res = node_data['response']
            new_values[node_key] = node_data['act_func'](res * node_input)

        # ===== CORRECTED OJA'S RULE =====
        # Only apply to output layer connections for stability
        # Math: dw = eta * y * (x - w * y)
        for (in_k, out_k), weight in self.weights.items():
            if out_k in self.output_keys:
                pre_val = self.values[in_k]  # Pre-synaptic activity (t)
                post_val = new_values[out_k]  # Post-synaptic activity (t+1)

                # Oja's rule: maintains ||w|| = 1 and performs PCA
                delta_w = self.eta * post_val * (pre_val - weight * post_val)
                self.weights[(in_k, out_k)] += delta_w

        self.values = new_values
        return [self.values[k] for k in self.output_keys]

    def reset(self):
        """Reset network state between episodes."""
        for k in self.values.keys():
            self.values[k] = 0.0

# ==========================================
# 2. GENOME EVALUATOR (proper NEAT interface)
# ==========================================
class GenomeEvaluator:
    """
    Wraps eval_genomes to provide proper callback interface for NEAT.
    Handles the use_oja parameter cleanly via class state.
    """
    def __init__(self, use_oja=False):
        self.use_oja = use_oja
        self.env = None

    def __call__(self, genomes, config):
        """Called by population.run() - must match interface exactly."""
        if self.env is None:
            self.env = gym.make("Hopper-v5")

        for _, genome in genomes:
            self._evaluate_genome(genome, config)

    def _evaluate_genome(self, genome, config):
        """Evaluate a single genome."""
        # Create appropriate network type
        if self.use_oja:
            net = OjaNetwork(genome, config, eta=0.05)
        else:
            net = neat.nn.RecurrentNetwork.create(genome, config)

        observation, _ = self.env.reset(seed=42)
        fitness = 0.0

        for step in range(1000):
            # Feed observation to network
            action = net.activate(observation)

            # Step environment and accumulate reward
            observation, reward, terminated, truncated, info = self.env.step(action)
            fitness += reward

            if terminated or truncated:
                # Penalty for falling/terminating early
                fitness -= 10.0
                break

        genome.fitness = fitness

    def close(self):
        """Clean up environment."""
        if self.env is not None:
            self.env.close()

# ==========================================
# 3. STRESS TEST WITH GRADIENT ADAPTATION
# ==========================================
def stress_test_genome(genome, config, use_oja, seed=100):
    """
    Adaptive slope test: gradually tilt the environment.
    Hopper must learn to adjust to changing gravity vector in real-time.
    """
    env = gym.make("Hopper-v5", max_episode_steps=4000)

    # Create network
    if use_oja:
        net = OjaNetwork(genome, config, eta=0.05)
    else:
        net = neat.nn.RecurrentNetwork.create(genome, config)

    observation, _ = env.reset(seed=seed)
    total_steps = 0
    max_slope = 0

    try:
        for step in range(4000):
            # Gradually increase slope: +1 degree every 100 steps
            slope_degree = step // 100
            angle_rad = np.radians(slope_degree)

            # Modify gravity vector to simulate tilted surface
            # gravity = [gx, gy, gz] where gz is vertical
            # Rotating around y-axis: gx = g*sin(theta), gz = -g*cos(theta)
            env.unwrapped.model.opt.gravity[:] = [
                9.81 * np.sin(angle_rad),
                0,
                -9.81 * np.cos(angle_rad)
            ]

            # Network processes observation
            action = net.activate(observation)

            # Step environment with modified gravity
            observation, reward, terminated, truncated, info = env.step(action)
            total_steps += 1
            max_slope = slope_degree

            if terminated or truncated:
                break
    finally:
        env.close()

    return total_steps, max_slope

# ==========================================
# 4. BATCH ABLATION RUNNER WITH BEST GENOME SAVING
# ==========================================
def run_batch_ablation(num_runs=20, save_best=True):
    """
    Run ablation study comparing Oja-enabled vs Oja-disabled evolution.
    Tracks the absolute best genome and saves it to pickle.
    """
    # Load NEAT config
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-hopper')

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found at {config_path}. "
            "Please ensure config-hopper exists in the script directory."
        )

    try:
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )
    except Exception as e:
        print(f"ERROR loading config: {e}")
        raise

    # CSV file setup
    csv_filename = 'hopper_ablation_results.csv'
    results = []
    on_list, off_list = [], []

    # Best genome tracking
    best_genome = None
    best_fitness = float('-inf')
    best_config = None

    try:
        print("\n" + "="*60)
        print("NEAT HOPPER ABLATION STUDY (Oja's Rule)")
        print("="*60)

        for run_idx in range(num_runs):
            print(f"\n--- RUN {run_idx + 1}/{num_runs} ---")

            # ===== EVOLUTION PHASE: Oja Enabled =====
            print(" [1/3] Evolving population (Oja ON)...")
            p_on = neat.Population(config)
            evaluator_on = GenomeEvaluator(use_oja=True)

            try:
                best_genome_on = p_on.run(evaluator_on, 50)  # 50 generations
                print(f" Best fitness (ON): {best_genome_on.fitness:.2f}")

                # Track absolute best genome
                if best_genome_on.fitness > best_fitness:
                    best_fitness = best_genome_on.fitness
                    best_genome = best_genome_on
                    best_config = config
                    print(f" [*] New best genome! Fitness: {best_fitness:.2f}")

            except Exception as e:
                print(f" ERROR during evolution: {e}")
                raise
            finally:
                evaluator_on.close()

            # ===== STRESS TEST: ON vs OFF =====
            print(" [2/3] Stress test (Oja ON)...")
            try:
                on_steps, on_slope = stress_test_genome(
                    best_genome_on, config, use_oja=True, seed=100
                )
                print(f" Steps: {on_steps}, Max Slope: {on_slope}Â°")
            except Exception as e:
                print(f" ERROR in stress test: {e}")
                raise

            print(" [3/3] Stress test (Oja OFF)...")
            try:
                off_steps, off_slope = stress_test_genome(
                    best_genome_on, config, use_oja=False, seed=100
                )
                print(f" Steps: {off_steps}, Max Slope: {off_slope}Â°")
            except Exception as e:
                print(f" ERROR in stress test: {e}")
                raise

            # Record results
            on_list.append(on_steps)
            off_list.append(off_steps)
            results.append([run_idx + 1, on_steps, on_slope, off_steps, off_slope])

            print(f" Result: ON={on_steps} vs OFF={off_steps} " +
                  f"(diff: {on_steps - off_steps:+d} steps)")

        # ===== WRITE RESULTS TO CSV =====
        print(f"\n[*] Writing results to {csv_filename}...")
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Run', 'ON_Steps', 'ON_Slope', 'OFF_Steps', 'OFF_Slope'])
            writer.writerows(results)

        # ===== SAVE BEST GENOME TO PICKLE =====
        if save_best and best_genome is not None:
            pickle_filename = 'best_genome.pickle'
            print(f"\n[*] Saving best genome (fitness={best_fitness:.2f}) to {pickle_filename}...")
            with open(pickle_filename, 'wb') as f:
                pickle.dump({
                    'genome': best_genome,
                    'config': best_config,
                    'fitness': best_fitness
                }, f)
            print(f"[+] Saved to {pickle_filename}")

        # ===== STATISTICAL ANALYSIS =====
        print("\n" + "="*60)
        print("STATISTICAL ANALYSIS")
        print("="*60)

        on_array = np.array(on_list)
        off_array = np.array(off_list)
        diff_array = on_array - off_array

        print(f"Oja ON  - Mean: {np.mean(on_array):7.2f}, Median: {np.median(on_array):7.2f}, " +
              f"Std: {np.std(on_array):7.2f}")
        print(f"Oja OFF - Mean: {np.mean(off_array):7.2f}, Median: {np.median(off_array):7.2f}, " +
              f"Std: {np.std(off_array):7.2f}")
        print(f"Difference - Mean: {np.mean(diff_array):+7.2f}, Median: {np.median(diff_array):+7.2f}")

        # Wilcoxon signed-rank test
        try:
            stat, p_val = wilcoxon(on_list, off_list)
            print(f"Wilcoxon Test - Statistic: {stat:.2f}, p-value: {p_val:.5f}")

            if p_val < 0.05:
                print("*** Significant difference detected (p < 0.05) ***")
            else:
                print(" No significant difference (p >= 0.05)")
        except Exception as e:
            print(f"WARNING: Wilcoxon test failed: {e}")

        print("="*60 + "\n")

    except KeyboardInterrupt:
        print("\n[!] Interrupted by user")
        raise
    except Exception as e:
        print(f"\n[!] Fatal error: {e}")
        raise

# ==========================================
# 5. MAIN ENTRY POINT
# ==========================================
if __name__ == '__main__':
    run_batch_ablation(num_runs=20, save_best=True)