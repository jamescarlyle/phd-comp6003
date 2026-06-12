import os
import neat
import numpy as np
import gymnasium as gym
import csv
from scipy.stats import wilcoxon

# ==========================================
# 1. THE MANUAL OJA NETWORK (Pure Python)
# ==========================================
class OjaNetwork:
    def __init__(self, genome, config, eta=0.05):
        self.eta = eta
        self.input_keys = config.genome_config.input_keys
        self.output_keys = config.genome_config.output_keys
        
        # 1. Extract weights into a mutable dictionary
        # Key: (in_node, out_node), Value: weight
        self.weights = {}
        for conn_key, conn in genome.connections.items():
            if conn.enabled:
                self.weights[conn_key] = conn.weight
        
        # 2. Store node properties
        self.nodes = {}
        for node_key, node in genome.nodes.items():
            self.nodes[node_key] = {
                'bias': node.bias,
                'response': node.response,
                'act_func': config.genome_config.activation_defs.get(node.activation)
            }
            
        # 3. Initialize neural activity (recurrent state)
        self.values = {k: 0.0 for k in self.input_keys}
        self.values.update({k: 0.0 for k in self.nodes.keys()})

    def activate(self, inputs):
        # Update input nodes
        for i, val in zip(self.input_keys, inputs):
            self.values[i] = val
            
        # Standard Recurrent Forward Pass
        new_values = self.values.copy()
        for node_key, node_data in self.nodes.items():
            node_input = node_data['bias']
            for (in_k, out_k), weight in self.weights.items():
                if out_k == node_key:
                    node_input += weight * self.values[in_k]
            
            res = node_data['response']
            new_values[node_key] = node_data['act_func'](res * node_input)
            
        # --- OJA'S RULE: Real-time Weight Adaptation ---
        for (in_k, out_k), weight in self.weights.items():
            # Only update connections leading to the output layer for stability
            if out_k in self.output_keys:
                pre_val = self.values[in_k]
                post_val = new_values[out_k]
                
                # Math: dw = eta * (pre*post - w*post^2)
                delta_w = self.eta * (pre_val * post_val - (weight * (post_val**2)))
                self.weights[(in_k, out_k)] += delta_w
                
                # Stability Clamp
                self.weights[(in_k, out_k)] = max(min(self.weights[(in_k, out_k)], 30), -30)

        self.values = new_values
        return [self.values[k] for k in self.output_keys]

# ==========================================
# 2. EVOLUTION PHASE (Flat Ground)
# ==========================================
def eval_genomes(genomes, config, use_oja):
    env = gym.make("Hopper-v5")
    for _, genome in genomes:
        # Create the network based on the test type
        if use_oja:
            net = OjaNetwork(genome, config, eta=0.05)
        else:
            net = neat.nn.RecurrentNetwork.create(genome, config)
            
        observation, _ = env.reset(seed=42)
        env.unwrapped.model.opt.gravity[:] = [0, 0, -9.81] 
        
        fitness = 0.0
        for _ in range(1000):
            action = net.activate(observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            fitness += reward
            if terminated or truncated:
                fitness -= 10.0
                break
        genome.fitness = fitness
    env.close()

# ==========================================
# 3. ABLATION STRESS TEST (Tilting Slope)
# ==========================================
def stress_test_genome(genome, config, use_oja, seed=100):
    env = gym.make("Hopper-v5", max_episode_steps=4000)
    
    if use_oja:
        net = OjaNetwork(genome, config, eta=0.05)
    else:
        net = neat.nn.RecurrentNetwork.create(genome, config)
        
    observation, _ = env.reset(seed=seed)
    total_steps = 0
    
    for step in range(4000): 
        # Tilt: 1 degree every 100 steps
        slope_degree = step // 100 
        angle_rad = np.radians(slope_degree)
        env.unwrapped.model.opt.gravity[:] = [9.81 * np.sin(angle_rad), 0, -9.81 * np.cos(angle_rad)]
        
        action = net.activate(observation)
        observation, reward, terminated, truncated, _ = env.step(action)
        
        total_steps += 1
        if terminated or truncated:
            break
            
    env.close()
    return total_steps, slope_degree

# ==========================================
# 4. MAIN BATCH HARNESS
# ==========================================
def run_batch_ablation(num_runs=20):
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-hopper')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    csv_filename = 'hopper_ablation_results.csv'
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Run', 'ON_Steps', 'ON_Slope', 'OFF_Steps', 'OFF_Slope'])

    on_list, off_list = [], []

    for i in range(num_runs):
        print(f"\n--- RUN {i+1}/{num_runs} ---")
        
        # 1. Evolve Genome (Oja Enabled)
        p = neat.Population(config)
        best = p.run(lambda g, c: eval_genomes(g, c, use_oja=True), 50)
        
        # 2. Test ON vs OFF
        on_steps, on_slope = stress_test_genome(best, config, use_oja=True)
        off_steps, off_slope = stress_test_genome(best, config, use_oja=False)
        
        on_list.append(on_steps)
        off_list.append(off_steps)
        
        with open(csv_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([i+1, on_steps, on_slope, off_steps, off_slope])
        
        print(f"  Result: ON: {on_steps} | OFF: {off_steps}")

    # Statistics
    print("\n" + "="*40)
    print("FINAL STATISTICAL ANALYSIS")
    print("="*40)
    stat, p_val = wilcoxon(on_list, off_list)
    print(f"Average Steps (ON):  {np.mean(on_list):.2f}")
    print(f"Average Steps (OFF): {np.mean(off_list):.2f}")
    print(f"Wilcoxon P-Value:    {p_val:.5f}")

if __name__ == '__main__':
    run_batch_ablation(num_runs=20)