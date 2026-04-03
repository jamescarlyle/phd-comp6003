import os
import neat
import numpy as np
import gymnasium as gym

# ==========================================
# 1. THE CUSTOM OJA NETWORK
# ==========================================
class OjaNetwork(neat.nn.RecurrentNetwork):
    def __init__(self, inputs, outputs, node_evals, output_keys, eta=0.005):
        super().__init__(inputs, outputs, node_evals)
        self.eta = eta
        self.output_keys = set(output_keys) # Fast lookup for final layer

def activate(self, inputs):
        # 1. Standard Forward Pass
        outputs = super().activate(inputs)
        
        # 2. Oja's Rule (RESTRICTED TO FINAL LAYER)
        for i, (node, act_func, agg_func, bias, response, links) in enumerate(self.node_evals):
            if node in self.output_keys:
                post_val = self.values[node]
                new_links = []
                
                for input_node, weight in links:
                    # FIX: Safely resolve the index for both Hidden and Input nodes
                    if input_node in self.input_indices:
                        pre_idx = self.input_indices[input_node]
                    else:
                        # For hidden/output nodes, the key is usually the index itself
                        pre_idx = input_node
                    
                    # Wrap in try/except just in case NEAT's internal indexing shifts
                    try:
                        pre_val = self.values[pre_idx]
                        
                        # Oja's Math
                        delta_w = self.eta * (pre_val * post_val - (post_val**2 * weight))
                        new_links.append((input_node, weight + delta_w))
                    except IndexError:
                        # Fallback: if index fails, don't update this specific weight
                        new_links.append((input_node, weight))
                
                self.node_evals[i] = (node, act_func, agg_func, bias, response, new_links)
                
        return outputs

# ==========================================
# 2. EVOLUTION PHASE (Flat Ground)
# ==========================================
def eval_genomes(genomes, config, use_oja):
    """Evaluates the population on FLAT ground to evolve the base walking gait."""
    env = gym.make("Hopper-v5", max_episode_steps=2000)
    
    for genome_id, genome in genomes:
        
        base_net = neat.nn.RecurrentNetwork.create(genome, config)
        if use_oja:
            net = OjaNetwork(config.genome_config.input_keys, 
                             config.genome_config.output_keys, 
                             base_net.node_evals, 
                             config.genome_config.output_keys, 
                             eta=0.005)
        else:
            net = base_net
            
        observation, _ = env.reset(seed=42)
        env.unwrapped.model.opt.gravity[:] = [0, 0, -9.81] 
        
        fitness = 0.0
        for _ in range(2000):
            action = net.activate(observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            fitness += reward
            
            if terminated or truncated:
                break
                
        genome.fitness = fitness
    env.close()

def stress_test_genome(genome, config, use_oja, seed=100):
    """Tests a single genome on a gradually tilting slope to measure adaptability."""
    env = gym.make("Hopper-v5", max_episode_steps=2000)
    
    base_net = neat.nn.RecurrentNetwork.create(genome, config)
    if use_oja:
        net = OjaNetwork(config.genome_config.input_keys, 
                         config.genome_config.output_keys, 
                         base_net.node_evals, 
                         config.genome_config.output_keys, 
                         eta=0.005)
    else:
        net = base_net
        
    observation, _ = env.reset(seed=seed)
    total_steps = 0
    fitness = 0.0
    
    for step in range(2000): 
        # Tilt logic
        slope_degree = step // 1000 
        angle_rad = np.radians(slope_degree)
        env.unwrapped.model.opt.gravity[:] = [9.81 * np.sin(angle_rad), 0, -9.81 * np.cos(angle_rad)]
        
        action = net.activate(observation)
        observation, reward, terminated, truncated, _ = env.step(action)
        
        total_steps += 1
        fitness += reward
        
        if terminated or truncated:
            fitness -= 10.0
            break
            
    env.close()
    return total_steps, fitness, slope_degree

# ==========================================
# 4. MAIN EXPERIMENT RUNNER
# ==========================================
def run_experiment():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-hopper')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    generations = 50 # Keep it short for testing, increase for real results

    print("\n=== RUNNING CONTROL (STANDARD NEAT) ===")
    p_control = neat.Population(config)
    p_control.add_reporter(neat.StdOutReporter(True))
    # Using lambda to pass the 'use_oja' flag
    best_control = p_control.run(lambda g, c: eval_genomes(g, c, use_oja=False), generations)

    print("\n=== RUNNING EXPERIMENT (OJA NEAT) ===")
    p_oja = neat.Population(config)
    p_oja.add_reporter(neat.StdOutReporter(True))
    best_oja = p_oja.run(lambda g, c: eval_genomes(g, c, use_oja=True), generations)

    print("\n==============================================")
    print("=== HYPOTHESIS TEST: GRADUAL SLOPE =========")
    print("==============================================")
    
    # We test both best genomes on 5 different random seeds to be sure
    test_seeds = [10, 20, 30, 40, 50]
    
    control_steps = []
    oja_steps = []

    for seed in test_seeds:
        # Unpack as: steps, fitness, slope
        c_steps, c_fit, c_slope = stress_test_genome(best_control, config, use_oja=False, seed=seed)
        o_steps, o_fit, o_slope = stress_test_genome(best_oja, config, use_oja=True, seed=seed)
        
        control_steps.append(c_steps)
        oja_steps.append(o_steps)
        
        print(f"Seed {seed}:")
        print(f"  Control -> Fitness: {c_fit:.2f} | Survived {c_steps} steps (Slope: {c_slope} deg)")
        print(f"  Oja     -> Fitness: {o_fit:.2f} | Survived {o_steps} steps (Slope: {o_slope} deg)")
    
    print("\n=== FINAL RESULTS (AVERAGE SURVIVAL) ===")
    print(f"Control Average Steps: {np.mean(control_steps)}")
    print(f"Oja Average Steps:     {np.mean(oja_steps)}")
    
    if np.mean(oja_steps) > np.mean(control_steps):
        print("CONCLUSION: Hypothesis Supported. Oja adaptation improved slope survival.")
    else:
        print("CONCLUSION: Hypothesis Rejected (or needs tuning). Static weights won.")

if __name__ == '__main__':
    run_experiment()