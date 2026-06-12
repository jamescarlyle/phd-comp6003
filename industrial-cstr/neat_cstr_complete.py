#!/usr/bin/env python3
"""
NEAT Evolution + CSTR Control with Online Hebbian Learning
Complete single-file implementation for process control research

Usage:
    python neat_cstr_complete.py [--generations 50] [--mode evolve|test|compare]

This file contains:
    - CSTR simulator with corrected heat dynamics
    - NEAT evolution wrapper
    - P-control baseline with proper scaling
    - Results comparison
"""

import numpy as np
import pickle
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Optional
import neat


# ============================================================================
# CSTR SIMULATION MODEL
# ============================================================================

@dataclass
class CSTRParams:
    """CSTR physical parameters"""
    V: float = 1.0              # Volume (L) = 0.001 mÂ³
    rho: float = 1000.0         # Density (kg/mÂ³)
    cp: float = 4186.0          # Heat capacity (J/kgÂ·K)
    Tin: float = 300.0          # Inlet temperature (K)
    Tref: float = 350.0         # Reference/setpoint (K)
    dt: float = 0.1             # Time step (seconds)
    Q_max: float = 10000.0      # Max heat input (W)


class CSTRSimulator:
    """Continuous Stirred Tank Reactor with proper heat balance"""
    
    def __init__(self, params: Optional[CSTRParams] = None):
        self.p = params or CSTRParams()
        self.T = self.p.Tin
        self.alpha = 100.0  # Initial heat transfer coefficient (W/K)
        self.alpha0 = 100.0
    
    def fouling_model(self, t: float) -> float:
        """Linear fouling degradation - reduces alpha over time"""
        t_start = 6000  # seconds
        fouling_rate = 0.001  # per second
        if t >= t_start:
            return max(10.0, self.alpha0 - fouling_rate * (t - t_start))
        return self.alpha0
    
    def dynamics(self, T: float, Q: float, t: float) -> float:
        """
        CSTR temperature dynamics with correct heat balance
        
        dT/dt = (1/(rho*V*cp)) * [F*rho*cp*(Tin-T) + Q - alpha*(T-Tamb)]
        
        Simplified with no inlet flow:
        dT/dt = (1/(rho*V*cp)) * [Q - alpha*(T-Tamb)]
        
        Args:
            T: Current temperature (K)
            Q: Heat input (W)
            t: Current time (s)
            
        Returns:
            dT/dt (K/s)
        """
        self.alpha = self.fouling_model(t)
        Tamb = 300.0  # Ambient temperature
        
        # Heat balance: m*cp*dT/dt = Q - alpha*(T - Tamb)
        # where m = rho*V
        mass = self.p.rho * self.p.V  # kg
        dT = (Q - self.alpha * (T - Tamb)) / (mass * self.p.cp)
        
        return dT
    
    def step(self, Q: float, t: float) -> Tuple[float, float]:
        """Simulate one time step with Euler method"""
        # Limit heat input
        Q = np.clip(Q, 0, self.p.Q_max)
        
        # Simple Euler integration
        dT = self.dynamics(self.T, Q, t)
        self.T = self.T + self.p.dt * dT
        
        # Saturate temperature
        self.T = np.clip(self.T, 250, 400)
        
        error = self.T - self.p.Tref
        return self.T, error
    
    def reset(self):
        """Reset to initial condition"""
        self.T = self.p.Tin
        self.alpha = self.alpha0


class SimpleNeuralNetwork:
    """Simple fully-connected network for direct weight access"""
    
    def __init__(self, input_size: int = 3, hidden_size: int = 4, output_size: int = 1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Weights - initialize small
        self.W_ih = np.random.randn(hidden_size, input_size) * 0.1
        self.W_ho = np.random.randn(output_size, hidden_size) * 0.1
        self.b_h = np.zeros(hidden_size)
        self.b_o = np.zeros(output_size)
    
    def forward(self, x: np.ndarray) -> float:
        """Forward pass - returns output in [-1, 1]"""
        h = np.tanh(self.W_ih @ x + self.b_h)
        o = self.W_ho @ h + self.b_o
        return float(np.tanh(o[0]))
    
    def hebbian_update(self, x: np.ndarray, h: np.ndarray, post: float, eta: float = 0.01):
        """Online Hebbian learning: dw = Î· * pre * post"""
        # Update output layer: dW_ho = Î· * post * h
        dW_ho = eta * post * h.reshape(1, -1)
        self.W_ho += dW_ho
        
        # Update input layer: dW_ih â‰ˆ Î· * post * dh/dx * x
        # where dh/dx â‰ˆ (1-hÂ²) for tanh
        dh_dx = (1 - h**2)  # Shape: (hidden_size,)
        dW_ih = eta * post * np.outer(dh_dx, x)  # Shape: (hidden_size, input_size)
        self.W_ih += dW_ih
    
    def oja_update(self, x: np.ndarray, h: np.ndarray, post: float, eta: float = 0.01):
        """Online Oja learning: dw = Î· * (pre*post - postÂ²*w)"""
        # Update output layer
        dW_ho = eta * (post * h.reshape(1, -1) - post**2 * self.W_ho)
        self.W_ho += dW_ho
        
        # Update input layer (simplified)
        dh_dx = (1 - h**2)
        dW_ih = eta * (post * np.outer(dh_dx, x) - post**2 * self.W_ih)
        self.W_ih += dW_ih


# ============================================================================
# CONTROL ALGORITHMS
# ============================================================================

class PController:
    """Proportional controller baseline"""
    
    def __init__(self, Kp: float = 500.0):
        self.Kp = Kp
    
    def control(self, state: Tuple[float, float, float]) -> float:
        """P-control: Q = Kp * error
        
        Converts temperature error to heat input (Watts)
        Kp=500 means 1K error -> 500W heat
        """
        T, Tref, error = state
        # P-control: error > 0 means T < setpoint, need more heat
        Q = self.Kp * error
        return np.clip(Q, 0, 10000)


class NeuralNetworkController:
    """Neural network controller with optional online learning"""
    
    def __init__(self, network, learning_rule: str = 'none', learning_rate: float = 0.001):
        self.net = network
        self.learning_rule = learning_rule
        self.eta = learning_rate
    
    def control(self, state: Tuple[float, float, float], learn: bool = False) -> float:
        """
        Neural network control with optional learning
        
        Args:
            state: (T, Tref, error)
            learn: Whether to apply online learning update
            
        Returns:
            Control signal (0-10000 Watts)
        """
        x = np.array(state, dtype=np.float32)
        
        # Forward pass
        if isinstance(self.net, SimpleNeuralNetwork):
            # Get hidden activations for learning
            h = np.tanh(self.net.W_ih @ x + self.net.b_h)
            o = self.net.W_ho @ h + self.net.b_o
            u = np.tanh(o[0])
            
            # Online learning
            if learn:
                if self.learning_rule == 'hebbian':
                    self.net.hebbian_update(x, h, u, self.eta)
                elif self.learning_rule == 'oja':
                    self.net.oja_update(x, h, u, self.eta)
        else:
            # NEAT network
            u = self.net.activate(state)[0]
        
        # Convert [-1, 1] to [0, 10000] Watts
        return np.clip((u + 1) * 5000, 0, 10000)


# ============================================================================
# SIMULATION RUNNER
# ============================================================================

def run_simulation(
    controller: Callable,
    sim_time: float = 100,
    learning_rule: str = 'none',
    verbose: bool = False
) -> float:
    """
    Run CSTR simulation with given controller
    
    Args:
        controller: Control function
        sim_time: Simulation time (seconds)
        learning_rule: 'hebbian', 'oja', or 'none'
        verbose: Print progress
        
    Returns:
        Integral squared error (cost)
    """
    
    params = CSTRParams()
    sim = CSTRSimulator(params)
    
    cost = 0.0
    n_steps = int(sim_time / params.dt)
    
    for step in range(n_steps):
        t = step * params.dt
        
        # Get current state
        error = sim.T - params.Tref
        state = (sim.T, params.Tref, error)
        
        # Control action
        Q = controller(state)
        
        # Simulate
        T_new, err = sim.step(Q, t)
        
        # Accumulate cost (squared error)
        cost += err**2 * params.dt
        
        if verbose and step % 100 == 0:
            print(f"  t={t:.1f}s, T={sim.T:.2f}K, e={error:.2f}K, Q={Q:.1f}W, cost_so_far={cost:.1f}")
    
    return cost


# ============================================================================
# NEAT EVOLUTION
# ============================================================================

NEAT_CONFIG = """
[NEAT]
fitness_criterion     = max
fitness_threshold     = 0.0
pop_size              = 30
generation_timeout    = 300

[DefaultGenome]
activation_default      = tanh
activation_mutate_rate  = 0.0
activation_options      = tanh
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum
bias_attr_mutation_type = gaussian
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5
conn_add_prob           = 0.5
conn_delete_prob        = 0.1
connection_init_mean    = 0.0
connection_init_stdev   = 1.0
connection_max_value    = 30
connection_min_value    = -30
connection_mutate_power = 0.5
connection_mutate_rate  = 0.7
connection_replace_rate = 0.1
enabled_default         = True
enabled_mutate_rate     = 0.01
feed_forward            = False
initial_connection      = partial_direct
node_add_prob           = 0.2
node_delete_prob        = 0.05
num_hidden              = 0
num_inputs              = 3
num_outputs             = 1
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
"""


class NEATEvolver:
    """NEAT evolution for CSTR control"""
    
    def __init__(self, learning_rule: str = 'hebbian', learning_rate: float = 0.001):
        self.learning_rule = learning_rule
        self.learning_rate = learning_rate
        self.best_ever = float('inf')
        self.generation = 0
    
    def evaluate_genome(self, genome, config) -> float:
        """Evaluate genome fitness"""
        try:
            net = neat.nn.RecurrentNetwork.create(genome, config)
            
            controller = NeuralNetworkController(
                net,
                learning_rule=self.learning_rule,
                learning_rate=self.learning_rate
            )
            
            cost = run_simulation(
                controller.control,
                sim_time=100,
                learning_rule=self.learning_rule
            )
            
            if cost < self.best_ever:
                self.best_ever = cost
                print(f"    â˜… New best: {cost:.4f}")
            
            return -cost  # Negative because NEAT maximizes
        except Exception as e:
            return -1000.0
    
    def eval_genomes(self, genomes, config):
        """Evaluate population"""
        print(f"\n  Generation {self.generation}: Evaluating {len(genomes)} genomes...")
        self.generation += 1
        
        fitness_vals = []
        for gid, genome in genomes:
            fitness = self.evaluate_genome(genome, config)
            genome.fitness = fitness
            fitness_vals.append(fitness)
        
        best = max(fitness_vals)
        avg = np.mean(fitness_vals)
        print(f"  Stats: best={-best:.4f}, avg={-avg:.4f}")
    
    def run(self, generations: int = 50) -> Tuple:
        """Run evolution"""
        # Write config to temp file
        with open('_neat_config_temp.txt', 'w') as f:
            f.write(NEAT_CONFIG)
        
        config = neat.config.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            '_neat_config_temp.txt'
        )
        
        p = neat.Population(config)
        p.reporters.add(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.reporters.add(stats)
        
        print(f"Starting NEAT evolution ({generations} generations, {self.learning_rule} learning)\n")
        winner = p.run(self.eval_genomes, generations)
        
        print(f"\nBest cost achieved: {self.best_ever:.4f}")
        return winner, stats


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def compare_controllers():
    """Compare all controller types"""
    
    print("\n" + "="*70)
    print("CSTR CONTROL COMPARISON: P-Control vs Neural Networks")
    print("="*70)
    
    results = {}
    
    # 1. P-Control baseline
    print("\n1. P-Control (Kp=500)")
    p_ctrl = PController(Kp=500)
    cost_p = run_simulation(p_ctrl.control, sim_time=100)
    results['P-Control'] = cost_p
    print(f"   Cost: {cost_p:.4f}")
    
    # 2. Fixed neural network
    print("\n2. Neural Network (Fixed Weights)")
    nn = SimpleNeuralNetwork()
    nn_ctrl = NeuralNetworkController(nn, learning_rule='none')
    cost_nn = run_simulation(nn_ctrl.control, sim_time=100)
    results['NN Fixed'] = cost_nn
    print(f"   Cost: {cost_nn:.4f}")
    
    # 3. NN with Hebbian learning
    print("\n3. Neural Network with Hebbian Learning")
    nn_hebb = SimpleNeuralNetwork()
    ctrl_hebb = NeuralNetworkController(nn_hebb, learning_rule='hebbian', learning_rate=0.001)
    cost_hebb = run_simulation(
        lambda state: ctrl_hebb.control(state, learn=True),
        sim_time=100,
        learning_rule='hebbian'
    )
    results['NN Hebbian'] = cost_hebb
    print(f"   Cost: {cost_hebb:.4f}")
    
    # 4. NN with Oja learning
    print("\n4. Neural Network with Oja Learning")
    nn_oja = SimpleNeuralNetwork()
    ctrl_oja = NeuralNetworkController(nn_oja, learning_rule='oja', learning_rate=0.001)
    cost_oja = run_simulation(
        lambda state: ctrl_oja.control(state, learn=True),
        sim_time=100,
        learning_rule='oja'
    )
    results['NN Oja'] = cost_oja
    print(f"   Cost: {cost_oja:.4f}")
    
    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    for name, cost in sorted(results.items(), key=lambda x: x[1]):
        if cost_p > 0:
            improvement = (cost_p - cost) / cost_p * 100
            print(f"{name:20s}: {cost:12.4f} ({improvement:+6.1f}% vs P-control)")
        else:
            print(f"{name:20s}: {cost:12.4f}")
    
    return results


def evolve_neat(generations: int = 50):
    """Run NEAT evolution"""
    print("\n" + "="*70)
    print("NEAT Evolution with Hebbian Learning")
    print("="*70)
    
    evolver = NEATEvolver(learning_rule='hebbian', learning_rate=0.001)
    winner, stats = evolver.run(generations=generations)
    
    # Save result
    with open('winner_neat.pkl', 'wb') as f:
        pickle.dump(winner, f)
    print(f"\nWinner saved to winner_neat.pkl")
    
    return winner


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='NEAT CSTR Control')
    parser.add_argument('--mode', default='compare', 
                       choices=['compare', 'evolve', 'test'],
                       help='Execution mode')
    parser.add_argument('--generations', type=int, default=50,
                       help='NEAT generations (for evolve mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'compare':
        compare_controllers()
    elif args.mode == 'evolve':
        evolve_neat(generations=args.generations)
    elif args.mode == 'test':
        print("Running quick test...")
        p_ctrl = PController(Kp=500)
        cost = run_simulation(p_ctrl.control, sim_time=100, verbose=True)
        print(f"\nP-Control cost: {cost:.4f}")


if __name__ == '__main__':
    main()