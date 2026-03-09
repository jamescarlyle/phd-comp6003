import math
import random
import copy
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ==========================================
# 1. CARTPOLE PHYSICS ENGINE
# ==========================================
def step_env(state, action, length):
    """Calculates the next CartPole state using Euler integration."""
    cart_position, cart_velocity, pole_angle, pole_velocity = state
    force = 3.0 if action == 1 else -2.0 # was 10
    cos_theta = math.cos(pole_angle)
    sin_theta = math.sin(pole_angle)
    
    # Constants
    gravity = 9.8
    cart_mass = 1.0
    pole_mass = 0.1
    total_mass = cart_mass + pole_mass
    pole_mass_length = pole_mass * length
    
    # Equations of motion
    temp = (force + pole_mass_length * pole_velocity**2 * sin_theta) / total_mass
    pole_acceleration = (gravity * sin_theta - cos_theta * temp) / (length * (4.0/3.0 - pole_mass * cos_theta**2 / total_mass))
    cart_acceleration = temp - pole_mass_length * pole_acceleration * cos_theta / total_mass
    
    tau = 0.02 # Seconds between state updates
    return [cart_position + tau * cart_velocity, cart_velocity + tau * cart_acceleration, pole_angle + tau * pole_velocity, pole_velocity + tau * pole_acceleration]

# ==========================================
# 2. NEAT-LITE (NO CLASSES)
# ==========================================
def new_genome(in_nodes, out_nodes):
    """Initializes a dense feedforward genome using dictionaries."""
    nodes = {i: {'type': 'in'} for i in range(in_nodes)}
    for i in range(out_nodes):
        nodes[in_nodes + i] = {'type': 'out'}
    
    conns = []
    for i in range(in_nodes):
        for j in range(out_nodes):
            conns.append({'in': i, 'out': in_nodes + j, 'w': random.uniform(-1, 1), 'enabled': True, 'innov': len(conns)})
            
    return {'nodes': nodes, 'conns': conns, 'fitness': 0}

def mutate(genome, innov_counter):
    """Mutates weights or augments topology (adds nodes/connections)."""
    r = random.random()
    if r < 0.8: # Weight mutation
        for c in genome['conns']:
            if random.random() < 0.2:
                c['w'] += random.uniform(-0.5, 0.5)
    elif r < 0.9: # Add Node mutation
        active_conns = [c for c in genome['conns'] if c['enabled']]
        if active_conns:
            c = random.choice(active_conns)
            c['enabled'] = False
            new_node = max(genome['nodes'].keys()) + 1
            genome['nodes'][new_node] = {'type': 'hidden'}
            
            # Add connections to/from new node
            genome['conns'].append({'in': c['in'], 'out': new_node, 'w': 1.0, 'enabled': True, 'innov': innov_counter})
            genome['conns'].append({'in': new_node, 'out': c['out'], 'w': c['w'], 'enabled': True, 'innov': innov_counter + 1})
            innov_counter += 2
    else: # Add Connection mutation
        n1 = random.choice(list(genome['nodes'].keys()))
        n2 = random.choice(list(genome['nodes'].keys()))
        if genome['nodes'][n1]['type'] != 'out' and genome['nodes'][n2]['type'] != 'in' and n1 != n2:
            if not any(c['in'] == n1 and c['out'] == n2 for c in genome['conns']):
                genome['conns'].append({'in': n1, 'out': n2, 'w': random.uniform(-1, 1), 'enabled': True, 'innov': innov_counter})
                innov_counter += 1
    return innov_counter

# ==========================================
# 3. NETWORK ACTIVATION & LOCAL LEARNING 
# ==========================================
def activate_and_learn(genome, inputs, rule, eta):
    """Feeds forward and applies Hebbian or Oja's rule locally."""
    acts = {n: 0.0 for n in genome['nodes']}
    for i, val in enumerate(inputs): 
        acts[i] = val
    
    # Feedforward (2 passes for recurrency/hidden nodes)
    order = [n for n, d in genome['nodes'].items() if d['type'] != 'in']
    for _ in range(2):
        new_acts = acts.copy()
        for n in order:
            s = sum(c['w'] * acts[c['in']] for c in genome['conns'] if c['enabled'] and c['out'] == n)
            new_acts[n] = math.tanh(s)
        acts = new_acts

    # Apply Local Learning Mechanisms
    if rule in ['hebbian', 'oja']:
        for c in genome['conns']:
            if c['enabled']:
                x, y, w = acts[c['in']], acts[c['out']], c['w']
                if rule == 'hebbian':
                    # Hebbian: Delta w = eta * x * y
                    c['w'] = max(-5.0, min(5.0, w + eta * x * y))
                elif rule == 'oja':
                    # Oja: Delta w = eta * y * (x - y*w) (Prevents weights from blowing up)
                    c['w'] = max(-5.0, min(5.0, w + eta * y * (x - y * w)))
                    
    return acts, genome

# ==========================================
# 4. SIMULATION ENVIRONMENT
# ==========================================
def simulate(genome, rule, max_steps, render=False):
    state = [0.0, 0.0, 0.0, 0.0]
    fitness = 0
    history = []
    
    # Deepcopy to prevent intra-lifetime learning from permanently overwriting the genetic baseline
    eval_gen = copy.deepcopy(genome) 

    for t in range(max_steps):
        # The pole lengthens dynamically over time to test adaptation
        length = 0.5 + 3*t / max_steps
        
        if render:
            history.append((state.copy(), length))

        # Inputs: cart_position, cart_velocity, pole_angle, pole_velocity, bias (1.0)
        acts, eval_gen = activate_and_learn(eval_gen, state + [1.0], rule, eta=0.02) #0.02
        
        # Determine action (node 5 is the output)
        action = 1 if acts.get(5, 0) > 0 else 0
        state = step_env(state, action, length)
        x, _, theta, _ = state

        # Termination (18 degrees angle, 2.9 units off center)
        if abs(x) > 2.9 or abs(theta) > 0.4:
            break
        fitness += 1

    return (fitness, history) if render else fitness

# ==========================================
# 5. EVOLUTION & ANIMATION PIPELINE
# ==========================================
if __name__ == "__main__":
    # --- Configuration ---
    LEARNING_RULE = 'none'  # Options: 'none', 'hebbian', 'oja'
    POP_SIZE = 40
    GENERATIONS = 20
    
    # Initialize Population
    population = [new_genome(5, 1) for _ in range(POP_SIZE)]
    innov = len(population[0]['conns'])
    best_genome = None
    best_fitness = -1

    print(f"Evolving for {GENERATIONS} generations using {LEARNING_RULE.upper()} plasticity...")
    for gen in range(GENERATIONS):
        for individual in population:
            individual['fitness'] = simulate(individual, LEARNING_RULE, max_steps=800, render=False)

        population.sort(key=lambda x: x['fitness'], reverse=True)
        if population[0]['fitness'] > best_fitness:
            best_fitness = population[0]['fitness']
            best_genome = copy.deepcopy(population[0])

        print(f"Gen {gen+1:02d} | Fitness: {population[0]['fitness']} | Nodes: {len(best_genome['nodes'])} | Connections: {len(best_genome['conns'])}")
        if best_fitness >= 700:
            print("Solved!")
            break

        # Next Generation (Elitism + Mutation)
        next_pop = [copy.deepcopy(population[0]), copy.deepcopy(population[1])]
        while len(next_pop) < POP_SIZE:
            parent = random.choice(population[:15]) # Select from top 15
            child = copy.deepcopy(parent)
            innov = mutate(child, innov)
            next_pop.append(child)
        population = next_pop

    for c in best_genome['conns']:
        print(f"Connection from: {c['in']} to: {c['out']} weight: {c['w']:.2f}")

    print("Generating Matplotlib Animation...")
    fitness, history = simulate(best_genome, LEARNING_RULE, max_steps=1500, render=True)

    # --- Setup Animation ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim(-3, 3)
    ax.set_ylim(-0.5, 3.5)
    ax.set_aspect('equal')
    ax.grid(True)
    
    cart = plt.Rectangle((0, 0), 0.6, 0.4, fc='blue')
    ax.add_patch(cart)
    line, = ax.plot([], [], 'r-', lw=4)
    info_text = ax.text(0.02, 0.83, '', transform=ax.transAxes, fontsize=10, 
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

    def init():
        cart.set_xy((-0.3, -0.2))
        line.set_data([], [])
        info_text.set_text('')
        return cart, line, info_text

    def animate(i):
        state, length = history[i]
        x, _, theta, _ = state
        
        # Update cart and pole geometry
        cart.set_xy((x - 0.3, -0.2))
        line.set_data([x, x + length * math.sin(theta)], [0, length * math.cos(theta)])
        
        # Update stats
        info_text.set_text(f"Step: {i}\nPole Length: {length:.2f}\nRule: {LEARNING_RULE.capitalize()}")
        return cart, line, info_text

    ani = animation.FuncAnimation(fig, animate, frames=len(history), init_func=init, blit=True, interval=20)
    
    # Depending on your environment, you can save or show it:
    # ani.save('cartpole_adaptation.mp4', writer='ffmpeg', fps=30)
    plt.title("CartPole + NEAT with Local Plasticity")
    plt.show()
