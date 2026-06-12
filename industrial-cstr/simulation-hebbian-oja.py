import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy

# 1. Chemical and Simulation Parameters
V, F, C_A0, C_C0, T_0 = 1.0, 0.1, 6000.0, 0.0, 300.0
rho, Cp, dH, k0, E_R, UA = 1000.0, 4184.0, -8.5e4, 4.7e9, 8000.0, 8.0e5
t_end, dt, T_set, T_c_base = 50.0, 0.1, 350.0, 300.0

# 2. Base Controllers
def p_controller(T_current, T_setpoint, Tc_base, Kc=1.0):
    Tc = Tc_base + Kc * (T_setpoint - T_current)
    return max(273.0, min(373.0, Tc))

def pid_controller(T_current, T_setpoint, Tc_base, e_int, e_prev, dt, Kc=1.0, Ki=0.5, Kd=0.5):
    e = T_setpoint - T_current
    e_int_new = e_int + e * dt
    e_deriv = (e - e_prev) / dt
    Tc = Tc_base + Kc * e + Ki * e_int_new + Kd * e_deriv
    Tc_clamped = max(273.0, min(373.0, Tc))
    if Tc != Tc_clamped: e_int_new = e_int
    return Tc_clamped, e_int_new, e

# 3. NEAT-Lite Implementation
def create_initial_population(size):
    return [{
        'node_bias': {-1: np.random.randn()}, 
        'connections': {(0, -1): np.random.randn(), (1, -1): np.random.randn()},
        'fitness': -np.inf,
        'next_node': 2
    } for _ in range(size)]

def is_valid_connection(connections, inputs):
    adj = {n: [] for n in set([i for i,o in connections]) | set([o for i,o in connections])}
    in_degree = {n: 0 for n in adj}
    for (i, o) in connections:
        adj[i].append(o); in_degree[o] += 1
    queue = [n for n in adj if in_degree[n] == 0]
    count = 0
    while queue:
        curr = queue.pop(0)
        count += 1
        for child in adj[curr]:
            in_degree[child] -= 1
            if in_degree[child] == 0: queue.append(child)
    return count == len(adj)

def mutate(genome):
    r = np.random.rand()
    if r < 0.2: 
        if not genome['connections']: return
        conn_keys = list(genome['connections'].keys())
        i, o = conn_keys[np.random.randint(len(conn_keys))]
        w = genome['connections'][(i, o)]
        del genome['connections'][(i, o)]
        
        new_node = genome['next_node']
        genome['next_node'] += 1
        genome['node_bias'][new_node] = 0.0
        genome['connections'][(i, new_node)] = 1.0
        genome['connections'][(new_node, o)] = w
    elif r < 0.5: 
        nodes = [0, 1] + list(genome['node_bias'].keys())
        valid_pairs = []
        for i in nodes:
            for o in genome['node_bias'].keys():
                if i != o and (i, o) not in genome['connections']:
                    genome['connections'][(i, o)] = 1.0
                    if is_valid_connection(genome['connections'].keys(), [0, 1]):
                        valid_pairs.append((i, o))
                    del genome['connections'][(i, o)]
        if valid_pairs:
            i, o = valid_pairs[np.random.randint(len(valid_pairs))]
            genome['connections'][(i, o)] = np.random.randn()
    else: 
        for k in genome['connections']:
            if np.random.rand() < 0.8: genome['connections'][k] += np.random.randn() * 0.5
        for k in genome['node_bias']:
            if np.random.rand() < 0.8: genome['node_bias'][k] += np.random.randn() * 0.5

def activate_network(genome, inputs_val, learning_rule=None, eta=0.01):
    connections = list(genome['connections'].keys())
    if not connections:
        return 0.5
    nodes = set([i for i,o in connections]) | set([o for i,o in connections])
    adj = {n: [] for n in nodes}; in_degree = {n: 0 for n in nodes}
    for (i, o) in connections:
        adj[i].append(o); in_degree[o] += 1
    queue = [n for n in nodes if in_degree[n] == 0]
    order = []
    while queue:
        curr = queue.pop(0)
        if curr not in [0, 1]: order.append(curr)
        for child in adj[curr]:
            in_degree[child] -= 1
            if in_degree[child] == 0: queue.append(child)
            
    node_values = {0: inputs_val[0], 1: inputs_val[1]}
    for n in order:
        val = genome['node_bias'].get(n, 0.0)
        for (i, o), w in genome['connections'].items():
            if o == n: val += node_values.get(i, 0.0) * w
        val = np.clip(val, -20, 20)
        node_values[n] = 1.0 / (1.0 + np.exp(-val)) 
        
    output = node_values.get(-1, 0.5)

    # Apply online learning rules
    if learning_rule in ['hebbian', 'oja']:
        for (i, o), w in genome['connections'].items():
            xi = node_values.get(i, 0.0)
            yo = node_values.get(o, 0.0)
            if learning_rule == 'hebbian':
                # Basic Hebbian: dw = eta * xi * yo
                dw = eta * xi * yo
            elif learning_rule == 'oja':
                # Oja's rule: dw = eta * yo * (xi - yo * w)
                dw = eta * yo * (xi - yo * w)
            # Clip weights to prevent mathematical explosions during runtime
            new_w = np.clip(w + dw, -30.0, 30.0)
            genome['connections'][(i, o)] = new_w

    return output

def neat_controller(genome, T_current, T_setpoint, e_prev, dt, learning_rule=None):
    e = T_setpoint - T_current
    e_deriv = (e - e_prev) / dt
    output = activate_network(genome, [e / 50.0, e_deriv / 10.0], learning_rule) 
    return 273.0 + output * 100.0, e 

# 4. Simulation Function
def run_simulation(controller_type, genome=None):
    # Pass genome as copy if running learning variant so we don't permanently modify best_genome_overall
    if genome is not None:
        genome = copy.deepcopy(genome)
        
    learning_rule = None
    if controller_type == 'NEAT-Hebbian': learning_rule = 'hebbian'
    elif controller_type == 'NEAT-Oja': learning_rule = 'oja'

    t_steps = int(t_end / dt) + 1
    t = np.linspace(0, t_end, t_steps)

    C_A, C_C, T, T_c = np.zeros(t_steps), np.zeros(t_steps), np.zeros(t_steps), np.zeros(t_steps)
    C_A[0], C_C[0], T[0] = C_A0, C_C0, T_0
    
    e_int, e_prev = 0.0, T_set - T_0

    if controller_type == 'P': T_c[0] = p_controller(T[0], T_set, T_c_base)
    elif controller_type == 'PID': T_c[0], e_int, e_prev = pid_controller(T[0], T_set, T_c_base, e_int, e_prev, dt)
    else: T_c[0], e_prev = neat_controller(genome, T[0], T_set, e_prev, dt, learning_rule)

    fitness_penalty = 0.0

    for i in range(1, t_steps):
        CA_prev, CC_prev, T_prev = C_A[i-1], C_C[i-1], T[i-1]
        
        if T_prev > 1000 or T_prev < 200:
            fitness_penalty += 100000
            T_prev = max(200.0, min(1000.0, T_prev))

        if controller_type == 'P': Tc_current = p_controller(T_prev, T_set, T_c_base)
        elif controller_type == 'PID': Tc_current, e_int, e_prev = pid_controller(T_prev, T_set, T_c_base, e_int, e_prev, dt)
        else: Tc_current, e_prev = neat_controller(genome, T_prev, T_set, e_prev, dt, learning_rule)
            
        T_c[i-1] = Tc_current
        
        time_elapsed = t[i-1]
        T_0_current = T_0 + time_elapsed / 8.0
    
        rate = k0 * np.exp(-E_R / T_prev) * CA_prev
        dCA_dt = (F / V) * (C_A0 - CA_prev) - rate
        dCC_dt = (F / V) * (C_C0 - CC_prev) + rate
        dT_dt = (F / V) * (T_0_current - T_prev) + (-dH / (rho * Cp)) * rate - (UA / (V * rho * Cp)) * (T_prev - Tc_current)
        
        C_A[i] = CA_prev + dCA_dt * dt
        C_C[i] = CC_prev + dCC_dt * dt
        T[i] = T_prev + dT_dt * dt

    if controller_type == 'P': T_c[-1] = p_controller(T[-1], T_set, T_c_base)
    elif controller_type == 'PID': T_c[-1], _, _ = pid_controller(T[-1], T_set, T_c_base, e_int, e_prev, dt)
    else: T_c[-1], _ = neat_controller(genome, T[-1], T_set, e_prev, dt, learning_rule)

    fitness = -np.sum(np.abs(T - T_set)) * dt - fitness_penalty
    return t, C_A, C_C, T, T_c, fitness

# 5. Execute Fast NEAT Evolution
pop_size = 15
pop = create_initial_population(pop_size)
best_genome_overall = None
max_fitness = -np.inf

# Reduced generations for faster execution to prevent timeout
print("Starting NEAT Evolution (20 Generations)...")
for gen in range(20):
    for genome in pop:
        with np.errstate(over='ignore', invalid='ignore'):
            _, _, _, _, _, fitness = run_simulation('NEAT', genome)
        genome['fitness'] = fitness if not np.isnan(fitness) else -1e9
        
    pop.sort(key=lambda g: g['fitness'], reverse=True)
    best = pop[0]
    if best['fitness'] > max_fitness:
        max_fitness = best['fitness']
        best_genome_overall = copy.deepcopy(best)

    weights = list(best['connections'].values())
    max_w, min_w = (max(weights), min(weights)) if weights else (0.0, 0.0)
    print(f"Gen {gen:2d} | Best Fitness: {best['fitness']:-9.2f} | Max W: {max_w:5.2f} | Min W: {min_w:5.2f}")
        
    survivors = pop[:3]
    new_pop = []
    while len(new_pop) < pop_size:
        child = copy.deepcopy(survivors[np.random.randint(len(survivors))])
        mutate(child)
        new_pop.append(child)
    pop = new_pop

# 6. Run and Plot All Controllers
t, _, CC_P, T_P, Tc_P, _ = run_simulation('P')
_, _, CC_PID, T_PID, Tc_PID, _ = run_simulation('PID')
_, _, CC_NEAT, T_NEAT, Tc_NEAT, _ = run_simulation('NEAT', best_genome_overall)
_, _, CC_HEB, T_HEB, Tc_HEB, _ = run_simulation('NEAT-Hebbian', best_genome_overall)
_, _, CC_OJA, T_OJA, Tc_OJA, _ = run_simulation('NEAT-Oja', best_genome_overall)

# Plotting
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
fig.subplots_adjust(right=0.75)

lines = [
    ('P-Controller', T_P, CC_P, Tc_P, 'red'),
    ('PID-Controller', T_PID, CC_PID, Tc_PID, 'darkorange'),
    ('NEAT (Static)', T_NEAT, CC_NEAT, Tc_NEAT, 'blue'),
    ('NEAT + Hebbian', T_HEB, CC_HEB, Tc_HEB, 'magenta'),
    ('NEAT + Oja', T_OJA, CC_OJA, Tc_OJA, 'lime')
]

for label, t_val, cc_val, tc_val, color in lines:
    ax1.plot(t, t_val, label=label, color=color, linewidth=2, alpha=0.8)
    ax2.plot(t, cc_val, label=label, color=color, linewidth=2, alpha=0.8)
    ax3.plot(t, tc_val, label=label, color=color, linewidth=2, alpha=0.8)

ax1.axhline(T_set, label='Setpoint', color='black', linestyle='--')
ax1.set_ylabel('Temperature (K)')
ax1.set_title('Reactor Temperature')
ax1.grid(True)
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

ax2.set_ylabel('Concentration (mol/m^3)')
ax2.set_title('Product Concentration (Propylene Glycol)')
ax2.grid(True)

ax3.axhline(373.0, color='gray', linestyle=':')
ax3.axhline(273.0, color='gray', linestyle=':')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Temperature (K)')
ax3.set_title('Cooling Jacket Temperature')
ax3.grid(True)

plt.show()
