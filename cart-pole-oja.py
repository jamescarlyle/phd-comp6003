import math, random                                              # standard library only — no external dependencies
from collections import defaultdict                              # used for adjacency list and in-degree tracking in topo sort

GRAVITY   = 9.8 # gravitational acceleration in m/s²
CART_MASS = 1.0 # mass of the cart in kg
POLE_MASS = 0.1 # mass of the pole in kg
POLE_HALF = 0.5 # half the pole length in metres
FORCE_MAG = 10.0 # magnitude of the left/right push force in Newtons
DT        = 0.02 # Euler integration timestep in seconds
MAX_STEPS = 500 # episode length cap — reaching this means "solved"
# combined mass used in denominator of equations of motion 
total_mass = CART_MASS + POLE_MASS
pole_mass_length = POLE_MASS * POLE_HALF

def step(state, action, slope_angle):
    x_position, x_velocity, pole_angle, pole_angular_velocity = state
    # map binary action to signed force
    force = FORCE_MAG if action == 1 else -FORCE_MAG
    # gravity component pulling cart along the slope
    g_along = GRAVITY * math.sin(slope_angle)
    # gravity component normal to slope (effective local g)
    g_perpendicular = GRAVITY * math.cos(slope_angle)
    # cache cosine of pole angle to avoid recomputing 
    cos_pole_angle = math.cos(pole_angle)
    # cache sine of pole angle to avoid recomputing
    sin_pole_angle = math.sin(pole_angle)
    # intermediate term: net horizontal force on system; subtract slope-induced drift, normalise by total mass
    net_horizontal_force = (force + POLE_MASS * POLE_HALF * pole_angular_velocity**2 * sin_pole_angle - CART_MASS * g_along) / total_mass
    # angular acceleration of pole from equations of motion
    th_acc = (g_perpendicular * sin_pole_angle - cos_pole_angle * net_horizontal_force) / (POLE_HALF * (4/3 - POLE_MASS * cos_pole_angle**2 / total_mass)) 
    # cart linear acceleration after subtracting pole reaction
    cart_linear_accel = net_horizontal_force - POLE_MASS * POLE_HALF * th_acc * cos_pole_angle / total_mass 
    # integrate cart velocity one step forward
    updated_velocity = x_velocity + DT * cart_linear_accel
    # integrate cart position one step forward
    updated_position = x_position + DT * updated_velocity
    # integrate pole angular velocity one step forward
    updated_pole_angular = pole_angular_velocity + DT * th_acc 
    # integrate pole angle one step forward
    updated_pole_angle = pole_angle + DT * updated_pole_angular
    # episode ends if cart leaves track or pole falls past 24° 
    done = abs(updated_position) > 2.4 or abs(updated_pole_angle) > math.radians(24)
    # return new state tuple and termination flag
    return (updated_position, updated_velocity, updated_pole_angle, updated_pole_angular), done

import math

def gemini_step(state, action, slope_angle):
    x_position, x_velocity, pole_angle, pole_angular_velocity = state
    
    # Constants (Assuming standard OpenAI Gym values)
    FORCE_MAG = 10.0
    GRAVITY = 9.8
    CART_MASS = 1.0
    POLE_MASS = 0.1
    POLE_HALF = 0.5
    DT = 0.02
    total_mass = CART_MASS + POLE_MASS
    pole_mass_length = POLE_MASS * POLE_HALF

    # 1. Map action to force
    force = FORCE_MAG if action == 1 else -FORCE_MAG
    
    # 2. Gravity components
    g_sin_slope = GRAVITY * math.sin(slope_angle)
    g_cos_slope = GRAVITY * math.cos(slope_angle)

    # 3. Cache trig for pole
    sin_theta = math.sin(pole_angle)
    cos_theta = math.cos(pole_angle)

    # 4. Corrected Net Horizontal Term
    # We subtract the total system weight pulling down the slope
    temp = (force + pole_mass_length * pole_angular_velocity**2 * sin_theta - total_mass * g_sin_slope) / total_mass

    # 5. Corrected Angular Acceleration
    # The gravity term must account for the slope: g*sin(theta - slope)
    # which expands to: g*sin(theta)*cos(slope) - g*cos(theta)*sin(slope)
    numerator = (g_cos_slope * sin_theta - g_sin_slope * cos_theta) - cos_theta * temp
    denominator = POLE_HALF * (4.0/3.0 - POLE_MASS * cos_theta**2 / total_mass)
    
    th_acc = numerator / denominator

    # 6. Linear Acceleration
    cart_accel = temp - (pole_mass_length * th_acc * cos_theta) / total_mass

    # 7. Integration (Semi-Implicit Euler)
    new_v = x_velocity + DT * cart_accel
    new_x = x_position + DT * new_v
    new_w = pole_angular_velocity + DT * th_acc
    new_th = pole_angle + DT * new_w

    # 8. Termination conditions
    done = abs(new_x) > 2.4 or abs(new_th) > math.radians(24)

    return (new_x, new_v, new_th, new_w), done

# mutable list so inner functions can increment without global
_innovation_counter = [0] 
# maps (source, destination) connections to unique   innovation numbers 
_innovation_map = {} 

# return a stable  innovation number for a given connection key
def new_innovation(from_to):
    # only assign a new number if this connection is seen for the first time
    if from_to not in _innovation_map:
        # increment the global counter
        _innovation_counter[0] += 1
        # record the new  innovation number for this connection
        _innovation_map[from_to] = _innovation_counter[0]
    # return the (possibly existing)  innovation number
    return _innovation_map[from_to]

# create a connection gene as a plain dict
def make_connection(source, destination, w,  innovation, enabled=True):
    # store topology, weight and enabled flag
    # learning_rate is the evolved Oja plasticity learning rate
    return {'source': source, 'destination': destination, 'w': w, 'enabled': enabled, 
            'innovation':  innovation, 'learning_rate': random.uniform(0.0, 0.05)}

# return a shallow copy of a connection dict
def copy_connection(c):
    # all values are scalars so shallow copy is sufficient
    return {**c} 

# create an empty genome as a plain dict
def make_genome():
    # nodes=node ids, connections= innovation->connection, fitness=score
    return {'nodes': [], 'connections': {}, 'fitness': 0.0}

# deep-copy a genome (connections need individual copies)
def copy_genome(g):
    # slice to copy the node id list
    return {'nodes': g['nodes'][:], 'connections': {k: copy_connection(v) for k, v in g['connections'].items()}, 'fitness': g['fitness']} # carry over the current fitness value

# build the starting fully-connected input→output genome
def minimal_genome(n_inputs=4, n_outputs=1): 
    # start with an empty genome dict
    genome = make_genome() 
    # ids: 0..n_inputs-1 = inputs, n_inputs = output
    genome['nodes'] = list(range(n_inputs + n_outputs)) 
    # connect every input node...
    for source in range(n_inputs): 
        # ...to every output node 
        for destination in range(n_inputs, n_inputs + n_outputs): 
            # get or create a stable  innovation number for this connection 
             innovation = new_innovation((source, destination)) 
            # random initial weight from N(0,1) 
        genome['connections'][ innovation] = make_connection(source, destination, random.gauss(0, 1),  innovation) 
    # return the completed minimal genome
    return genome 

# nudge all weights and learning_rates with Gaussian noise
def perturb_weights(genome): 
    for connection in genome['connections'].values():
        # 90% chance: small perturbation 
        if random.random() < 0.9: 
            # add small Gaussian noise to the weight 
            connection['w'] += random.gauss(0, 0.2) 
            # perturb plasticity rate, clamp to non-negative
            connection['learning_rate'] = max(0.0, connection['learning_rate'] + random.gauss(0, 0.005)) 
        else: 
            # 10% chance: full weight reset 
            connection['w'] = random.gauss(0, 1) 

# structural mutation: add a new random connection
def add_connection(genome): 
    # pick a random source node 
    source = random.choice(genome['nodes']) 
    # pick a random destination node 
    destination = random.choice(genome['nodes']) 
    # look up or create the  innovation number for this connection 
    innovation = new_innovation((source, destination)) 
    # only add if this connection does not already exist 
    if  innovation not in genome['connections']: 
        # insert new connection with random weight
        genome['connections'][ innovation] = make_connection(source, destination, random.gauss(0, 1),  innovation) 

# structural mutation: split a connection with a new node
def add_node(genome): 
    # gather all active connections to choose from 
    enabled = [connection for connection in genome['connections'].values() if connection['enabled']] 
    # nothing to split if all connections are disabled
    if not enabled: 
        return 
    # pick a random active connection to split 
    old = random.choice(enabled) 
    # disable the original connection 
    old['enabled'] = False 
    # assign the next available integer node id 
    new_id = max(genome['nodes']) + 1 
    # register the new node in the genome 
    genome['nodes'].append(new_id) 
    #  innovation number for the in-connection to the new node 
    innovation_1 = new_innovation((old['source'], new_id)) 
    #  innovation number for the out-connection from the new node
    innovation_2 = new_innovation((new_id, old['destination'])) 
    # in-weight=1 preserves the original signal
    genome['connections'][ innovation_1] = make_connection(old['source'], new_id, 1.0,  innovation_1) 
    # out-weight=old weight preserves network function
    genome['connections'][ innovation_2] = make_connection(new_id, old['destination'], old['w'],  innovation_2) 

# apply stochastic mutations and return a new genome
def mutate(genome):
    # work on a copy so the parent genome is unchanged
    mutated_genome = copy_genome(genome)
    # single draw gates all three mutation types
    r = random.random()
    # 80% chance: nudge weights and learning_rates
    if r < 0.80: perturb_weights(mutated_genome)
    # 5% chance: add a new random connection
    if r < 0.05: add_connection(mutated_genome)
    # 3% chance: insert a new hidden node
    if r < 0.03: add_node(mutated_genome)
    # return the mutated offspring genome
    return mutated_genome

# produce a child genome by combining two parents
def crossover(parent_1, parent_2):
    # ensure p1 is always the fitter parent
    if parent_2['fitness'] > parent_1['fitness']:
        # swap so matching genes come from the better parent
        parent_1, parent_2 = parent_2, parent_1
    # start with an empty child genome
    child = make_genome()
    # inherit node topology from the fitter parent
    child['nodes'] = parent_1['nodes'][:]
    # iterate over fitter parent's connections
    for  innovation, connection in parent_1['connections'].items():
        # matching gene: 50/50 chance from either parent
        if  innovation in parent_2['connections'] and random.random() < 0.5:
            # take from weaker parent
            child['connections'][ innovation] = copy_connection(parent_2['connections'][ innovation])
        # disjoint gene or coin-flip lost: take from fitter parent
        else:
            # take from fitter parent
            child['connections'][ innovation] = copy_connection(connection)
    # return the completed child genome
    return child

# return nodes in feed-forward activation order (Kahn's algorithm)
def topo_sort(nodes, connections):
    # tracks how many active inputs each node is still waiting for
    in_degree = defaultdict(int)
    # forward adjacency list of enabled connections only
    adjacent = defaultdict(list)
    # build the graph from active connections only
    for connection in connections.values():
        # skip disabled connections
        if connection['enabled']:
            # record the forward connection
            adjacent[connection['source']].append(connection['destination'])
            # destination has one more unresolved incoming edge
            in_degree[connection['destination']] += 1
    # seed queue with source nodes (no incoming edges)
    queue = [node for node in nodes if in_degree[node] == 0]
    # will hold the final sorted node sequence
    order = []
    # process until no more nodes are ready
    while queue:
        # take the next node with no unresolved inputs
        node = queue.pop(0)
        # add it to the activation sequence
        order.append(node)
        # notify each successor
        for adj_node in adjacent[node]:
            # one fewer unresolved input for this successor
            in_degree[adj_node] -= 1
            # successor is now ready to activate
            if in_degree[adj_node] == 0: queue.append(adj_node)
    # append any remaining nodes to handle cycles gracefully
    order += [node for node in nodes if node not in order]
    # return the complete activation order
    return order

# squash any real value into the open interval (0, 1)
def sigmoid(x):
    # clamp input to ±20 to prevent float overflow
    return 1.0 / (1.0 + math.exp(-max(-20, min(20, x))))

# build a runtime network state dict from a genome
def make_net(genome, n_inputs=4):
    # reference to the source genome (not copied),
    # sensor input count, output node id, pre-computed activation order,
    # and plastic weights initialised to the evolved base values
    return {'genome': genome,
            'n_inputs': n_inputs,
            'out_node': n_inputs,
            'order': topo_sort(genome['nodes'], genome['connections']),
            'w_plastic': {k: c['w'] for k, c in genome['connections'].items()}}

# run one forward pass and apply Oja plasticity in-place
def net_forward(net, inputs):
    # shorthand reference to the genome
    genome = net['genome']
    # shorthand for the number of input nodes
    n_inputs = net['n_inputs']
    # shorthand for the mutable plastic weight table
    w_plastic = net['w_plastic']
    # seed activation map with raw sensor values
    activation_map = {i: v for i, v in enumerate(inputs)}
    # activate each node in topological order
    for node in net['order']:
        # input nodes are pre-set from sensor values
        if node < n_inputs:
            # skip — do not overwrite input activations
            continue
        # weighted sum of all incoming activations
        # considering only active edges targeting this node
        total = sum(w_plastic[connection['innovation']] * activation_map.get(connection['source'], 0.0)
                    for connection in genome['connections'].values()
                    if connection['enabled'] and connection['destination'] == node)
        # apply sigmoid to get this node's output activation
        activation_map[node] = sigmoid(total)
    # Oja plasticity update over all connections
    for connection in genome['connections'].values():
        # skip disabled or non-plastic connections
        if not connection['enabled'] or connection['learning_rate'] == 0:
            # move on to the next connection
            continue
        # pre-synaptic (input-side) activation
        x = activation_map.get(connection['source'], 0.0)
        # post-synaptic (output-side) activation
        y = activation_map.get(connection['destination'], 0.0)
        # Oja rule: Δw = η(yx − y²w)
        w_plastic[connection['innovation']] += connection['learning_rate'] * (y * x - y * y * w_plastic[connection['innovation']])
    # threshold output activation to binary left/right action
    return 1 if activation_map.get(net['out_node'], 0.5) >= 0.5 else 0

# run a full episode and return the number of steps survived
def evaluate(genome, slope_angle, seed=42):
    # local RNG for deterministic, reproducible evaluation
    rng = random.Random(seed)
    # small random initial state near the upright equilibrium
    state = tuple(rng.uniform(-0.05, 0.05) for _ in range(4))
    # build a fresh network with reset plastic weights
    net = make_net(genome)
    # run up to the maximum allowed timesteps
    for t in range(MAX_STEPS):
        # choose an action from the current observation
        action = net_forward(net, list(state))
        # advance the physics simulation one step
        state, done = step(state, action, slope_angle)
        # pole fell or cart went out of bounds
        if done:
            # return steps survived (1-indexed)
            return t + 1
    # survived the full episode — maximum possible fitness
    return MAX_STEPS

# NEAT compatibility distance coefficients and speciation threshold
C1, C2, COMPAT_THRESH = 1.0, 0.4, 3.0

# compute NEAT genetic compatibility distance between two genomes
def compatibility_distance(genome_1, genome_2):
    #  innovation number sets for each genome
    genome_1_connections, genome_2_connections = set(genome_1['connections']), set(genome_2['connections'])
    # shared   innovations (matching genes)
    common = genome_1_connections & genome_2_connections
    # count of non-matching (disjoint/excess) genes
    disjoint = len(genome_1_connections.symmetric_difference(genome_2_connections))
    # no shared genes — distance is purely topological
    if not common:
        # scale by C1 and return early
        return disjoint * C1
    # mean absolute weight difference across matching genes
    average_weight = sum(abs(genome_1['connections'][k]['w'] - genome_2['connections'][k]['w']) for k in common) / len(common)
    # weighted sum of structural and weight distances
    return C1 * disjoint / max(len(genome_1_connections), len(genome_2_connections), 1) + C2 * average_weight

# create a new species dict with the given representative genome
def make_species(rep):
    # empty member list, zero stagnation counter
    return {'rep': rep, 'members': [], 'best_fitness': 0.0, 'stagnant': 0}

# apply NEAT fitness sharing — divide each member's fitness by species size
def species_adjust_fitness(species):
    # number of members currently in this species
    n = len(species['members'])
    # iterate over all member genomes
    for genome in species['members']:
        # divide raw fitness by size to penalise large species
        genome['fitness'] /= n

# remove the weakest members, keeping only the top fraction
def species_cull(species, keep=0.4):
    # sort members descending by adjusted fitness
    species['members'].sort(key=lambda g: -g['fitness'])
    # keep top 40%, always at least 1
    species['members'] = species['members'][:max(1, int(len(species['members']) * keep))]

# produce n offspring from this species
def species_breed(species, n):
    # accumulate new child genomes here
    offspring = []
    # produce exactly n children
    for _ in range(n):
        # 75% chance of sexual reproduction when possible
        if len(species['members']) > 1 and random.random() < 0.75:
            # pick two distinct parents uniformly at random
            parent_1, parent_2 = random.sample(species['members'], 2)
            # produce a child by recombination
            child = crossover(parent_1, parent_2)
        # 25% chance (or only one member) — asexual clone
        else:
            # deep-copy a random member
            child = copy_genome(random.choice(species['members']))
        # mutate the child before adding to the offspring pool
        offspring.append(mutate(child))
    # return the list of new offspring genomes
    return offspring

# total genomes in the population each generation
POP_SIZE = 80
# maximum number of generations to run
GENS = 60
# target difficulty — 12-degree slope in radians
MAX_SLOPE = math.radians(12)

# compute the slope angle for a given generation index
def slope_for_gen(generation, total):
    # normalise generation to [0, 1]
    t = generation / max(total - 1, 1)
    # quadratic ramp: grows slowly at first then faster
    return MAX_SLOPE * t * t

# assign each genome in pop to the nearest compatible species
def speciate(pop, species_list):
    # clear member lists before re-assignment
    for species in species_list:
        # empty the list while keeping the representative
        species['members'] = []
    # process every genome in the population
    for genome in pop:
        # flag: has this genome found a species?
        placed = False
        # check each existing species in order
        for species in species_list:
            # close enough to this species' representative?
            if compatibility_distance(genome, species['rep']) < COMPAT_THRESH:
                # assign genome to this species
                species['members'].append(genome)
                # mark as placed
                placed = True
                # stop checking further species
                break
        # no compatible species found for this genome
        if not placed:
            # create a new species with this genome as rep
            ns = make_species(genome)
            # add genome as the founding member
            ns['members'].append(genome)
            # register the new species
            species_list.append(ns)
    # discard empty species (rep changed but nobody followed)
    return [species for species in species_list if species['members']]

# main entry point — runs the full NEAT evolutionary loop
def run_neat():
    # fix random seed for reproducibility
    random.seed(0)
    # initialise population with minimal genomes
    pop = [minimal_genome() for _ in range(POP_SIZE)]
    # bootstrap speciation with a single species
    species_list = speciate(pop, [make_species(pop[0])])
    # will store a copy of the all-time best genome
    best_ever = None
    # all-time best fitness value
    best_fit = 0

    # main generational loop
    for gen in range(GENS):
        # compute this generation's slope angle
        slope = slope_for_gen(gen, GENS)
        # convert to degrees for human-readable logging
        deg = math.degrees(slope)

        # evaluate every genome in the population
        for g in pop:
            # fitness = number of steps survived on current slope
            g['fitness'] = float(evaluate(g, slope))

        # scan for a new all-time best genome
        for g in pop:
            # found something better than before
            if g['fitness'] > best_fit:
                # update the best fitness record
                best_fit = g['fitness']
                # save a deep copy so it is not overwritten by mutation
                best_ever = copy_genome(g)

        # compute mean fitness across the population
        avg = sum(g['fitness'] for g in pop) / len(pop)
        # log generation index, slope, best fitness, average, and species count
        print(f"Gen {gen:3d} | slope={deg:5.2f}° | "
              f"best={int(best_fit):4d} | avg={avg:6.1f} | "
              f"species={len(species_list):3d}")

        # solved: max steps survived on near-full slope
        if best_fit >= MAX_STEPS and deg >= math.degrees(MAX_SLOPE) * 0.95:
            # announce success
            print("✓ Solved on full slope!")
            # stop evolving early
            break

        # per-species processing before reproduction
        for sp in species_list:
            # apply fitness sharing within this species
            species_adjust_fitness(sp)
            # remove the weakest 60% of members
            species_cull(sp)
            # find the fittest surviving member
            top = max(sp['members'], key=lambda g: g['fitness'])
            # did this species improve this generation?
            if top['fitness'] > sp['best_fitness']:
                # update the species-level best fitness record
                sp['best_fitness'] = top['fitness']
                # reset stagnation counter on improvement
                sp['stagnant'] = 0
            # no improvement this generation
            else:
                # increment stagnation counter
                sp['stagnant'] += 1

        # remove species that have been stagnant too long,
        # always preserving at least 2 species
        species_list = [sp for sp in species_list
                        if sp['stagnant'] < 15 or len(species_list) <= 2]

        # sum adjusted fitness across all surviving species,
        # guarding against division by zero
        total_adj = sum(g['fitness'] for sp in species_list
                        for g in sp['members']) or 1e-9

        # accumulate next generation here
        new_pop = []
        # allocate offspring proportional to each species' fitness
        for sp in species_list:
            # total adjusted fitness for this species
            sp_fit = sum(g['fitness'] for g in sp['members'])
            # proportional share, at least 1 offspring
            alloc = max(1, round(POP_SIZE * sp_fit / total_adj))
            # breed the allocated number of offspring
            new_pop.extend(species_breed(sp, alloc))

        # top up if rounding left the population short
        while len(new_pop) < POP_SIZE:
            # clone and mutate a random genome from the old population
            new_pop.append(mutate(copy_genome(random.choice(pop))))
        # trim to exact target population size
        pop = new_pop[:POP_SIZE]

        # refresh species representatives before re-speciation
        for sp in species_list:
            # pick a random surviving member as the new representative
            sp['rep'] = random.choice(sp['members'])
        # assign the new population to species
        species_list = speciate(pop, species_list)

    # section header for results table
    print("\n── Final evaluation of best genome across all slopes ──")
    # test the champion on five slope angles
    for deg in [0, 3, 6, 9, 12]:
        # run one deterministic episode at this slope
        f = evaluate(best_ever, math.radians(deg))
        # scale bar to 40 characters wide for visual clarity
        bar = "█" * int(f / MAX_STEPS * 40)
        # print slope, step count, and ASCII progress bar
        print(f" slope={deg:2d}° steps={f:4d}/{MAX_STEPS} {bar}")

    # total node count in the champion genome
    n_nodes = len(best_ever['nodes'])
    # count only active (enabled) connections
    n_connections = sum(1 for c in best_ever['connections'].values() if c['enabled'])
    # print final network size summary
    print(f"\nBest genome: {n_nodes} nodes, {n_connections} active connections")
    # signal clean completion
    print("Done.")

# only run when executed directly, not when imported
if __name__ == "__main__":
    # launch the full NEAT evolutionary loop
    run_neat()