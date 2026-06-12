import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.genome import DefaultGenome, BiasNode
from tensorneat.problem.rl import BraxEnv
from tensorneat.common import ACT, AGG

# Define the pipeline
pipeline = Pipeline(
    algorithm=NEAT(
        pop_size=1024,
        species_size=20,
        survival_threshold=0.1,
        compatibility_threshold=1.0,
        genome=DefaultGenome(
            num_inputs=11,
            num_outputs=3,
            init_hidden_layers=(),
            node_gene=BiasNode(
                activation_options=ACT.tanh,
                aggregation_options=AGG.sum,
            ),
            output_transform=ACT.tanh,
        ),
    ),
    problem=BraxEnv(
        env_name="hopper",
        max_step=1000,
    ),
    seed=42,
    generation_limit=100,
    fitness_target=3200,
)

# Initialize state
state = pipeline.setup()

# Run until termination
state, best = pipeline.auto_run(state)

import pickle

# Save the best genome
with open('best_genome.pkl', 'wb') as f:
    pickle.dump(best, f)

print("Best genome saved to best_genome.pkl")