import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import jax.numpy as jnp
import jax.random as jrandom
import mujoco
from mujoco import mjx

# Ensure CPU only
jax.config.update("jax_platform_name", "cpu")
print("JAX devices:", jax.devices())


# --- Minimal Hopper‑style XML (MuJoCo) ---
XML = """
<mujoco model="hopper">
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.1" />
    <body name="torso" pos="0 0 1.2">
      <joint name="rootx" type="slide" axis="1 0 0"/>
      <joint name="rootz" type="slide" axis="0 0 1"/>
      <joint name="rooty" type="hinge" axis="0 1 0"/>
      <geom name="torso_geom" type="capsule" fromto="0 0 0.2 0 0 0.8"
            size="0.1" rgba="0.8 0.6 0.4 1" />
      <body name="thigh" pos="0 0 0.8">
        <joint name="thigh_joint" axis="0 -1 0" range="-150 0" />
        <geom name="thigh_geom" type="capsule" fromto="0 0 0 0 0 -0.4"
              size="0.08" rgba="0.8 0.4 0.4 1"/>
        <body name="leg" pos="0 0 -0.4">
          <joint name="leg_joint" axis="0 -1 0" range="-150 0" />
          <geom name="leg_geom" type="capsule" fromto="0 0 0 0 0 -0.4"
                size="0.06" rgba="0.6 0.4 0.8 1"/>
          <body name="foot" pos="0 0 -0.4">
            <joint name="foot_joint" axis="0 -1 0" range="-45 45" />
            <geom name="foot_geom" type="capsule" fromto="-0.1 0 -0.1 0.1 0 -0.1"
                  size="0.06" rgba="0.4 0.8 0.4 1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="thigh_motor" gear="200" joint="thigh_joint"/>
    <motor name="leg_motor" gear="200" joint="leg_joint"/>
    <motor name="foot_motor" gear="100" joint="foot_joint"/>
  </actuator>
</mujoco>
"""


# --- MJX Environment (no Brax) ---


class MjxEnv:
    def __init__(self, mjx_model, max_step=1000):
        self.mjx_model = mjx_model
        self.max_step = max_step
        self._step = jax.jit(self._mjx_step)

    @staticmethod
    def _mjx_step(model, data, action):
        data = data.replace(ctrl=action)
        data = mjx.step(model, data)
        return data

    def reset(self, rng):
        del rng  # MJX does not need RNG here

        mjx_data = mjx.make_data(self.mjx_model)

        qpos = mjx_data.qpos
        # Take first 11 components as observation (like Brax Hopper)
        obs = jnp.concatenate([qpos[:11]]).astype(jnp.float32)

        return mjx_data, obs

    def step(self, mjx_data, action):
        mjx_data = self._step(self.mjx_model, mjx_data, action)

        qpos = mjx_data.qpos
        obs = jnp.concatenate([qpos[:11]]).astype(jnp.float32)

        # Hopper‑style reward: move forward + stay alive
        x_vel = qpos[0]
        z_height = qpos[2]
        is_alive = (z_height > 0.7).astype(jnp.float32)

        reward = 0.5 * x_vel + 2.0 * is_alive
        done = (z_height < 0.7).astype(jnp.bool_)

        return mjx_data, obs, reward, done


# --- TensorNEAT RL interface ---


class TensorNEAT_MjxEnv:
    jitable = True

    def __init__(self, mjx_model, max_step=1000):
        self.env = MjxEnv(mjx_model, max_step)

        self.num_inputs = 11
        self.num_outputs = 3
        self.max_step = max_step
        self.input_shape = (self.num_inputs,)

    def setup(self, state):
        return state

    def reset(self, rng):
        return self.env.reset(rng)

    def step(self, mjx_data, action):
        return self.env.step(mjx_data, action)

    def evaluate(self, state, genomes, rng, algorithm_forward, num_rollouts=1):
        """
        state: TensorNEAT pipeline state (can be ignored)
        genomes: (pop_size, ...) genomes batch
        rng: JAX RNGKey (scalar) – split happens OUTSIDE this class
        algorithm_forward: network‑evaluator, signature: (genomes, obs_batch) -> actions
        num_rollouts: ignored for now.

        Returns: (pop_size,) fitness array.
        """
        _ = state, num_rollouts

        def rollout(rng, genome, net_forward_single):
            mjx_data, obs = self.reset(rng)
            total_reward = jnp.zeros((), dtype=jnp.float32)

            for _ in range(self.max_step):
                # Turn scalar obs into (1, 11) for batch eval
                obs_b = obs[None, :]
                actions_b = net_forward_single(genome, obs_b)
                action = jnp.tanh(actions_b[0])  # (1, 3) -> (3,)

                mjx_data, obs, reward, done = self.step(mjx_data, action)
                total_reward = total_reward + reward

                total_reward = jax.lax.cond(
                    done,
                    lambda: total_reward,
                    lambda: total_reward,
                )

            return total_reward

        def net_forward_single(genome, obs_batch):
            # genome: (1,) genome tensor; obs_batch: (B, 11)
            return algorithm_forward(genome[None], obs_batch)

        # Split rng outside or pass pre‑split rngs
        rngs = jrandom.split(rng, len(genomes))

        fitnesses = jax.vmap(rollout, in_axes=(0, 0, None))(rngs, genomes, net_forward_single)
        return fitnesses


# --- Load MJX model ---


m = mujoco.MjModel.from_xml_string(XML)
mjx_model = mjx.put_model(m)


# --- TensorNEAT pipeline ---


from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.genome import DefaultGenome, BiasNode
from tensorneat.common import ACT, AGG


problem = TensorNEAT_MjxEnv(mjx_model, max_step=1000)

pipeline = Pipeline(
    algorithm=NEAT(
        pop_size=1024,
        species_size=20,
        survival_threshold=0.1,
        compatibility_threshold=1.0,
        genome=DefaultGenome(
            num_inputs=problem.num_inputs,
            num_outputs=problem.num_outputs,
            init_hidden_layers=(),
            node_gene=BiasNode(
                activation_options=ACT.tanh,
                aggregation_options=AGG.sum,
            ),
            output_transform=ACT.tanh,
        ),
    ),
    problem=problem,
    seed=42,
    generation_limit=100,
    fitness_target=3200,
)


# --- Run ---


# Initialize state
state = pipeline.setup()


# Run until termination
try:
    state, best = pipeline.auto_run(state)
except Exception as e:
    print("Error during auto_run:", e)
    raise


# Save the best genome
import pickle
with open('best_genome.pkl', 'wb') as f:
    pickle.dump(best, f)

print("Best genome saved to best_genome.pkl")
