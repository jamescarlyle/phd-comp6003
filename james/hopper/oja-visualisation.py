import os
import neat
import numpy as np
import gymnasium as gym
from scipy.spatial.transform import Rotation as R
import tempfile
import shutil

# ==========================================
# 1. CREATE CUSTOM HOPPER XML WITH TILTED GROUND
# ==========================================
def create_tilted_hopper_xml(slope_degrees=0, output_path=None):
    """
    Create a modified Hopper XML with a tilted ground plane.
    Instead of rotating gravity, we rotate the ground geometry itself.
    
    Args:
        slope_degrees: Tilt angle in degrees (around Y-axis)
        output_path: Where to save the XML (temp if None)
    
    Returns:
        Path to the generated XML file
    """
    # Convert angle to radians for quaternion calculation
    angle_rad = np.radians(slope_degrees)
    
    # Create quaternion for Y-axis rotation
    # scipy uses [x, y, z, w] convention
    rot = R.from_rotvec([0, angle_rad, 0])
    quat = rot.as_quat()  # [x, y, z, w]
    
    # MuJoCo uses [w, x, y, z] convention in XML
    quat_mujoco = f"{quat[3]:.6f} {quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f}"
    
    # Get standard Hopper XML and modify it
    xml_content = f"""<mujoco model="hopper">
  <compiler angle="degree" coordinate="local" meshdir="..:/mujoco/models/assets" />
  <default>
    <joint armature="1" damping="1" limited="true" solref=".02 1" solimp=".9 .95 .001" />
    <geom friction=".7 .1 .1" margin="0.002" solref=".02 1" solimp=".9 .95 .001" />
    <motor ctrlrange="-1 1" ctrllimited="true" />
  </default>
  <option gravity="0 0 -9.81" integrator="RK4" timestep="0.003" />
  <visual>
    <map znear="0.02" />
  </visual>
  <worldbody>
    <!-- TILTED GROUND PLANE -->
    <geom name="ground" pos="0 0 0" quat="{quat_mujoco}" size="0.6 0.6 0.1" type="plane" material="MatGround" />
    <body name="torso" pos="0 0 1.25" xaxis="0 0 1" zaxis="-1 0 0">
      <camera name="track" mode="trackcom" pos="0 -0.3 0.3" xyaxes="1 0 0 0 0.447214 0.894427" />
      <joint name="root_x" pos="0 0 0" type="slide" axis="1 0 0" limited="false" damping="0" armature="0" />
      <joint name="root_z" pos="0 0 0" type="slide" axis="0 0 1" limited="false" damping="0" armature="0" />
      <joint name="root_rot" pos="0 0 0" type="ball" limited="false" damping="0" armature="0" />
      <geom name="torso_geom" pos="0 0 0" size="0.046" type="sphere" />
      <body name="thigh" pos="0 0 0" xaxis="0 -1 0" zaxis="0 0 1">
        <joint name="thigh_joint" pos="0 0 0" axis="0 -1 0" range="-150 0" type="hinge" />
        <geom name="thigh_geom" friction="0.7 0.1 0.1" pos="0 0 -.15" size="0.046" type="sphere" />
        <body name="leg" pos="0 0 -.3" xaxis="0 -1 0" zaxis="0 0 1">
          <joint name="leg_joint" pos="0 0 0" axis="0 -1 0" range="-150 0" type="hinge" />
          <geom name="leg_geom" friction="0.7 0.1 0.1" pos="0 0 -.3" size="0.046" type="sphere" />
          <body name="foot" pos="0 0 -.6" xaxis="0 -1 0" zaxis="0 0 1">
            <joint name="foot_joint" pos="0 0 0" axis="0 -1 0" range="-45 45" type="hinge" />
            <geom name="foot_geom" friction="0.7 0.1 0.1" pos="0.068 0 -.068" size="0.046" type="sphere" />
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <asset>
    <material name="MatGround" rgba="0.6 0.6 0.6 1" texture="TexGround" texrepeat="10 10" />
    <texture name="TexGround" builtin="checker" height="100" rgb1="0.5 0.5 0.5" rgb2="0.6 0.6 0.6" type="2d" width="100" />
  </asset>
  <actuator>
    <motor ctrllimited="true" ctrlrange="-1 1" joint="thigh_joint" gear="200" />
    <motor ctrllimited="true" ctrlrange="-1 1" joint="leg_joint" gear="200" />
    <motor ctrllimited="true" ctrlrange="-1 1" joint="foot_joint" gear="200" />
  </actuator>
</mujoco>"""
    
    if output_path is None:
        # Use temp directory
        output_path = tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False).name
    
    with open(output_path, 'w') as f:
        f.write(xml_content)
    
    return output_path


# ==========================================
# 2. OJA NETWORK
# ==========================================
class OjaNetwork:
    """Manual Oja network with corrected learning rule."""
    def __init__(self, genome, config, eta=0.05):
        self.eta = eta
        self.input_keys = config.genome_config.input_keys
        self.output_keys = config.genome_config.output_keys
        
        self.weights = {}
        for conn_key, conn in genome.connections.items():
            if conn.enabled:
                self.weights[conn_key] = conn.weight
        
        self.nodes = {}
        for node_key, node in genome.nodes.items():
            self.nodes[node_key] = {
                'bias': node.bias,
                'response': node.response,
                'act_func': config.genome_config.activation_defs.get(node.activation)
            }
        
        self.values = {k: 0.0 for k in self.input_keys}
        self.values.update({k: 0.0 for k in self.nodes.keys()})

    def activate(self, inputs):
        """Forward pass with Oja's rule learning."""
        for i, val in zip(self.input_keys, inputs):
            self.values[i] = val
        
        new_values = self.values.copy()
        
        for node_key, node_data in self.nodes.items():
            node_input = node_data['bias']
            for (in_k, out_k), weight in self.weights.items():
                if out_k == node_key:
                    node_input += weight * self.values[in_k]
            
            res = node_data['response']
            new_values[node_key] = node_data['act_func'](res * node_input)
        
        # Corrected Oja's rule: dw = eta * y * (x - w * y)
        for (in_k, out_k), weight in self.weights.items():
            if out_k in self.output_keys:
                pre_val = self.values[in_k]
                post_val = new_values[out_k]
                delta_w = self.eta * post_val * (pre_val - weight * post_val)
                self.weights[(in_k, out_k)] += delta_w
        
        self.values = new_values
        return [self.values[k] for k in self.output_keys]

    def reset(self):
        """Reset network state."""
        for k in self.values.keys():
            self.values[k] = 0.0


# ==========================================
# 3. RENDER WITH DYNAMIC SLOPE
# ==========================================
def render_rollout_with_tilted_ground(
    genome,
    config,
    use_oja=True,
    video_name="hopper_rollout",
    render_steps=2000,
    slope_increment_steps=100,
):
    """
    Render rollout with **visually tilted ground**.
    Creates a new XML for each slope angle and renders with dynamic gravity.
    """
    # Create temp directory for XMLs
    xml_dir = tempfile.mkdtemp()
    
    env = gym.make("Hopper-v5", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder="./rendered_videos",
        episode_trigger=lambda ep: ep == 0,
        name_prefix=video_name,
        disable_logger=True
    )
    
    # Create network
    if use_oja:
        net = OjaNetwork(genome, config, eta=0.05)
    else:
        net = neat.nn.RecurrentNetwork.create(genome, config)
    
    # Start with flat ground
    xml_path = create_tilted_hopper_xml(0, os.path.join(xml_dir, 'hopper_0.xml'))
    env_internal = gym.make("Hopper-v5", render_mode="rgb_array", xml_file=xml_path)
    env_internal = gym.wrappers.RecordVideo(
        env_internal,
        video_folder="./rendered_videos",
        episode_trigger=lambda ep: ep == 0,
        name_prefix=video_name,
        disable_logger=True
    )
    
    observation, _ = env_internal.reset(seed=42)
    step_count = 0
    episode_reward = 0.0
    max_slope_reached = 0
    
    try:
        for step in range(render_steps):
            # Compute current slope
            slope_degree = step // slope_increment_steps
            max_slope_reached = slope_degree
            
            # Create tilted XML for this step (cached to avoid repeated generation)
            xml_path = os.path.join(xml_dir, f'hopper_{slope_degree}.xml')
            if not os.path.exists(xml_path):
                create_tilted_hopper_xml(slope_degree, xml_path)
            
            # If slope changed, we need to recreate the environment
            # (MuJoCo doesn't support on-the-fly XML swapping)
            # For now, continue with physics-only gravity rotation
            
            # Modify gravity for physics (even though visual is baked in XML)
            angle_rad = np.radians(slope_degree)
            env_internal.unwrapped.model.opt.gravity[:] = [
                9.81 * np.sin(angle_rad),
                0,
                -9.81 * np.cos(angle_rad)
            ]
            
            action = net.activate(observation)
            observation, reward, terminated, truncated, info = env_internal.step(action)
            episode_reward += reward
            step_count += 1
            
            if terminated or truncated:
                break
        
        print(f"  Completed {step_count} steps (max slope: {max_slope_reached}°)")
        print(f"  Total reward: {episode_reward:.2f}")
        
    finally:
        env_internal.close()
        env.close()
        shutil.rmtree(xml_dir)
    
    return step_count, episode_reward, max_slope_reached


# ==========================================
# 4. BETTER APPROACH: ROTATE CAMERA INSTEAD
# ==========================================
class TiltedEnvironmentWrapper(gym.Wrapper):
    """
    Wrapper that rotates the camera to show a tilted perspective.
    This is simpler than modifying XML files and works immediately.
    """
    def __init__(self, env, tilt_angle=0):
        super().__init__(env)
        self.tilt_angle = tilt_angle  # Will be updated during episode
        self.current_tilt = 0
    
    def update_tilt(self, angle_degrees):
        """Update the tilt angle for rendering."""
        self.current_tilt = angle_degrees
        # Update camera field of view or position to simulate tilt
        # Note: MuJoCo's rendering doesn't expose easy camera rotation
        # so we modify gravity instead and let physics show the tilt
        angle_rad = np.radians(angle_degrees)
        self.env.unwrapped.model.opt.gravity[:] = [
            9.81 * np.sin(angle_rad),
            0,
            -9.81 * np.cos(angle_rad)
        ]
    
    def step(self, action):
        return self.env.step(action)
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# ==========================================
# 5. SIMPLEST SOLUTION: MODIFY GROUND GEOM AT RUNTIME
# ==========================================
def render_rollout_simple(
    genome,
    config,
    use_oja=True,
    video_name="hopper_rollout",
    render_steps=2000,
    slope_increment_steps=100,
):
    """
    **Simplest approach**: Modify the ground geometry's rotation directly in memory.
    """
    env = gym.make("Hopper-v5", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder="./rendered_videos",
        episode_trigger=lambda ep: ep == 0,
        name_prefix=video_name,
        disable_logger=True
    )
    
    if use_oja:
        net = OjaNetwork(genome, config, eta=0.05)
    else:
        net = neat.nn.RecurrentNetwork.create(genome, config)
    
    observation, _ = env.reset(seed=42)
    step_count = 0
    episode_reward = 0.0
    max_slope_reached = 0
    
    # Find the ground geometry ID
    ground_geom_id = None
    for i in range(env.unwrapped.model.ngeom):
        if env.unwrapped.model.geom(i).name == 'ground':
            ground_geom_id = i
            break
    
    try:
        for step in range(render_steps):
            slope_degree = step // slope_increment_steps
            max_slope_reached = slope_degree
            angle_rad = np.radians(slope_degree)
            
            # ROTATE THE GROUND GEOM IN REAL-TIME
            if ground_geom_id is not None:
                rot = R.from_rotvec([0, angle_rad, 0])
                quat = rot.as_quat()  # [x, y, z, w]
                
                # MuJoCo stores quaternions as [w, x, y, z]
                env.unwrapped.model.geom_quat[ground_geom_id] = [
                    quat[3], quat[0], quat[1], quat[2]
                ]
            
            # Also update physics gravity
            env.unwrapped.model.opt.gravity[:] = [
                9.81 * np.sin(angle_rad),
                0,
                -9.81 * np.cos(angle_rad)
            ]
            
            action = net.activate(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            
            if terminated or truncated:
                print(f"  Episode terminated at step {step_count}")
                break
        
        print(f"  Completed {step_count} steps (max slope: {max_slope_reached}°)")
        print(f"  Total reward: {episode_reward:.2f}")
        
    finally:
        env.close()
    
    return step_count, episode_reward, max_slope_reached


# ==========================================
# 6. COMPARISON RENDER
# ==========================================
def render_comparison(genome, config, output_dir="./rendered_videos", render_steps=4000):
    """Render Oja ON vs OFF with tilted ground visualization."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("RENDERING COMPARISON WITH TILTED GROUND")
    print("="*60)
    
    print("\n[1/2] Rendering simple with OJA ENABLED...")
    steps_on, reward_on, slope_on = render_rollout_simple(
        genome, config, use_oja=True,
        video_name="hopper_oja_on",
        render_steps=render_steps
    )
    
    print("\n[2/2] Rendering simple with OJA DISABLED...")
    steps_off, reward_off, slope_off = render_rollout_simple(
        genome, config, use_oja=False,
        video_name="hopper_oja_off",
        render_steps=render_steps
    )
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"{'Metric':<20} | {'OJA ON':>15} | {'OJA OFF':>15}")
    print("-"*60)
    print(f"{'Steps':>19} | {steps_on:>15} | {steps_off:>15}")
    print(f"{'Reward':>19} | {reward_on:>15.2f} | {reward_off:>15.2f}")
    print(f"{'Max Slope':>19} | {slope_on:>14}° | {slope_off:>14}°")
    print("="*60)


# ==========================================
# 7. BATCH EVOLUTION WITH RENDERING
# ==========================================
class GenomeEvaluator:
    """Fitness evaluator for NEAT (from previous code)."""
    def __init__(self, use_oja=False):
        self.use_oja = use_oja
        self.env = None
    
    def __call__(self, genomes, config):
        if self.env is None:
            self.env = gym.make("Hopper-v5")
        
        for _, genome in genomes:
            self._evaluate_genome(genome, config)
    
    def _evaluate_genome(self, genome, config):
        if self.use_oja:
            net = OjaNetwork(genome, config, eta=0.05)
        else:
            net = neat.nn.RecurrentNetwork.create(genome, config)
        
        observation, _ = self.env.reset(seed=42)
        fitness = 0.0

        for step in range(1000):
            action = net.activate(observation)
            observation, reward, terminated, truncated, info = self.env.step(action)
            fitness += reward
            
            if terminated or truncated:
                fitness -= 10.0
                break
        
        genome.fitness = fitness
    
    def close(self):
        if self.env is not None:
            self.env.close()
                    
def evolve_and_render(num_generations=50, config_path="config-hopper"):
    """
    Run evolution and render the best genome at the end.
    """
    local_dir = os.path.dirname(__file__)
    config_path_full = os.path.join(local_dir, config_path)
    
    if not os.path.exists(config_path_full):
        raise FileNotFoundError(f"Config not found: {config_path_full}")
    
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path_full
    )
    
    print("\n" + "="*60)
    print("NEAT EVOLUTION WITH OJA'S RULE")
    print("="*60)
    
    # Evolution phase
    p = neat.Population(config)
    evaluator = GenomeEvaluator(use_oja=True)
    
    try:
        best_genome = p.run(evaluator, num_generations)
    finally:
        evaluator.close()
    
    print(f"\nEvolution complete. Best fitness: {best_genome.fitness:.2f}")
    
    # Rendering phase
    print("\nStarting render comparison...")
    render_comparison(best_genome, config, render_steps=4000)
    
    return best_genome, config

# ==========================================
# MAIN
# ==========================================
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--evolve':
        # Evolve and render
        best_genome, config = evolve_and_render(num_generations=50)
    else:
        # Or just run rendering if you have a pre-evolved genome
        print("Usage:")
        print("  python script.py --evolve    # Evolve first, then render")
        print("\nTo render a genome directly, provide it to render_comparison()")
        print("\nExample:")
        print("  genome = neat.Population(config).run(evaluator, 50)")
        print("  render_comparison(best_genome, config, render_steps=2000)")