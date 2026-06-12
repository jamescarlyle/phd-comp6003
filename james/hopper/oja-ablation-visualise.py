import os
import pickle
import numpy as np
import gymnasium as gym
import neat
import cv2

# ==========================================
# 1. OJA NETWORK (same as training)
# ==========================================
class OjaNetwork:
    """Manual Oja network implementation."""
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

        # Initialize neural activity
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
# 2. OVERLAY TEXT ON FRAME
# ==========================================
def add_text_overlay(frame, text_lines, position="top-left", font_scale=0.6, thickness=1):
    """
    Add text overlay to frame.

    Args:
        frame: RGB numpy array (HxWx3, uint8)
        text_lines: List of strings to display
        position: "top-left", "top-right", "bottom-left", "bottom-right"
        font_scale: Font size multiplier
        thickness: Text line thickness

    Returns:
        Modified frame
    """
    frame = frame.copy()  # Don't modify original
    height, width = frame.shape[:2]

    # Convert RGB to BGR for OpenCV
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_color = (255, 255, 255)  # White in BGR
    bg_color = (0, 0, 0)  # Black background

    # Text positioning
    margin = 10
    line_height = 25

    if "top" in position:
        y_start = margin + 20
        y_increment = line_height
    else:
        y_start = height - margin - (len(text_lines) * line_height)
        y_increment = line_height

    if "right" in position:
        x_pos = "right"
    else:
        x_pos = margin

    # Draw each line
    for i, text in enumerate(text_lines):
        y = y_start + (i * y_increment)

        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )

        if x_pos == "right":
            x = width - text_width - margin
        else:
            x = x_pos

        # Draw background rectangle
        cv2.rectangle(
            frame_bgr,
            (x - 5, y - text_height - 5),
            (x + text_width + 5, y + baseline + 5),
            bg_color,
            -1
        )

        # Draw text
        cv2.putText(
            frame_bgr,
            text,
            (x, y),
            font,
            font_scale,
            font_color,
            thickness,
            cv2.LINE_AA
        )

    # Convert back to RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return frame_rgb

# ==========================================
# 3. DRAW GRAVITY VECTOR
# ==========================================
def draw_gravity_arrow(frame, angle_rad, arrow_length=80):
    """
    Draw gravity vector arrow in top-right corner.

    Args:
        frame: RGB numpy array
        angle_rad: Angle in radians (0 = downward, positive = tilted right)
        arrow_length: Length of arrow in pixels

    Returns:
        Modified frame with arrow
    """
    frame = frame.copy()
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    height, width = frame.shape[:2]

    # Arrow origin (top-right corner)
    origin_x = width - 50
    origin_y = 50

    # Gravity points downward, so add pi/2 to angle
    # angle_rad=0 means vertical down, positive angles tilt to the right
    gravity_angle = np.pi / 2 + angle_rad

    # Calculate arrow endpoint
    end_x = int(origin_x + arrow_length * np.cos(gravity_angle))
    end_y = int(origin_y + arrow_length * np.sin(gravity_angle))

    # Draw arrow line
    cv2.arrowedLine(
        frame_bgr,
        (origin_x, origin_y),
        (end_x, end_y),
        (0, 255, 0),  # Green
        3,
        tipLength=0.3
    )

    # Draw circle at origin
    cv2.circle(frame_bgr, (origin_x, origin_y), 5, (255, 0, 0), -1)  # Blue

    # Convert back to RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return frame_rgb

# ==========================================
# 4. EPISODE RENDERER WITH GRADIENT
# ==========================================
def render_episode_with_gradient(genome, config, use_oja, num_steps=1000, seed=42, 
                                  apply_gradient=False, max_gradient=45):
    """
    Render an episode with optional increasing gradient.

    Args:
        genome: NEAT genome to evaluate
        config: NEAT config object
        use_oja: Boolean for Oja learning rule
        num_steps: Maximum steps per episode
        seed: Random seed
        apply_gradient: If True, gradually increase tilt over episode
        max_gradient: Maximum tilt angle in degrees

    Returns:
        Tuple of (frames list, total_reward)
    """
    env = gym.make("Hopper-v5", render_mode="rgb_array")

    # Create network
    if use_oja:
        net = OjaNetwork(genome, config, eta=0.05)
    else:
        net = neat.nn.RecurrentNetwork.create(genome, config)

    observation, _ = env.reset(seed=seed)
    frames = []
    total_reward = 0.0
    current_slope = 0

    try:
        for step in range(num_steps):
            # Update gradient if enabled
            if apply_gradient:
                # Gradually increase slope: increment every 50 steps
                current_slope = min(max_gradient, (step // 50) * 5)
                angle_rad = np.radians(current_slope)

                # Modify gravity vector
                env.unwrapped.model.opt.gravity[:] = [
                    9.81 * np.sin(angle_rad),
                    0,
                    -9.81 * np.cos(angle_rad)
                ]

            # Get network action
            action = net.activate(observation)

            # Render current frame
            frame = env.render()
            if frame is not None:
                # Add overlays
                frame_with_overlay = frame.copy()

                # Add gravity vector arrow
                if apply_gradient:
                    angle_rad = np.radians(current_slope)
                else:
                    angle_rad = 0.0
                frame_with_overlay = draw_gravity_arrow(frame_with_overlay, angle_rad)

                # Add text info
                text_lines = [
                    f"Step: {step + 1}",
                    f"Slope: {current_slope}Â°",
                    f"Reward: {total_reward:.1f}",
                ]
                if use_oja:
                    text_lines.append("Mode: OJA ON")
                else:
                    text_lines.append("Mode: OJA OFF")

                frame_with_overlay = add_text_overlay(frame_with_overlay, text_lines, 
                                                      position="top-left")

                frames.append(frame_with_overlay)

            # Step environment
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                print(f"  Episode ended at step {step + 1}, total reward: {total_reward:.2f}")
                break
    finally:
        env.close()

    return frames, total_reward

# ==========================================
# 5. FRAMES TO MP4 CONVERSION
# ==========================================
def frames_to_mp4(frames, output_filename, fps=30):
    """
    Convert list of frames to MP4 using OpenCV.

    Args:
        frames: List of numpy arrays (HxWx3, uint8)
        output_filename: Path to output MP4 file
        fps: Frames per second for output video
    """
    if not frames:
        print(f"ERROR: No frames to save for {output_filename}")
        return

    # Get frame dimensions
    height, width = frames[0].shape[:2]

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"ERROR: Failed to open video writer for {output_filename}")
        return

    print(f"  Writing {len(frames)} frames to {output_filename}...")
    for i, frame in enumerate(frames):
        # OpenCV expects BGR, but gymnasium gives RGB
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{len(frames)} frames written")

    out.release()
    print(f"  âœ“ Saved to {output_filename}")

# ==========================================
# 6. MAIN RENDERING PIPELINE
# ==========================================
def main():
    """Load best genome and render Oja vs non-Oja episodes to MP4."""

    # Check for pickle file
    pickle_filename = 'best_genome.pickle'
    if not os.path.exists(pickle_filename):
        raise FileNotFoundError(
            f"Pickle file not found: {pickle_filename}\n"
            "Please run the training script first to generate best_genome.pickle"
        )

    print("="*60)
    print("RENDERING BEST GENOME TO MP4 (with Gravity Overlay)")
    print("="*60)

    # Load best genome and config
    print(f"\n[*] Loading genome from {pickle_filename}...")
    with open(pickle_filename, 'rb') as f:
        data = pickle.load(f)

    genome = data['genome']
    config = data['config']
    fitness = data['fitness']

    print(f"[+] Loaded genome with fitness: {fitness:.2f}")
    print(f"    Connections: {len(genome.connections)}")
    print(f"    Nodes: {len(genome.nodes)}")

    # Render Oja episode (no gradient for baseline)
    print(f"\n[*] Rendering Oja episode (flat ground)...")
    frames_oja, reward_oja = render_episode_with_gradient(
        genome, config, use_oja=True, num_steps=1000, seed=42,
        apply_gradient=False
    )

    # Render non-Oja episode (no gradient for baseline)
    print(f"\n[*] Rendering non-Oja episode (flat ground)...")
    frames_non_oja, reward_non_oja = render_episode_with_gradient(
        genome, config, use_oja=False, num_steps=1000, seed=42,
        apply_gradient=False
    )

    # Optional: Also render with gradient to show adaptation
    print(f"\n[*] Rendering Oja episode (with increasing gradient)...")
    frames_oja_gradient, reward_oja_gradient = render_episode_with_gradient(
        genome, config, use_oja=True, num_steps=1000, seed=42,
        apply_gradient=True, max_gradient=45
    )

    print(f"\n[*] Rendering non-Oja episode (with increasing gradient)...")
    frames_non_oja_gradient, reward_non_oja_gradient = render_episode_with_gradient(
        genome, config, use_oja=False, num_steps=1000, seed=42,
        apply_gradient=True, max_gradient=45
    )

    # Save MP4 files
    print(f"\n[*] Converting frames to MP4...")

    oja_filename = 'hopper_oja_flat.mp4'
    frames_to_mp4(frames_oja, oja_filename, fps=30)

    non_oja_filename = 'hopper_non_oja_flat.mp4'
    frames_to_mp4(frames_non_oja, non_oja_filename, fps=30)

    oja_gradient_filename = 'hopper_oja_gradient.mp4'
    frames_to_mp4(frames_oja_gradient, oja_gradient_filename, fps=30)

    non_oja_gradient_filename = 'hopper_non_oja_gradient.mp4'
    frames_to_mp4(frames_non_oja_gradient, non_oja_gradient_filename, fps=30)

    # Summary
    print("\n" + "="*60)
    print("RENDERING COMPLETE")
    print("="*60)
    print("\nFlat Ground Episodes:")
    print(f"  Oja:     {oja_filename}")
    print(f"           {len(frames_oja)} frames, reward: {reward_oja:.2f}")
    print(f"  Non-Oja: {non_oja_filename}")
    print(f"           {len(frames_non_oja)} frames, reward: {reward_non_oja:.2f}")

    print("\nGradient Episodes (gravity angle shown with arrow):")
    print(f"  Oja:     {oja_gradient_filename}")
    print(f"           {len(frames_oja_gradient)} frames, reward: {reward_oja_gradient:.2f}")
    print(f"  Non-Oja: {non_oja_gradient_filename}")
    print(f"           {len(frames_non_oja_gradient)} frames, reward: {reward_non_oja_gradient:.2f}")

    print(f"\nGradient Performance Improvement:")
    print(f"  Reward diff (Oja - Non-Oja): {reward_oja_gradient - reward_non_oja_gradient:+.2f}")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()