# Evolving Adaptability: Synaptic Plasticity as a Mechanism for Robust Locomotion in Dynamic Environments

## Abstract
This paper details the iterative development and validation of a neuroevolutionary framework that integrates **Oja’s Rule**—a localized, unsupervised synaptic plasticity mechanism—into the **NeuroEvolution of Augmenting Topologies (NEAT)** algorithm. The objective was to determine if real-time weight adaptation could provide a survival advantage for a robotic agent (the "Hopper-v5") subjected to a dynamically destabilizing environment. Through a series of experimental phases, including a pilot study, a large-scale batch analysis, and a rigorous paired-ablation study, the research demonstrated that plastic controllers significantly outperformed static ones. Final results indicated an **823% increase** in survival duration for plastic agents ($p < 0.00001$), suggesting that local learning rules can effectively compensate for environmental shifts that surpass the generalization limits of static evolutionary solutions.

---

## 1. Introduction
The field of evolutionary robotics often relies on fixed-weight neural networks to solve motor control tasks. While algorithms like NEAT are proficient at discovering efficient topologies for static environments, these "stiff" controllers frequently fail when faced with out-of-distribution environmental shifts. In biological systems, locomotion is not merely a product of hard-coded reflexes but a result of continuous sensory-motor adaptation.

This research sought to bridge this gap by implementing **Hebbian plasticity** within an evolved framework. The core hypothesis posited that if a neural network could adjust its weights in real-time based on local neuronal activity, it would "learn" to compensate for physical stressors—such as a shifting center of gravity—more effectively than a static network.

### 1.1 The Mathematical Foundation: Oja’s Rule
Standard Hebbian learning ($\Delta w = \eta yx$) is notoriously unstable, often leading to infinite weight growth. To maintain biological and computational plausibility, the researchers utilized **Oja’s Rule**, which introduces a normalized term to constrain weight growth while preserving the associative properties of the learning. The update is defined as:

$$\Delta w_{ij} = \eta (y_i x_j - w_{ij} y_i^2)$$

Where:
* $\eta$ is the learning rate.
* $y_i$ is the activity of the post-synaptic neuron.
* $x_j$ is the activity of the pre-synaptic neuron.
* $w_{ij}$ is the current weight of the connection.

---

## 2. Experimental Methodology

### 2.1 The Agent and Environment
The experiment utilized the **Gymnasium Hopper-v5** environment, powered by the MuJoCo physics engine. The Hopper is a single-legged robot with four joints, making it highly sensitive to balance and gravity. 

### 2.2 The Stress Test: The Tilting Slope
To measure "adaptability," the researchers developed a **Shifting Gravity Stress Test**. Unlike standard evaluation on flat ground, this test applied a gradual tilt to the world’s gravity vector.
* **The Increment:** The gravity was tilted by $1^\circ$ every 100 time-steps.
* **The Goal:** Survival. The agent was required to maintain forward momentum without falling while the environment became progressively more hostile.
* **The Ceiling:** The test was eventually extended to **4000 steps** to ensure that even the most robust controllers would eventually encounter a failure state (at approximately a $40^\circ$ incline).

---

## 3. The Iterative Journey

### 3.1 Phase 1: Integration and Pilot Testing
The initial phase involved wrapping the standard `neat-python` recurrent network with a plasticity layer. A pilot test comparing a single Oja-enabled genome against a static control group suggested a massive performance leap. However, the researchers recognized that a single seed comparison was insufficient to claim statistical validity.

### 3.2 Phase 2: The Inconclusive Batch
An overnight run was conducted with 20 independent evolutionary seeds. Surprisingly, the **Mann-Whitney U test** returned a $p$-value of **0.8589**, indicating no significant difference between the two populations. 

**The Autopsy:**
Analysis revealed a **"Ceiling Effect."** The initial 1000-step limit was too easy; both groups were reaching the maximum score, squashing the variance and hiding the true differences. Furthermore, the high variance in the evolutionary process meant that "lucky" static evolutions were outperforming "poor" plastic evolutions, muddling the data.

### 3.3 Phase 3: The Ablation Pivot
To eliminate evolutionary noise, the researchers shifted to an **Ablation Study**. Instead of comparing two different populations, they evolved a single "Plastic-Ready" genome and tested it twice:
1.  **Condition A (ON):** Oja's updates active.
2.  **Condition B (OFF):** Updates disabled (Static).

### 3.4 Phase 4: Debugging the "Silent Bug"
Initial ablation results showed identical scores for "ON" and "OFF" conditions ($1604$ steps vs $1604$ steps). This led to a critical realization: the library’s internal activation logic was not persisting the weight changes. The `node_evals` list in `neat-python` used immutable tuples, causing the Oja updates to be discarded every time the `activate` function finished.

**The Solution:**
The researchers bypassed the library's pre-compiled activation sequence and authored a **Manual Oja Network** class. This pure-Python implementation maintained a mutable weight dictionary, ensuring that every $\Delta w$ was physically written to the network's state at every time-step.

---

## 4. Results and Statistical Analysis

### 4.1 Quantitative Outcomes
With the manual override active, the results diverged sharply. In a final batch of 20 independent runs, the following data was captured:

| Metric | Oja ON (Plastic) | Oja OFF (Static) |
| :--- | :--- | :--- |
| **Mean Survival Steps** | **896.10** | **108.85** |
| **Average Max Slope** | **~9°** | **~1°** |
| **Survival Improvement** | **823%** | **N/A** |

### 4.2 Statistical Significance
A **Wilcoxon Signed-Rank Test** was applied to the paired data. The test yielded a $p$-value of **0.00001**. This result definitively rejected the null hypothesis, proving that plasticity—not just the underlying evolved topology—was responsible for the increased robustness.

---

## 5. Discussion

### 5.1 Stability vs. Agility
A key finding involved the optimization of the learning rate ($\eta$). Early tests at $0.01$ showed moderate gains, but increasing the rate to **$0.05$** provided the agility required to keep pace with the $1^\circ/100$-step environmental shift. This suggests that the speed of the plasticity must be tuned to the rate of environmental change. 

### 5.2 Mechanisms of Failure
When Oja was disabled, agents failed almost immediately at the $1^\circ$ mark ($M=108.85$). This indicates that the evolved topologies were highly "overfit" to the flat-ground physics used during training. The plastic agents, however, were able to redistribute their synaptic strengths to compensate for the shifting "kick-back" forces of the tilting terrain.

---

## 6. Conclusion
The journey from a pilot observation to a $p=0.00001$ statistical certainty highlights the importance of experimental control and transparency in neural network internals. By integrating Oja’s Rule and isolating the variable through a paired ablation study, this research demonstrated that localized plasticity can transform a fragile evolved controller into a robust, adaptive system.

Future work should explore **Neuromodulation**, where evolution controls not only the weights but the learning rates themselves, allowing the agent to "learn when to learn."

---

## 7. Appendices
### Final Code Architecture (Simplified)
The system utilized a custom `OjaNetwork` class that manually computed the recurrent forward pass to ensure the weight matrix remained mutable throughout the agent's lifespan. The integration of `scipy.stats.wilcoxon` ensured the rigor of the final claims.

### 1. **Corrected Oja's Rule**[1][2]
The mathematical formula now correctly implements:
$$\Delta w = \eta \cdot y(x - w \cdot y)$$

This maintains weight normalization ($$\sum w_i^2 = 1$$) and performs proper PCA while preventing unbounded weight growth.[1]

### 2. **Proper NEAT Interface**
- Created `GenomeEvaluator` class with `__call__` method
- Properly implements the callback interface required by `population.run()`
- Manages environment lifecycle within the evaluator

### 3. **Robust Recurrent Network**
- Uses double-buffering pattern correctly (reads from `self.values`, writes to `new_values`)
- Proper separation of pre- and post-synaptic values for Oja's rule

### 4. **Adaptive Gradient Handling**
- Properly modifies gravity vector each step to simulate tilting surface
- Hopper learns real-time adaptation to changing physics
- Slope increases gradually allowing smooth learning progression

### 5. **Production-Grade Code Quality**
- Comprehensive error handling and validation
- Informative logging and progress tracking
- Resource cleanup via `try/finally` blocks
- Docstrings explaining key concepts
- Proper CSV batching (not appending in loop)

### 6. **Better Statistics**
- Reports median in addition to mean (more robust)
- Shows effect sizes explicitly
- Includes warning about test assumptions

The code now correctly tests whether **Oja's rule enables better real-time adaptation** to dynamic environmental changes (changing surface gradient). The weight normalization should help maintain stable learning even as the hopper encounters steeper slopes.

Sources
[1] 19.2 Models of Hebbian learning | Neuronal Dynamics online bookneuronaldynamics.epfl.ch › online › Ch19.S2.html https://neuronaldynamics.epfl.ch/online/Ch19.S2.html
[2] Oja's rule - Wikipedia https://en.wikipedia.org/wiki/Oja's_rule
[3] ◆ __init__() https://fossies.org/dox/Gymnasium-1.2.2/classgymnasium_1_1envs_1_1mujoco_1_1hopper__v5_1_1HopperEnv.html
[4] MuJoCo Physics Engine - Emergent Mind https://www.emergentmind.com/topics/mujoco-physics-engine
[5] Source code for nn.recurrent - NEAT-Python's documentation! https://neat-python.readthedocs.io/en/latest/_modules/nn/recurrent.html
[6] Oja's plasticity rule overcomes several challenges of ... https://arxiv.org/html/2408.08408v1
[7] Minari Documentation https://minari.farama.org/datasets/mujoco/hopper/medium-v0/
[8] MuJoCo — Advanced Physics Simulation https://mujoco.org
[9] Source code for ctrnn https://neat-python.readthedocs.io/en/latest/_modules/ctrnn.html
[10] Hopper¶ https://gymnasium.org.cn/environments/mujoco/hopper/
[11] Overview# https://mujoco.readthedocs.io/en/2.3.7/overview.html
[12] Continuous-time recurrent neural network implementation https://neat-python.readthedocs.io/en/latest/ctrnn.html
[13] Hopper - Gymnasium Documentation https://gymnasium.farama.org/v0.27.1/environments/mujoco/hopper/
[14] Changing the geom mass fails to affect simulation properties? #1717 https://github.com/google-deepmind/mujoco/issues/1717
[15] Need help with a NEAT implementation in Python : r/MLQuestions https://www.reddit.com/r/MLQuestions/comments/13oo4ck/need_help_with_a_neat_implementation_in_python/


### Summary of Convergence
The breakthrough occurred when the researchers abandoned the "black-box" activation of the library in favor of a manual implementation, proving that in AI research, the fidelity of the mechanism is as important as the logic of the algorithm.