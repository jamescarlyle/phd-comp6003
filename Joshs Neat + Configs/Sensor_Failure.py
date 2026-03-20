# This is based on this paper: https://ntrs.nasa.gov/api/citations/20090033812/downloads/20090033812.pdf
# In short we need to develop a fn which mirrors the types of sensor failure.

#------------ Brief Summary ------------
# Terminology used from here: Y = output of a sensor

# Bias
# Y = X + B + noise
# B = Constant bias value

# Drift
# Y = X + d(t) + noise
# d(t) = Time variant drift fn
# Inserting a sin or smth periodic could be interesting

# Scaling
# Y = a(t) * X + noise
# a(t) = scalaing const (can vary with time)

# Noise
# Y = noise
# just noisy im fairly sure it means Y_obs = Y_actual + noise
# mean = 0

# Hard Fault
# Y = C + noise
# When C = 0 this is just complete sensor failure.

#------------ Brief Summary ------------
# In the above paper there is a given range and median, this will help.
# There are also a reference to the associated drifts alongside the types of sensors which give rise to them.

# Bias:
# Range: 1.2% - 60%
# Median: 20%
# Context: % changer over the nominal value

# Scaling:
# Range: 0.3-0.7 or 2.5-4.8
# Median: 0.45 or 3.28
# Context: Scale Factor

# Drift:
# Range: 6% - 75%
# Median: 29%
# Context: % change over the nominal value reported at the end of the dataset/drift (Asymptoic???)

# Noise:
# Range: 2.5% - 250%
# Median: 20%
# Context: %peak to peak values over the nominal value

# Intermittent Dropout:
# Range: 2 - 10 drops
# Median: 8 drops
# Context: over a range of 20% to 29% of the reported data set with the median range of 23%
# Note: Not entirely sure that this means

#------------ Brief Summary ------------

import matplotlib.pyplot as plt
import numpy as np
import random
import math

# --- 1 ---
def sensor_bias(obs, bias, mean, stan_dev):
    return obs + bias + np.random.normal(mean, stan_dev)


# --- 2 ---
def sensor_drift(obs, drift_fn, timestep, mean, stan_dev):
    return obs + drift_fn(timestep) + np.random.normal(mean, stan_dev)


# --- 3 ---
def sensor_drift_const(obs, scale_factor, mean, stan_dev):
    return obs * scale_factor + np.random.normal(mean, stan_dev)


# --- 4 ---
def sensor_drift_dynamic(obs, scale_fn, timestep, mean, stan_dev):
    return obs * scale_fn(timestep) + np.random.normal(mean, stan_dev)

def hard_fault_constant(obs, mean, stan_dev):
    return obs + np.random.normal(mean, stan_dev)


# --- 5 ---
def hard_fault_random(obs, fail_const, rate, mean, stan_dev):
    sample = random.random()
    if sample <= rate:
        return fail_const + np.random.normal(mean, stan_dev)
    else:
        return obs + np.random.normal(mean, stan_dev)


count = 500

X_vals = np.linspace(0, 1, count)

Y_noiseless = np.array([10 for i in range(0, count)])

Y_noise_1 = np.array([sensor_bias(10,
                                  1,
                                  0,
                                  0) for i in range(0, count)])

Y_noise_2 = np.array([sensor_drift(10,
                                   math.sin,
                                   X_vals[i],
                                   0,
                                   0) for i in range(0, count)])
Y_noise_3 = np.array([hard_fault_random(10, 0, 0.1, 0, 0) for i in range(0, count)])



plt.scatter(X_vals, Y_noiseless)
plt.scatter(X_vals, Y_noise_3)
plt.ylim((0,15))
plt.show()

















