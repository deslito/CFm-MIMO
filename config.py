# config.py - Centralized configuration

# System Parameters
M = 100  # Number of APs
K = 20   # Number of UEs
L = 4    # Antennas per AP

# Propagation Parameters
path_loss_exponent = 3.8
shadowing_std = 8  # dB
power_allocation_v = 0.5

# Part 1: K-Means
KMEANS_CLUSTERS = 4  # Adjust based on performance
KMEANS_MAX_ITER = 300

# Part 2: TD3
TD3_STATE_DIM = None  # Define after state design
TD3_ACTION_DIM = None  # Define after action design
TD3_LEARNING_RATE = 3e-4
TD3_BATCH_SIZE = 256
TD3_BUFFER_SIZE = 100000
TD3_EPISODES = 500

# Simulation Parameters
NUM_REALIZATIONS = 100  # For CDF curves
RANDOM_SEED = 42