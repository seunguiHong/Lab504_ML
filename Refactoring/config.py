# config.py

# ============================================================
# Data
# ============================================================

data_path = "data/target_and_features.mat"
feature_groups = ["d12m_y"]
target_group = "dy"
target_indices = None

# ============================================================
# Model
# ============================================================

model = "pcNN"          # "rawNN" or "pcNN"

pc_ncomp = 3
pc_keep = [1]           # zero-based; [1] means PC2
standardize = True

archi = [3]             # [] means closed-form OLS; [3] means one-hidden-layer NN

dropout_grid = [0.0]
l1l2_grid = [[0.0, 0.0]]

learning_rate = 0.02
decay_rate = 0.001
momentum = 0.9
nesterov = True

epochs = 500
patience = 20
batch_size = 32
shuffle = False
loss = "mse"
huber_delta = 1.0

# ============================================================
# Forecasting design
# ============================================================

oos_start = "1989-01-31"
purge_gap = 12
validation_frac = 0.15
hyper_freq = 60

benchmark = "zero"      # "zero" (Random Walk) or "historical_mean" (Expectation Hypothesis)

nmc = 100
navg = 10

# ============================================================
# Output
# ============================================================

run_tag = "pcNN_d12m_y_pc2"
out_file = "results/TL_ycpc2_[3].mat"