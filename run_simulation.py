import numpy as np
from metrics import compute_metrics

# fake state (10x10x10 cube with some grains)
state = np.zeros((10, 10, 10), dtype=int)
state[2:6, 2:6, 2:6] = 1   # one "grain"
state[6:9, 6:9, 6:9] = 2   # another "grain"

theta = np.zeros_like(state, dtype=float)
phi = np.zeros_like(state, dtype=float)

# fake impurity masks
W_mask = np.zeros_like(state, dtype=bool)
Re_mask = np.zeros_like(state, dtype=bool)
C_mask = np.zeros_like(state, dtype=bool)

W_mask[3,3,3] = True   # one W impurity
Re_mask[7,7,7] = True  # one Re impurity
C_mask[5,5,5] = True   # one C impurity

# grain IDs (just copy state for test)
grain_ids = state.copy()

# run metrics
m = compute_metrics(state, theta, phi, defects=None,
                    W_mask=W_mask, Re_mask=Re_mask, C_mask=C_mask,
                    grain_ids=grain_ids, rng_seed=42)

print("Keys:", m.keys())
print("Values:", m)
