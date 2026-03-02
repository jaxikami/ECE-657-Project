import numpy as np
import torch
from torch.utils.data import TensorDataset

def get_fresh_batch_dataset(num_samples=20000):
    """
    Generates a fresh dataset for a single epoch to prevent overfitting.
    """
    # Physical Constants
    I_crit, alpha, L, Fn_max = 450.0, 0.25, 0.5, 10.0
    
    # 1. Sample State Space
    cx = np.random.uniform(0.05, 6.0, num_samples)
    cN = np.random.uniform(0.0, 200.0, num_samples)
    cq = np.random.uniform(0.0, 25.0, num_samples)
    
    # 2. Sample Nominal Actions
    i_nom = np.random.uniform(0, 3000, num_samples)
    fn_nom = np.random.uniform(0, 20, num_samples)
    
    # 3. Apply Hard Constraints
    shading_factor = np.exp(-alpha * cx * L)
    i_limit = I_crit / (shading_factor + 1e-6)
    i_safe = np.minimum(i_nom, i_limit)
    
    fn_limit = np.where(cN > 180.0, 0.0, Fn_max)
    fn_safe = np.minimum(fn_nom, fn_limit)
    
    # 4. Convert to Tensors
    states = torch.tensor(np.stack([cx, cN, cq], axis=1), dtype=torch.float32)
    nom_actions = torch.tensor(np.stack([i_nom, fn_nom], axis=1), dtype=torch.float32)
    target_actions = torch.tensor(np.stack([i_safe, fn_safe], axis=1), dtype=torch.float32)
    
    return states, nom_actions, target_actions