import torch

def get_fresh_batch_dataset(num_samples=450000, device='cuda'):
    # 1. Physical Constants (Must match env.py exactly)
    I_MAX = 3000.0
    FN_MAX = 20.0
    I_CRIT = 450.0
    ALPHA = 0.25
    L = 0.5
    N_LIMIT = 180.0

    # 2. Sample State Space (Raw Physical Units)
    cx = torch.rand(num_samples, device=device) * 6.0
    cN = torch.rand(num_samples, device=device) * 200.0
    cq = torch.rand(num_samples, device=device) * 25.0
    
    # 3. Sample Actions in the Agent's Space [-1, 1]
    # This ensures the safeguard learns the mapping the Agent actually uses.
    a_nom = torch.rand(num_samples, 2, device=device) * 2.0 - 1.0
    
    # 4. Denormalize Actions to Physical Units for Constraint Checking
    a_scaled = (a_nom + 1.0) / 2.0
    i_phys = a_scaled[:, 0] * I_MAX
    fn_phys = a_scaled[:, 1] * FN_MAX
    
    # 5. Apply Physical Constraints
    # Light Constraint
    shading = torch.exp(-ALPHA * cx * L)
    i_limit = I_CRIT / (shading + 1e-6)
    i_safe_phys = torch.minimum(i_phys, i_limit * 0.98) # 2% buffer for safety
    
    # Nitrate Constraint (If cN is high, we must cap the feed Fn)
    # Since the agent's goal is to keep cN < 180, we restrict Fn if we are close.
    fn_safe_phys = torch.where(cN > N_LIMIT * 0.95, 
                               torch.zeros_like(fn_phys), 
                               fn_phys)

    # 6. Re-normalize Safe Physical Actions back to [-1, 1]
    i_safe_norm = (i_safe_phys / I_MAX) * 2.0 - 1.0
    fn_safe_norm = (fn_safe_phys / FN_MAX) * 2.0 - 1.0
    
    # 7. Stack Outputs
    states = torch.stack([cx, cN, cq], dim=1)
    nom_actions = a_nom # Already in [-1, 1]
    target_actions = torch.stack([i_safe_norm, fn_safe_norm], dim=1)
    
    return states, nom_actions, target_actions