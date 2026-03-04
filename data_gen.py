import torch

def get_fresh_batch_dataset(num_samples=450000, bias=0.5, device='cuda'):
    # 1. Physical Constants (Must match env.py exactly)
    I_MAX = 3000.0
    FN_MAX = 20.0
    I_CRIT = 450.0
    ALPHA = 0.25
    L = 0.5
    N_LIMIT = 180.0

    # 2. Define 20/80 Split
    n_high = int(num_samples * bias)
    n_low = num_samples - n_high
    
    # --- BIAS FOR BIOMASS (cx) --- 
    cx_uniform = torch.rand(n_low, device=device) * 6.0
    cx_high = 4.0 + torch.rand(n_high, device=device) * 2.0
    cx = torch.cat([cx_uniform, cx_high])

    # --- BIAS FOR NITRATE (cN) ---
    cN_uniform = torch.rand(n_low, device=device) * 200.0
    cN_boundary = 160.0 + torch.rand(n_high, device=device) * 40.0
    cN = torch.cat([cN_uniform, cN_boundary])
    
    cq = torch.rand(num_samples, device=device) * 25.0
    
    # 3. Sample Actions in the Agent's Space [-1, 1]
    a_nom_uniform = torch.rand(n_low, 2, device=device) * 2.0 - 1.0
    a_nom_high = 0.5 + torch.rand(n_high, 2, device=device) * 0.5
    a_nom = torch.cat([a_nom_uniform, a_nom_high])
    
    # 4. Denormalize Actions to Physical Units
    a_scaled = (a_nom + 1.0) / 2.0
    i_phys = a_scaled[:, 0] * I_MAX
    fn_phys = a_scaled[:, 1] * FN_MAX
    
    # 5. Apply Physical Constraints with Buffer
    # --- Light Constraint ---
    shading = torch.exp(-ALPHA * cx * L)
    i_limit = I_CRIT / (shading + 1e-6)
    i_safe_phys = torch.minimum(i_phys, i_limit * 0.95)
    
    # --- Updated Nitrogen Constraint: Preventative Budgeting ---
    # We calculate the remaining capacity before hitting 95% of the limit.
    # Logic: Safe_FN = min(Requested_FN, (N_Limit * 0.95) - Current_N)
    n_buffer_target = N_LIMIT * 0.95
    fn_budget = torch.clamp(n_buffer_target - cN, min=0.0)
    fn_safe_phys = torch.minimum(fn_phys, fn_budget)

    # 6. Re-normalize Safe Physical Actions back to [-1, 1]
    i_safe_norm = (i_safe_phys / I_MAX) * 2.0 - 1.0
    fn_safe_norm = (fn_safe_phys / FN_MAX) * 2.0 - 1.0
    
    # 7. Stack Outputs
    states = torch.stack([cx, cN, cq], dim=1)
    nom_actions = a_nom
    target_actions = torch.stack([i_safe_norm, fn_safe_norm], dim=1)
    
    return states, nom_actions, target_actions