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
    # 1. Uniformly sample across the new 0-170 range
    cN_uniform = torch.rand(n_low, device=device) * 170.0
    
    # 2. Focus high-bias samples on the top edge (150-170)
    # This ensures the model sees many cases where a small action triggers a violation
    cN_boundary = 150.0 + torch.rand(n_high, device=device) * 20.0
    
    cN = torch.cat([cN_uniform, cN_boundary])
    
    cq = torch.rand(num_samples, device=device) * 25.0
    
    # 3. Sample Actions: Force "Dangerous" Intent
    a_nom_low = torch.rand(n_low, 2, device=device) * 2.0 - 1.0
    
    # CRITICAL CHANGE: In the high-bias group, we force actions to be high (0.5 to 0.9).
    # Since cN is already high here, high fn_phys will trigger the Nitrogen budget.
    a_nom_high = 0.5 + torch.rand(n_high, 2, device=device) * 0.4 
    a_nom = torch.cat([a_nom_low, a_nom_high])
    
    # 4. Denormalize Actions to Physical Units
    a_scaled = (a_nom + 1.0) / 2.0
    i_phys = a_scaled[:, 0] * I_MAX
    fn_phys = a_scaled[:, 1] * FN_MAX
    
    # 5. Apply Physical Constraints with Buffer
    # --- Light Constraint ---
    shading = torch.exp(-ALPHA * cx * L)
    i_limit = I_CRIT / (shading + 1e-6)
    i_safe_phys = torch.minimum(i_phys, i_limit * 0.95)
    
    # --- Nitrogen Constraint: Preventative Budgeting ---
    n_buffer_target = N_LIMIT * 0.95
    fn_budget = torch.clamp(n_buffer_target - cN, min=0.0)
    fn_safe_phys = torch.minimum(fn_phys, fn_budget)

    # Calculate the "Distance to Wall" for the 4th state feature
    n_distance = torch.relu(n_buffer_target - cN)
    n_distance_norm = n_distance / n_buffer_target

    # 6. Re-normalize Safe Physical Actions back to [-1, 1]
    i_safe_norm = (i_safe_phys / I_MAX) * 2.0 - 1.0
    fn_safe_norm = (fn_safe_phys / FN_MAX) * 2.0 - 1.0
    
    # 7. Stack Outputs
    states = torch.stack([cx, cN, cq, n_distance_norm], dim=1)
    nom_actions = a_nom
    target_actions = torch.stack([i_safe_norm, fn_safe_norm], dim=1)
    
    return states, nom_actions, target_actions