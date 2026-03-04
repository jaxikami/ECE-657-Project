import torch

def get_fresh_batch_dataset(num_samples=450000, device='cuda'):
    # 1. Physical Constants (Must match env.py exactly)
    I_MAX = 3000.0
    FN_MAX = 20.0
    I_CRIT = 450.0
    ALPHA = 0.25
    L = 0.5
    N_LIMIT = 180.0

    # 2. Sample State Space with Boundary Bias
    # We want 50% uniform samples and 50% near-limit samples
    half = num_samples // 2
    
    # --- BIAS FOR BIOMASS (cx) --- 
    # High biomass increases shading, making the Light constraint tighter
    cx_uniform = torch.rand(half, device=device) * 6.0
    cx_high = 4.0 + torch.rand(half, device=device) * 2.0 
    cx = torch.cat([cx_uniform, cx_high])

    # --- BIAS FOR NITRATE (cN) ---
    # Bias 50% of samples to be near or above the 180.0 limit
    cN_uniform = torch.rand(half, device=device) * 200.0
    cN_boundary = 160.0 + torch.rand(half, device=device) * 40.0
    cN = torch.cat([cN_uniform, cN_boundary])
    
    cq = torch.rand(num_samples, device=device) * 25.0
    
    # 3. Sample Actions in the Agent's Space [-1, 1]
    # Bias actions toward +1 (maximum intensity/feed) to force the safeguard to work
    a_nom_uniform = torch.rand(half, 2, device=device) * 2.0 - 1.0
    a_nom_high = 0.5 + torch.rand(half, 2, device=device) * 0.5 
    a_nom = torch.cat([a_nom_uniform, a_nom_high])
    
    # 4. Denormalize Actions to Physical Units
    a_scaled = (a_nom + 1.0) / 2.0
    i_phys = a_scaled[:, 0] * I_MAX
    fn_phys = a_scaled[:, 1] * FN_MAX
    
    # 5. Apply Physical Constraints with Buffer
    # LIGHT: Use a 5% buffer to handle the 0.05 noise in env.py
    shading = torch.exp(-ALPHA * cx * L)
    i_limit = I_CRIT / (shading + 1e-6)
    i_safe_phys = torch.minimum(i_phys, i_limit * 0.95) 
    
    # NITRATE: Tighten the shut-off to 170.0 to prevent noise-driven spikes
    fn_safe_phys = torch.where(cN > 180.0 * 0.95, 
                               torch.zeros_like(fn_phys), 
                               fn_phys)

    # 6. Re-normalize Safe Physical Actions back to [-1, 1]
    i_safe_norm = (i_safe_phys / I_MAX) * 2.0 - 1.0
    fn_safe_norm = (fn_safe_phys / FN_MAX) * 2.0 - 1.0
    
    # 7. Stack Outputs
    states = torch.stack([cx, cN, cq], dim=1)
    nom_actions = a_nom 
    target_actions = torch.stack([i_safe_norm, fn_safe_norm], dim=1)
    
    return states, nom_actions, target_actions