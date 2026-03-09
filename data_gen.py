import torch

def get_fresh_batch_dataset(num_samples=450000, bias=0.5, device='cuda'):
    # 1. Physical Constants (Directly from the article case study)
    I_MIN = 120.0               # Absolute physical minimum light 
    I_MAX = 400.0               # Absolute physical maximum light 
    FN_MAX = 40.0               # Absolute physical maximum nitrate feed [cite: 2136]
    N_LIMIT_PATH = 800.0        # g1: Nitrate path constraint [cite: 2120]
    RATIO_LIMIT = 0.011         # g2: Bioproduct/biomass ratio limit [cite: 2128]
    CONTROL_INTERVAL = 20.0     # 20-hour predictive window for nitrate budget
    SAFE_BUFFER = 0.98          # 2% safety room for constraint boundaries

    # 2. Define High/Low Bias Split
    n_high = int(num_samples * bias)
    n_low = num_samples - n_high
    
    # --- State Sampling ---
    # Biomass (cx): 0 to 6 g/L
    cx = torch.rand(num_samples, device=device) * 6.0

    # Nitrate (cN): Focused sampling near the 800 mg/L limit
    cN_uniform = torch.rand(n_low, device=device) * N_LIMIT_PATH
    cN_boundary = 750.0 + torch.rand(n_high, device=device) * 50.0
    cN = torch.cat([cN_uniform, cN_boundary])
    
    # Bioproduct (cq): Uniformly sample across its range (no focus on ratio limits anymore)
    cq = torch.rand(num_samples, device=device) * 0.1
    
    # t_norm: Normalized Time (0 to 1) for finite horizon
    t_norm = torch.rand(num_samples, device=device)

    # 3. Sample Actions (Intent z) in [-1, 1]
    a_nom = torch.rand(num_samples, 2, device=device) * 2.0 - 1.0
    
    # 4. Denormalize Actions to Physical Units
    # Using the full absolute range [120, 400] as requested
    a_scaled = (a_nom + 1.0) / 2.0
    i_phys = I_MIN + a_scaled[:, 0] * (I_MAX - I_MIN)
    fn_phys = a_scaled[:, 1] * FN_MAX
    
    # 5. Apply Constraints to Determine Target Actions
    
    # --- g1: Nitrate Path Constraint (20h Accumulation) ---
    # Ensures current_N + (Fn * 20h) <= 800 * 0.98
    n_max_allowed = N_LIMIT_PATH * SAFE_BUFFER
    fn_budget = (n_max_allowed - cN) / CONTROL_INTERVAL
    fn_safe_phys = torch.clamp(fn_budget, min=0.0, max=FN_MAX)
    fn_target_phys = torch.minimum(fn_phys, fn_safe_phys)

    # --- g3: Terminal Nitrate Constraint (End of Episode limit < 150) ---
    TOTAL_TIME = 240.0
    N_LIMIT_TERM = 150.0
    # Calculate time remaining in episode
    t_phys = t_norm * TOTAL_TIME
    time_remaining = TOTAL_TIME - t_phys
    
    # We only care about g3 if we are approaching the end of the episode
    # E.g. within the last control interval (20h)
    near_end = time_remaining <= CONTROL_INTERVAL
    
    # Required MAXIMUM feed to avoid exceeding 150 * 0.95 (safety buffer matching env.py) at the absolute end
    # Assuming some small baseline consumption, but primarily we just need to ensure current + feed * time <= limit
    # fn * time_remaining <= N_LIMIT_TERM * 0.95 - cN
    # fn <= (150 * 0.95 - cN) / time_remaining
    # To prevent div by 0 at the exact end, clamp time_remaining from below
    safe_time_remaining = torch.clamp(time_remaining, min=1.0)
    
    fn_max_term = (N_LIMIT_TERM * 0.95 - cN) / safe_time_remaining
    fn_max_term = torch.clamp(fn_max_term, min=0.0)
    
    # If we are near the end, our TARGET feed must be AT MOST the allowed amount to avoid g3 violation
    fn_target_phys = torch.where(
        near_end,
        torch.minimum(fn_max_term, fn_target_phys),
        fn_target_phys
    )
    
    # We no longer force the Safeguard to drop Light to I_MIN for g2.
    # The physical kinetics show that dropping to I_MIN actually worsens the ratio!
    # By leaving Light unchanged here, the RL Agent will naturally learn to balance 
    # the Tradeoff correctly using its environmental reward signal.
    i_target_phys = i_phys

    # 6. Re-normalize Target Actions back to [-1, 1]
    # Normalization matches the full absolute physical range
    i_target_norm = ((i_target_phys - I_MIN) / (I_MAX - I_MIN)) * 2.0 - 1.0
    fn_target_norm = (fn_target_phys / FN_MAX) * 2.0 - 1.0
    
    # 7. Stack Outputs
    states = torch.stack([cx, cN, cq, t_norm], dim=1)
    nom_actions = a_nom
    target_actions = torch.stack([i_target_norm, fn_target_norm], dim=1)
    
    return states, nom_actions, target_actions