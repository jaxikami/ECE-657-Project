import torch

def get_fresh_batch_dataset(num_samples=450000, bias=0.5, device='cuda'):
    """
    Generates a fresh batch of synthetic environmental states and safe target actions.
    
    The generation follows these steps:
    1. Define physical constants and limits for the bioreactor.
    2. Sample the system states (Biomass, Nitrate, Bioproduct, Normalized Time). 
       A portion of the samples are biased near the constraint boundaries to improve 
       the neural network's learning of critical constraint threshold areas.
    3. Sample random nominal actions in normalized space [-1, 1].
    4. Denormalize actions to their physical equivalent ranges.
    5. Apply the safety constraints (Path constraints and Terminal constraints) to 
       project the nominal actions into safe 'target' actions.
    6. Re-normalize the safe target actions back to [-1, 1] for training.
    
    Args:
        num_samples (int): Total number of data samples to generate.
        bias (float): Fraction of samples (0 to 1) deliberately placed near the 
                      Nitrate path constraint boundary.
        device (str): Compute device for PyTorch tensors ('cuda' or 'cpu').
        
    Returns:
        tuple: (states, nom_actions, target_actions)
            - states (Tensor): Shape (N, 4). Contains [Biomass, Nitrate, Bioproduct, Time].
            - nom_actions (Tensor): Shape (N, 2). Original random nominal actions [-1, 1].
            - target_actions (Tensor): Shape (N, 2). Safe actions after applying constraints [-1, 1].
    """
    # =========================================================================
    # 1. Physical Constants & Operational Limits
    # Derived from the specific bioreactor case study parameters:
    # =========================================================================
    I_MIN = 120.0               # Minimum light intensity allowed (uE/m2/s)
    I_MAX = 400.0               # Maximum light intensity allowed (uE/m2/s)
    FN_MAX = 40.0               # Maximum allowable nitrate feed rate (mg/L/h)
    N_LIMIT_PATH = 800.0        # Constraint g1: Maximum allowed nitrate concentration in the tank (mg/L)
    RATIO_LIMIT = 0.011         # Constraint g2: Bioproduct/biomass ratio limit (historic reference, currently inactive)
    CONTROL_INTERVAL = 20.0     # Time step / predictive window duration for control decisions (hours)
    SAFE_BUFFER = 0.98          # 2% safety margin applied to constraint thresholds to avoid hard physical limits
    TOTAL_TIME = 240.0          # Total duration of an episode (hours)
    N_LIMIT_TERM = 150.0        # Constraint g3: Terminal nitrate limit at the end of the episode (mg/L)

    # =========================================================================
    # 2. State Sampling
    # Randomly initialize states across operational ranges.
    # We apply a 'bias' to intentionally sample states near the critical nitrate limit.
    # =========================================================================
    n_high = int(num_samples * bias)  # Number of samples near the constraint boundary
    n_low = num_samples - n_high      # Number of samples uniformly distributed
    
    # Biomass (cx): Sampled linearly between 0 and 6 g/L
    cx = torch.rand(num_samples, device=device) * 6.0

    # Nitrate (cN): Segmented sampling to focus on the 800 mg/L limit
    # First portion is uniformly spread from 0 to 800 mg/L
    cN_uniform = torch.rand(n_low, device=device) * N_LIMIT_PATH
    # Second portion is clustered tightly between 750 and 800 mg/L (the danger zone)
    cN_boundary = 750.0 + torch.rand(n_high, device=device) * 50.0
    # Concatenate to form the full batch
    cN = torch.cat([cN_uniform, cN_boundary])
    
    # Bioproduct (cq): Uniformly sampled between 0 and 0.1 g/L
    # Explicit focus on ratio limits was dropped to allow unconstrained exploration here
    cq = torch.rand(num_samples, device=device) * 0.1
    
    # Normalized Time (t_norm): Represented as a fraction from 0.0 (start) to 1.0 (end)
    t_norm = torch.rand(num_samples, device=device)

    # =========================================================================
    # 3. Sample Nominal Actions 
    # These represent the 'intent' (z) of the agent before safety checks.
    # Uniformly distributed in the neural network's action space [-1, 1].
    # =========================================================================
    a_nom = torch.rand(num_samples, 2, device=device) * 2.0 - 1.0
    
    # =========================================================================
    # 4. Denormalize Actions to Physical Units
    # Transform policy outputs [-1, 1] back into actual physics scales [MIN, MAX]
    # =========================================================================
    a_scaled = (a_nom + 1.0) / 2.0  # Shift and scale to [0, 1]
    
    # Map to light intensity [120, 400]
    i_phys = I_MIN + a_scaled[:, 0] * (I_MAX - I_MIN)
    # Map to Nitrate feed rate [0, 40]
    fn_phys = a_scaled[:, 1] * FN_MAX
    
    # =========================================================================
    # 5. Apply Constraints to Determine Safe "Target" Actions
    # Here we project the nominal actions to safe actions using known physics.
    # =========================================================================
    
    # --- Constraint g1: Nitrate Path Constraint (20h Accumulation) ---
    # We must ensure that current_N + (Feed_Rate * 20h) <= 800 * 0.98.
    # Determine the absolute maximum allowable nitrate reading:
    n_max_allowed = N_LIMIT_PATH * SAFE_BUFFER
    # Calculate the maximum feed rate that exactly hits the limit over the control interval
    fn_budget = (n_max_allowed - cN) / CONTROL_INTERVAL
    # Ensure the calculated feed budget stays within physical actuator limits [0, 40]
    fn_safe_phys = torch.clamp(fn_budget, min=0.0, max=FN_MAX)
    # The projected safe target is the minimum between what we *want* to do and what is *safe*
    fn_target_phys = torch.minimum(fn_phys, fn_safe_phys)

    # --- Constraint g3: Terminal Nitrate Constraint ---
    # At the end of the episode (240h), the left-over nitrate must be < 150 mg/L.
    # Convert normalized time back to actual hours to deduce time remaining
    t_phys = t_norm * TOTAL_TIME
    time_remaining = TOTAL_TIME - t_phys
    
    # We only actively intervene on g3 when approaching the end of the episode 
    # (i.e. we are inside the final control interval window).
    near_end = time_remaining <= CONTROL_INTERVAL
    
    # Required MAXIMUM feed to avoid exceeding the terminal 150 limit (with a 5% safety buffer matching env.py).
    # Logic: current_nitrate + feed_rate * time_remaining <= 150 * 0.95
    # Thus: feed_rate <= (150 * 0.95 - current_nitrate) / time_remaining.
    # We clamp time_remaining from below (min=1.0) to prevent division by zero gracefully.
    safe_time_remaining = torch.clamp(time_remaining, min=1.0)
    
    fn_max_term = (N_LIMIT_TERM * 0.95 - cN) / safe_time_remaining
    fn_max_term = torch.clamp(fn_max_term, min=0.0)
    
    # If we are near the end, our TARGET feed must be AT MOST the allowed terminal amount.
    # We use torch.where to apply this rule strictly to the 'near_end' batch indices.
    fn_target_phys = torch.where(
        near_end,
        torch.minimum(fn_max_term, fn_target_phys),  # Apply strict end-of-episode cap
        fn_target_phys                               # Keep standard running operation target
    )

    i_target_phys = i_phys

    # =========================================================================
    # 6. Re-normalize Target Actions back to [-1, 1]
    # The dataset requires features scaled to neural network friendly limits.
    # =========================================================================
    i_target_norm = ((i_target_phys - I_MIN) / (I_MAX - I_MIN)) * 2.0 - 1.0
    fn_target_norm = (fn_target_phys / FN_MAX) * 2.0 - 1.0
    
    # =========================================================================
    # 7. Stack Outputs
    # Combine individual vectors into batched 2D tensors for returning.
    # =========================================================================
    states = torch.stack([cx, cN, cq, t_norm], dim=1)
    nom_actions = a_nom
    target_actions = torch.stack([i_target_norm, fn_target_norm], dim=1)
    
    return states, nom_actions, target_actions