import torch

def get_fresh_batch_dataset(num_samples=450000, device='cuda'):
    """
    GPU-accelerated dataset generation for the RX 9070 XT.
    Generates and constrains data directly in VRAM to eliminate 
    PCIe transport bottlenecks.
    """
    # 1. Physical Constants from the pc-gym environment [cite: 10, 11]
    I_crit = 450.0  # Critical light intensity [cite: 11]
    alpha = 0.25    # Light attenuation coefficient [cite: 11]
    L = 0.5         # Path length [cite: 11]
    Fn_max = 10.0   # Maximum nitrate feed rate [cite: 11]
    
    # 2. Sample State Space directly on GPU [cite: 14]
    # Standardizing ranges based on environment dynamics [cite: 11]
    cx = torch.rand(num_samples, device=device) * (6.0 - 0.05) + 0.05  # Biomass
    cN = torch.rand(num_samples, device=device) * 200.0                # Nitrate
    cq = torch.rand(num_samples, device=device) * 25.0                 # Phycocyanin
    
    # 3. Sample Nominal "Unsafe" Actions [cite: 13]
    i_nom = torch.rand(num_samples, device=device) * 3000.0  # Light intensity
    fn_nom = torch.rand(num_samples, device=device) * 20.0   # Feed rate
    
    # 4. Vectorized Safety Projection (The Target Manifold) 
    # Photoinhibition Constraint Calculation [cite: 11]
    shading_factor = torch.exp(-alpha * cx * L)
    i_limit = I_crit / (shading_factor + 1e-6)
    i_safe = torch.minimum(i_nom, i_limit)
    
    # Nitrate Toxicity Constraint [cite: 11]
    # If cN > 180, set feed to 0 to prevent toxicity [cite: 11]
    fn_limit = torch.where(cN > 180.0, 
                           torch.tensor(0.0, device=device), 
                           torch.tensor(Fn_max, device=device))
    fn_safe = torch.minimum(fn_nom, fn_limit)
    
    # 5. Stack and Return Tensors 
    # Returns are already on the GPU, ready for ActionProjectionNetwork
    states = torch.stack([cx, cN, cq], dim=1)
    nom_actions = torch.stack([i_nom, fn_nom], dim=1)
    target_actions = torch.stack([i_safe, fn_safe], dim=1)
    
    return states, nom_actions, target_actions