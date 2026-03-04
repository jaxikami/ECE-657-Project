import torch
import numpy as np
import os
from env import PhotoProductionEnv
from pretrain import ActionProjectionNetwork 

def run_synchronized_stress_test(num_test_samples=5000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = PhotoProductionEnv(train_mode=False) 
    
    # 1. Load Safeguard and Normalization
    safeguard = ActionProjectionNetwork(state_dim=3, action_dim=2).to(device)
    if not os.path.exists("action_projection_network.pth") or not os.path.exists("norm_constants.npz"):
        print("❌ Error: Missing files.")
        return

    safeguard.load_state_dict(torch.load("action_projection_network.pth", map_location=device))
    safeguard.eval()

    norms = np.load("norm_constants.npz")
    s_mean = torch.tensor(norms['s_mean'], dtype=torch.float32).to(device)
    s_std = torch.tensor(norms['s_std'], dtype=torch.float32).to(device)
    
    # Constants from env.py / data_gen.py
    I_MAX, FN_MAX = 3000.0, 20.0
    I_CRIT, ALPHA, L = 450.0, 0.25, 0.5
    N_LIMIT_BUFFER = 180.0 * 0.95

    # --- TEST A: LIGHT CONSTRAINT (I) ---
    print(f"\n--- [TEST A] Light Constraint Stress Test ({num_test_samples} samples) ---")
    # Vary biomass widely to change the shading limit
    cx_test = torch.linspace(0.1, 6.0, num_test_samples, device=device)
    cN_safe = torch.full((num_test_samples,), 50.0, device=device) # Safe N levels
    cq_rand = torch.rand(num_test_samples, device=device) * 25.0
    states_i = torch.stack([cx_test, cN_safe, cq_rand], dim=1)
    
    z_intent_max = torch.ones((num_test_samples, 2), device=device) # Request max I and FN

    with torch.no_grad():
        s_norm_i = (states_i - s_mean) / (s_std + 1e-8)
        u_safe_i = safeguard(s_norm_i, z_intent_max).cpu().numpy()

    i_violations = 0
    for i in range(num_test_samples):
        # Calculate physical limit: I_limit = I_crit / exp(-alpha * cx * L)
        shading = np.exp(-ALPHA * cx_test[i].item() * L)
        i_limit_phys = (I_CRIT / (shading + 1e-6)) * 0.95
        
        # Denormalize safeguard output to physical units
        i_phys = ((u_safe_i[i, 0] + 1.0) / 2.0) * I_MAX
        
        if i_phys > i_limit_phys + 1e-1:
            i_violations += 1
    
    print(f"Result: {i_violations} Light violations (Fail Rate: {(i_violations/num_test_samples)*100:.2f}%)")

    # --- TEST B: NITROGEN BUDGET (N) ---
    print(f"\n--- [TEST B] Nitrogen Budget Stress Test ({num_test_samples} samples) ---")
    # Push N levels close to and above the buffer
    cx_safe = torch.full((num_test_samples,), 1.0, device=device) # Low biomass (safe light)
    cN_test = torch.linspace(150.0, 200.0, num_test_samples, device=device)
    states_n = torch.stack([cx_safe, cN_test, cq_rand], dim=1)

    with torch.no_grad():
        s_norm_n = (states_n - s_mean) / (s_std + 1e-8)
        u_safe_n = safeguard(s_norm_n, z_intent_max).cpu().numpy()

    n_violations = 0
    for i in range(num_test_samples):
        current_cN = cN_test[i].item()
        # Safe fN = max(0, Budget) where Budget = (Limit * 0.95) - current_N
        fn_phys = ((u_safe_n[i, 1] + 1.0) / 2.0) * FN_MAX
        
        if (current_cN + fn_phys) > (N_LIMIT_BUFFER + 1e-3):
            n_violations += 1

    print(f"Result: {n_violations} Nitrogen violations (Fail Rate: {(n_violations/num_test_samples)*100:.2f}%)")

    # --- FINAL VERDICT ---
    if i_violations == 0 and n_violations == 0:
        print("\n🚀 EXCELLENT: Both constraints are perfectly enforced.")
    else:
        print("\n⚠️ WARNING: One or more constraints failed. Check normalization or training epochs.")

if __name__ == "__main__":
    run_synchronized_stress_test()