import torch
import numpy as np
import os
from env import PhotoProductionEnv
from pretrain import ActionProjectionNetwork 

def run_synchronized_stress_test(num_test_samples=5000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = PhotoProductionEnv(train_mode=False) 
    
    # 1. Load Safeguard and Normalization (Updated state_dim to 4)
    safeguard = ActionProjectionNetwork(state_dim=4, action_dim=2).to(device)
    if not os.path.exists("action_projection_network.pth") or not os.path.exists("norm_constants.npz"):
        print("❌ Error: Missing files. Run pretrain.py first.")
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

    # Helper to calculate the 4th feature (Budget Distance)
    def get_4d_state(cx, cN, cq):
        n_dist = torch.clamp(N_LIMIT_BUFFER - cN, min=0.0) / N_LIMIT_BUFFER
        return torch.stack([cx, cN, cq, n_dist], dim=1)

    # --- TEST A: LIGHT CONSTRAINT (I) ---
    print(f"\n--- [TEST A] Light Constraint Stress Test ({num_test_samples} samples) ---")
    cx_test = torch.linspace(0.1, 6.0, num_test_samples, device=device)
    cN_safe = torch.full((num_test_samples,), 50.0, device=device) 
    cq_rand = torch.rand(num_test_samples, device=device) * 25.0
    
    # Generate 4D state for Test A
    states_i_4d = get_4d_state(cx_test, cN_safe, cq_rand)
    z_intent_max = torch.ones((num_test_samples, 2), device=device) 

    with torch.no_grad():
        s_norm_i = (states_i_4d - s_mean) / (s_std + 1e-8)
        # Safeguard now returns (intent - delta)
        u_safe_i = safeguard(s_norm_i, z_intent_max).cpu().numpy()

    i_violations = 0
    for i in range(num_test_samples):
        shading = np.exp(-ALPHA * cx_test[i].item() * L)
        i_limit_phys = (I_CRIT / (shading + 1e-6)) * 0.95
        i_phys = ((u_safe_i[i, 0] + 1.0) / 2.0) * I_MAX
        
        if i_phys > i_limit_phys + 1e-1:
            i_violations += 1
    
    print(f"Result: {i_violations} Light violations (Fail Rate: {(i_violations/num_test_samples)*100:.2f}%)")

    # --- TEST B: NITROGEN BUDGET (N) ---
    print(f"\n--- [TEST B] Nitrogen Budget Stress Test ({num_test_samples} samples) ---")
    cx_safe = torch.full((num_test_samples,), 1.0, device=device) 
    cN_test = torch.linspace(150.0, 200.0, num_test_samples, device=device)
    
    # Generate 4D state for Test B
    states_n_4d = get_4d_state(cx_safe, cN_test, cq_rand)

    with torch.no_grad():
        s_norm_n = (states_n_4d - s_mean) / (s_std + 1e-8)
        u_safe_n = safeguard(s_norm_n, z_intent_max).cpu().numpy()

    n_violations = 0
    for i in range(num_test_samples):
        current_cN = cN_test[i].item()
        fn_phys = ((u_safe_n[i, 1] + 1.0) / 2.0) * FN_MAX
        
        # Predicted next state must be below 95% threshold
        if (current_cN + fn_phys) > (N_LIMIT_BUFFER + 1e-3):
            n_violations += 1

    print(f"Result: {n_violations} Nitrogen violations (Fail Rate: {(n_violations/num_test_samples)*100:.2f}%)")

    # --- FINAL VERDICT ---
    if i_violations == 0 and n_violations == 0:
        print("\n🚀 EXCELLENT: Both constraints perfectly enforced with Residual Delta logic.")
    else:
        print("\n⚠️ WARNING: Violations detected. Ensure pretrain.py uses Asymmetric Loss.")

if __name__ == "__main__":
    run_synchronized_stress_test()