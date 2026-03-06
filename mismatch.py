import torch
import numpy as np
import os
from pretrain import ActionProjectionNetwork 

def run_synchronized_stress_test(num_test_samples=5000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Safeguard
    safeguard = ActionProjectionNetwork(state_dim=4, action_dim=2).to(device)
    if not os.path.exists("action_projection_network.pth"):
        print("❌ Error: Missing weights. Run pretrain.py first.")
        return

    safeguard.load_state_dict(torch.load("action_projection_network.pth", map_location=device))
    safeguard.eval()
    
    # Physical Constants
    I_MIN, I_MAX = 120.0, 400.0
    FN_MAX = 40.0
    N_LIMIT = 800.0
    RATIO_LIMIT = 0.011
    TOTAL_TIME = 240.0
    CONTROL_FREQ = 20.0
    SIGMA = 0.05  # 5% Gaussian deviation

    def denorm_fn(val_norm):
        return ((val_norm + 1.0) / 2.0) * FN_MAX

    def denorm_i(val_norm):
        return I_MIN + ((val_norm + 1.0) / 2.0) * (I_MAX - I_MIN)

    def gaussian_sample(mean, std_scale=SIGMA):
        """Helper to sample around a mean with 0.05 relative deviation."""
        return torch.normal(mean, mean * std_scale, (num_test_samples,), device=device)

    # --- TEST 1: Nitrate Accumulation (20h Window) ---
    print(f"\n--- [TEST 1] g1: 20h Predictive Budget (Target < {N_LIMIT*0.995}) ---")
    t_norm = torch.rand(num_test_samples, device=device)
    t_phys = t_norm * TOTAL_TIME
    delta_t = torch.ceil(t_phys / CONTROL_FREQ) * CONTROL_FREQ - t_phys
    delta_t = torch.where(delta_t == 0, torch.tensor(CONTROL_FREQ, device=device), delta_t)
    
    # Target N levels to test budget (750 to 760)
    cN_test = 750.0 + torch.rand(num_test_samples, device=device) * 10.0
    
    # STOCHASTIC SAFE STATES for cx and cq
    cx_safe_1 = gaussian_sample(1.0).clamp(0.1, 6.0)
    cq_safe_1 = gaussian_sample(0.005).clamp(0.0, 0.01)
    
    z_intent = torch.ones((num_test_samples, 2), device=device) # Max Feed Intent

    with torch.no_grad():
        s_phys_1 = torch.stack([cx_safe_1, cN_test, cq_safe_1, t_norm], dim=1)
        u_safe_1 = safeguard(s_phys_1, z_intent).cpu().numpy()

    g1_passes = 0
    target_limit = N_LIMIT * 0.995 # 796 mg/L
    for i in range(num_test_samples):
        fn_phys = denorm_fn(u_safe_1[i, 1])
        if (cN_test[i].item() + (fn_phys * delta_t[i].item())) < target_limit:
            g1_passes += 1
    print(f"Result: {g1_passes}/{num_test_samples} passed (20h state < {target_limit})")

    # --- TEST 2: Bioproduct Ratio Constraint (g2) ---
    print(f"\n--- [TEST 2] g2: Ratio Violation (I within 2% of I_min) ---")
    cx_test_2 = 1.0 + torch.rand(num_test_samples, device=device) * 5.0
    cq_violating = cx_test_2 * RATIO_LIMIT * 0.99 # Violation point
    
    # STOCHASTIC SAFE STATE for cN
    cN_safe_2 = gaussian_sample(100.0).clamp(0.0, 200.0)
    t_zero = torch.zeros(num_test_samples, device=device)
    
    with torch.no_grad():
        s_phys_2 = torch.stack([cx_test_2, cN_safe_2, cq_violating, t_zero], dim=1)
        u_safe_2 = safeguard(s_phys_2, z_intent).cpu().numpy()

    g2_passes = 0
    i_pass_threshold = I_MIN + (I_MIN * 0.02) # 122.4
    for i in range(num_test_samples):
        i_phys = denorm_i(u_safe_2[i, 0])
        if i_phys <= i_pass_threshold:
            g2_passes += 1
    print(f"Result: {g2_passes}/{num_test_samples} passed (I <= {i_pass_threshold:.2f})")

    # --- TEST 3: Identity Mapping (Safe Region) ---
    print(f"\n--- [TEST 3] Identity Mapping: Stochastic Safe Zone (Diff <= 1%) ---")
    
    # Gaussian sampling for ALL states
    cx_safe_3 = gaussian_sample(2.0).clamp(0.1, 6.0)
    cN_safe_3 = gaussian_sample(50.0).clamp(0.0, 150.0)
    cq_safe_3 = gaussian_sample(0.005).clamp(0.0, 0.01)
    t_safe_3 = gaussian_sample(0.2).clamp(0.0, 1.0)
    
    z_safe_intent = -0.5 + torch.rand((num_test_samples, 2), device=device) * 1.0

    with torch.no_grad():
        s_safe = torch.stack([cx_safe_3, cN_safe_3, cq_safe_3, t_safe_3], dim=1)
        u_safe_3 = safeguard(s_safe, z_safe_intent)
    
    diff = torch.abs(u_safe_3 - z_safe_intent)
    identity_passes = torch.all(diff <= 0.02, dim=1).sum().item()
    max_dev = diff.max().item()

    print(f"Max Absolute Deviation: {max_dev:.6e}")
    print(f"Result: {identity_passes}/{num_test_samples} points within 1% tolerance.")

if __name__ == "__main__":
    run_synchronized_stress_test()