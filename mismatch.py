import torch
import numpy as np
import os
# We assume the new ActionProjectionNetwork class is available in pretrain.py
from pretrain import ActionProjectionNetwork 

def static_normalize(states):
    """Matches the logic in pretrain.py exactly."""
    max_vals = torch.tensor([6.0, 170.0, 25.0, 1.0], device=states.device)
    return (states / max_vals) * 2.0 - 1.0

def run_synchronized_stress_test(num_test_samples=5000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Safeguard
    safeguard = ActionProjectionNetwork(state_dim=4, action_dim=2).to(device)
    if not os.path.exists("action_projection_network.pth"):
        print("❌ Error: Missing weights. Run the NEW pretrain.py first.")
        return

    safeguard.load_state_dict(torch.load("action_projection_network.pth", map_location=device))
    safeguard.eval()
    
    # Constants for physical verification (from env.py)
    I_MAX, FN_MAX = 3000.0, 20.0
    I_CRIT, ALPHA, L = 450.0, 0.25, 0.5
    N_LIMIT_BUFFER = 180 * 0.95

    def get_4d_state_norm(cx, cN, cq):
        n_dist = torch.clamp(N_LIMIT_BUFFER - cN, min=0.0) / N_LIMIT_BUFFER
        states_raw = torch.stack([cx, cN, cq, n_dist], dim=1)
        return static_normalize(states_raw)

    # --- TEST A: LIGHT CONSTRAINT (I) ---
    print(f"\n--- [TEST A] Light Constraint Stress Test ({num_test_samples} samples) ---")
    cx_test = torch.linspace(0.1, 6.0, num_test_samples, device=device)
    cN_safe = torch.full((num_test_samples,), 50.0, device=device) 
    cq_rand = torch.rand(num_test_samples, device=device) * 25.0
    
    z_intent_max = torch.ones((num_test_samples, 2), device=device) 

    with torch.no_grad():
        s_norm_i = get_4d_state_norm(cx_test, cN_safe, cq_rand)
        u_safe_i = safeguard(s_norm_i, z_intent_max).cpu().numpy()

    i_violations = 0
    for i in range(num_test_samples):
        shading = np.exp(-ALPHA * cx_test[i].item() * L)
        # We test against 0.96 because training was at 0.95 (1% tolerance)
        i_limit_phys = (I_CRIT / (shading + 1e-6)) * 0.96
        i_phys = ((u_safe_i[i, 0] + 1.0) / 2.0) * I_MAX
        
        if i_phys > i_limit_phys + 1e-1:
            i_violations += 1
    
    print(f"Result: {i_violations} Light violations (Fail Rate: {(i_violations/num_test_samples)*100:.2f}%)")

    # --- TEST B: NITROGEN BUDGET (N) ---
    print(f"\n--- [TEST B] Nitrogen Budget Stress Test ({num_test_samples} samples) ---")
    cx_safe = torch.full((num_test_samples,), 1.0, device=device) 
    cN_test = torch.linspace(150.0, 170.0, num_test_samples, device=device)
    
    with torch.no_grad():
        s_norm_n = get_4d_state_norm(cx_safe, cN_test, cq_rand)
        u_safe_n = safeguard(s_norm_n, z_intent_max).cpu().numpy()

    n_violations = 0
    for i in range(num_test_samples):
        current_cN = cN_test[i].item()
        fn_phys = ((u_safe_n[i, 1] + 1.0) / 2.0) * FN_MAX
        
        # Predicted next state must be below 96% threshold
        if (current_cN + fn_phys) > (180.0 * 0.96):
            n_violations += 1

    print(f"Result: {n_violations} Nitrogen violations (Fail Rate: {(n_violations/num_test_samples)*100:.2f}%)")
    
    # --- TEST C: IDENTITY MAPPING CHECK (0.1% Threshold) ---
    print(f"\n--- [TEST C] Identity Mapping Test ({num_test_samples} samples) ---")
    cx_safe = torch.rand(num_test_samples, device=device) * 2.0
    cN_safe = torch.rand(num_test_samples, device=device) * 50.0
    
    z_intent_safe = -0.8 + torch.rand((num_test_samples, 2), device=device) * 0.3

    with torch.no_grad():
        s_norm = get_4d_state_norm(cx_safe, cN_safe, cq_rand)
        u_safe_out = safeguard(s_norm, z_intent_safe)

    diff = torch.abs(u_safe_out - z_intent_safe)
    rel_diff = diff / (torch.abs(z_intent_safe) + 1e-8)
    
    identity_violations = torch.sum(rel_diff > 0.001).item()
    max_val_diff = torch.max(diff).item()

    print(f"Max Absolute Deviation: {max_val_diff:.6f}")
    print(f"Result: {identity_violations} Identity violations > 0.1%")

    if i_violations == 0 and n_violations == 0 and identity_violations == 0:
        print("\n🚀 SUCCESS: Static Normalization + Residual logic passed all tests.")
    else:
        print("\n⚠️ WARNING: Minor violations remain. Consider increasing safety_loss weight.")

if __name__ == "__main__":
    run_synchronized_stress_test()