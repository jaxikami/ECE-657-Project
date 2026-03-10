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
    
    # =========================================================================
    # 1. Physical Constants & Operational Limits
    # Derived from the specific bioreactor case study parameters.
    # =========================================================================
    I_MIN, I_MAX = 120.0, 400.0   # Light intensity bounds
    FN_MAX = 40.0                 # Max allowable nitrate feed rate
    N_LIMIT = 800.0               # Constraint g1: Max allowed nitrate concentration
    RATIO_LIMIT = 0.011           # Constraint g2 target
    TOTAL_TIME = 240.0            # Total duration of an episode (hours)
    CONTROL_FREQ = 20.0           # 20-hour control decision window
    SIGMA = 0.05                  # 5% Gaussian deviation used for exploring edge cases

    # Helper functions to convert normalized actions [-1, 1] back to physical quantities
    def denorm_fn(val_norm):
        return ((val_norm + 1.0) / 2.0) * FN_MAX

    def denorm_i(val_norm):
        return I_MIN + ((val_norm + 1.0) / 2.0) * (I_MAX - I_MIN)

    def gaussian_sample(mean, std_scale=SIGMA):
        """Helper to sample states around a specific mean with a given relative deviation."""
        return torch.normal(mean, mean * std_scale, (num_test_samples,), device=device)

    # =========================================================================
    # --- TEST 1: Nitrate Accumulation (20h Window) Penalty g1 ---
    # Purpose: Verify the filter securely restricts feed intent so that accumulation 
    # over the upcoming control window stays under the absolute safety limit limit.
    # =========================================================================
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

    # =========================================================================
    # --- TEST 2: Terminal Nitrate Constraint (g3) ---
    # Purpose: Verify the network prevents greedily maximizing feed at the very
    # end of an episode, guaranteeing the terminal nitrate budget is satisfied.
    # =========================================================================
    N_LIMIT_TERM = 150.0
    print(f"\n--- [TEST 2] g3: Terminal Nitrate Protection (Target < {N_LIMIT_TERM}) ---")
    
    # Generate states very close to the end of the episode (e.g. last 1-2 hours)
    t_norm_terminal = 0.9 + torch.rand(num_test_samples, device=device) * 0.05
    t_phys_term = t_norm_terminal * TOTAL_TIME
    delta_t_term = TOTAL_TIME - t_phys_term # Time remaining until terminal check
    
    # Generate Nitrate levels dangerously close to the 142.5 limit, but ensure it's mathematically possible to satisfy
    # by keeping starting levels strictly <= 142.5 (since we cannot have negative feed)
    cN_test_3 = (N_LIMIT_TERM * 0.98) - torch.rand(num_test_samples, device=device) * 5.0
    
    # STOCHASTIC SAFE STATES for cx and cq
    cx_safe_3 = gaussian_sample(2.0).clamp(0.1, 6.0)
    cq_safe_3 = gaussian_sample(0.005).clamp(0.0, 0.01)
    
    # Agent intent: Wants to minimize growth (Min Light) and provide MAX Nitrate Feed (1.0)
    # This greedy behavior would normally accumulate Nitrate above the 150 limit.
    z_intent_3 = torch.zeros((num_test_samples, 2), device=device)
    z_intent_3[:, 0] = -1.0 
    z_intent_3[:, 1] = 1.0 
    
    with torch.no_grad():
        s_phys_3 = torch.stack([cx_safe_3, cN_test_3, cq_safe_3, t_norm_terminal], dim=1)
        u_safe_3 = safeguard(s_phys_3, z_intent_3).cpu().numpy()

    # Verify that the safeguard forced the Nitrate Feed to be low enough
    g3_passes = 0
    target_limit_term = N_LIMIT_TERM * 0.98
    for i in range(num_test_samples):
        fn_phys = denorm_fn(u_safe_3[i, 1])
        # Simple heuristic check: the added feed over the remaining time should keep us below the limit
        if (cN_test_3[i].item() + (fn_phys * delta_t_term[i].item())) <= target_limit_term:
            g3_passes += 1
            
    print(f"Result: {g3_passes}/{num_test_samples} passed (Terminal state projected < {target_limit_term})")

    # =========================================================================
    # --- TEST 3: Identity Mapping (Safe Region) ---
    # Purpose: Prove that the neural network's architecture (skip connections) 
    # prevents it from interfering with intents when perfectly safe.
    # =========================================================================
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