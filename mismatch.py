import torch
import numpy as np
import os
from env import PhotoProductionEnv
from pretrain import ActionProjectionNetwork 

def run_synchronized_stress_test(num_test_samples=5000):
    # 1. Setup Environment and Hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = PhotoProductionEnv(train_mode=False) # No noise for testing
    
    # 2. Load the Safeguard Weights
    safeguard = ActionProjectionNetwork(state_dim=3, action_dim=2, latent_dim=512).to(device)
    
    if not os.path.exists("action_projection_network.pth") or not os.path.exists("norm_constants.npz"):
        print("❌ Error: Missing weights or normalization constants. Run pretrain.py first.")
        return

    safeguard.load_state_dict(torch.load("action_projection_network.pth", map_location=device))
    safeguard.eval()

    # 3. Load OPTION B Normalization Constants
    norms = np.load("norm_constants.npz")
    s_mean = torch.tensor(norms['s_mean'], dtype=torch.float32).to(device)
    s_std = torch.tensor(norms['s_std'], dtype=torch.float32).to(device)
    a_mean = torch.tensor(norms['a_mean'], dtype=torch.float32).to(device)
    a_std = torch.tensor(norms['a_std'], dtype=torch.float32).to(device)
    
    print(f"✅ Loaded Final Buffer Stats: Mean={s_mean.cpu().numpy()}")

    # 4. Generate "Extreme Danger" States
    # Matches the 90% bias range from data_gen.py
    cx_high = 4.0 + torch.rand(num_test_samples, device=device) * 2.0  # [4.0, 6.0]
    cN_danger = 175.0 + torch.rand(num_test_samples, device=device) * 25.0 # [175, 200]
    cq_rand = torch.rand(num_test_samples, device=device) * 25.0
    states_danger = torch.stack([cx_high, cN_danger, cq_rand], dim=1)

    # 5. Intent: Full Production (This would normally trigger violations)
    # Intent z is in range [-1, 1]
    z_intent = torch.ones((num_test_samples, 2), device=device) 

    # 6. Perform Safety Projection
    with torch.no_grad():
        # Step A: Normalize states using the Option B "Ruler"
        s_norm = (states_danger - s_mean) / (s_std + 1e-8)
        
        # Step B: Project the intent
        u_safe_norm = safeguard(s_norm, z_intent) 
        
        # Step C: Denormalize back to [-1, 1] for the environment
        u_safe_cpu = u_safe_norm.cpu().numpy()

    # 7. Physical Validation in the Environment
    violations = 0
    i_penalties = []
    n_penalties = []

    print(f"--- Running Stress Test on {num_test_samples} Samples ---")
    for i in range(num_test_samples):
        # Manually override env state to our danger sample
        env.state = states_danger[i].cpu().numpy()
        
        # Execute step with the safeguard's action
        _, _, _, info = env.step(u_safe_cpu[i])
        
        if not info['is_safe']:
            violations += 1
            # Track how "bad" the violation was
            if info['penalties']['safety'] < 0:
                # Based on env.py: penalty_safety = -(1.0 + i_penalty + n_penalty)/5
                # We can roughly estimate severity from the info dict
                pass

    # 8. Final Report
    fail_rate = (violations / num_test_samples) * 100
    print(f"\n[STRESS TEST RESULTS]")
    print(f"Failure Rate: {fail_rate:.4f}%")
    
    if fail_rate < 0.05:
        print("🚀 EXCELLENT: The safeguard has mastered the final biased distribution.")
    elif fail_rate < 1.0:
        print("⚠️ WARNING: Minor violations at extreme edges. Check if SmoothL1 beta is too high.")
    else:
        print("❌ CRITICAL: Normalization mismatch or underfitting. Retrain with more epochs.")

if __name__ == "__main__":
    run_synchronized_stress_test()