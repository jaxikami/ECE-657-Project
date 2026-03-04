import torch
import numpy as np
from env import PhotoProductionEnv
from pretrain import ActionProjectionNetwork # Ensure this matches your pretrain class name

def check_stress_test_alignment(num_test_samples=1000):
    # 1. Setup Environment and Load Weights
    env = PhotoProductionEnv(train_mode=False)
    safeguard = ActionProjectionNetwork(state_dim=3, action_dim=2, latent_dim=512)
    
    try:
        safeguard.load_state_dict(torch.load("action_projection_network.pth"))
        safeguard.eval()
        norms = np.load("norm_constants.npz")
        s_mean, s_std = torch.tensor(norms['s_mean']), torch.tensor(norms['s_std'])
    except Exception as e:
        print(f"❌ Setup Error: {e}")
        return

    print(f"--- Running Stress Test on {num_test_samples} Boundary Samples ---")

    # 2. Generate "Danger Zone" States (Nitrate near 180.0, Biomass high)
    # Sampling Nitrate in the 175-185 range to see how it handles the "cliff"
    cN_danger = 175.0 + torch.rand(num_test_samples) * 10.0
    cx_high = 4.0 + torch.rand(num_test_samples) * 2.0
    cq_rand = torch.rand(num_test_samples) * 25.0
    
    states_danger = torch.stack([cx_high, cN_danger, cq_rand], dim=1)
    
    # 3. Generate "Aggressive" Intent (Agent wants max Light and max Feed)
    # Action space is [-1, 1], so [1.0, 1.0] is the most dangerous intent
    z_intent = torch.ones((num_test_samples, 2)) 

    # 4. Batch Projection
    with torch.no_grad():
        # Using the same normalization logic as training
        s_norm = (states_danger - s_mean) / (s_std + 1e-8)
        # Note: If your actions are already [-1, 1], normalization might be identity
        # but we use the training constants for strict consistency.
        u_safe_norm = safeguard(s_norm, z_intent) 
        
        # Denormalize back to physical units for the environment check
        # Physical I = (norm_val + 1)/2 * 3000
        # Physical Fn = (norm_val + 1)/2 * 20
        i_phys = ((u_safe_norm[:, 0] + 1) / 2) * 3000.0
        fn_phys = ((u_safe_norm[:, 1] + 1) / 2) * 20.0

    # 5. Statistical Analysis of Violations
    violations = 0
    total_i_reduction = 0.0
    total_fn_reduction = 0.0

    for i in range(num_test_samples):
        # Manually inject state into env to test specific points
        env.state = states_danger[i].numpy()
        phys_action = np.array([u_safe_norm[i, 0].item(), u_safe_norm[i, 1].item()])
        
        _, _, _, info = env.step(phys_action)
        
        if not info['is_safe']:
            violations += 1
            
        # Track how much the safeguard "pushed back"
        # (Intent was 1.0, 1.0)
        total_i_reduction += (1.0 - u_safe_norm[i, 0].item())
        total_fn_reduction += (1.0 - u_safe_norm[i, 1].item())

    # 6. Results Summary
    fail_rate = (violations / num_test_samples) * 100
    print(f"\n[RESULTS]")
    print(f"Failure Rate: {fail_rate:.2f}% ({violations}/{num_test_samples} violations)")
    print(f"Avg Light Throttling: {total_i_reduction/num_test_samples:.4f} units")
    print(f"Avg Feed Throttling: {total_fn_reduction/num_test_samples:.4f} units")

    if fail_rate < 0.5:
        print("✅ MANIFOLD ALIGNED: The safeguard is robust at the boundaries.")
    else:
        print("⚠️ MISMATCH DETECTED: Consider increasing the 20/80 bias or lowering the LR.")

if __name__ == "__main__":
    check_stress_test_alignment()