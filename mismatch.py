import torch
import numpy as np
from env import PhotoProductionEnv
from res_net_agent import ActionProjectionNetwork

def check_manifold_alignment():
    # 1. Initialize Environment and Load Safeguard
    env = PhotoProductionEnv(train_mode=False)
    # Ensure latent_dim matches your fixed architecture (512)
    safeguard = ActionProjectionNetwork(state_dim=3, action_dim=2, latent_dim=512)
    
    try:
        safeguard.load_state_dict(torch.load("action_projection_network.pth"))
        safeguard.eval()
        print("✅ Safeguard weights loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return

    # 2. Load Normalization Constants
    norms = np.load("norm_constants.npz")
    s_mean, s_std = torch.tensor(norms['s_mean']), torch.tensor(norms['s_std'])
    a_mean, a_std = torch.tensor(norms['a_mean']), torch.tensor(norms['a_std'])

    # 3. Test a "Danger" Scenario
    # State: Low biomass (highly sensitive to light)
    state = env.reset() 
    state_t = torch.FloatTensor(state).unsqueeze(0)
    
    # Intent: Maximum Light Intensity (z = 1.0)
    z_intent = torch.tensor([[1.0, 0.0]]) 
    
    # 4. Project using Safeguard
    with torch.no_grad():
        s_norm = (state_t - s_mean) / (s_std + 1e-8)
        z_norm = (z_intent - a_mean) / (a_std + 1e-8)
        u_safe_norm = safeguard(s_norm, z_norm)
        u_safe = (u_safe_norm * (a_std + 1e-8)) + a_mean
        u_phys = u_safe.numpy().flatten()

    # 5. Verify in Environment
    _, _, _, info = env.step(u_phys)
    
    print("\n--- Diagnostic Results ---")
    print(f"Initial Biomass: {info['biomass']:.3f}")
    print(f"Environment I_limit: {450.0 / np.exp(-0.25 * info['biomass'] * 0.5):.2f}")
    print(f"Safeguard Projected I: {(u_phys[0]+1)/2 * 3000:.2f}")
    
    if info['is_safe']:
        print("✅ ALIGNED: The safeguard's projection is accepted by the environment.")
    else:
        print("❌ MISMATCHED: The environment rejected the safeguard's 'safe' action.")
        print(f"   Safety Violation Breakdown: {info['penalties']['safety']}")

if __name__ == "__main__":
    check_manifold_alignment()