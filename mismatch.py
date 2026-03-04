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

    print("\n--- Diagnostic 1: Light Intensity ($I$) Boundary ---")
    state = env.reset() # Standard starting point
    state_t = torch.FloatTensor(state).unsqueeze(0)
    z_intent = torch.tensor([[1.0, 0.0]]) # Intent: Max Light, No Feed
    
    with torch.no_grad():
        s_norm = (state_t - s_mean) / (s_std + 1e-8)
        z_norm = (z_intent - a_mean) / (a_std + 1e-8)
        u_safe_norm = safeguard(s_norm, z_norm)
        u_safe = (u_safe_norm * (a_std + 1e-8)) + a_mean
        u_phys = u_safe.numpy().flatten()

    _, _, _, info = env.step(u_phys)
    print(f"Current Biomass: {info['biomass']:.3f}")
    print(f"Env I_limit: {450.0 / np.exp(-0.25 * info['biomass'] * 0.5):.2f}")
    print(f"Safeguard I: {(u_phys[0]+1)/2 * 3000:.2f}")
    print(f"Result: {'✅ SAFE' if info['is_safe'] else '❌ UNSAFE'}")

    print("\n--- Diagnostic 2: Nitrate ($c_N$) Boundary ---")
    # Manually set state to be dangerously close to the 180.0 limit
    env.state = np.array([0.5, 179.0, 5.0]) 
    state_danger = env.get_state_norm()
    state_t_danger = torch.FloatTensor(state_danger).unsqueeze(0)
    
    # Intent: Maximum Nitrate Feed (z_fn = 1.0)
    z_intent_fn = torch.tensor([[0.0, 1.0]]) 
    
    with torch.no_grad():
        s_norm = (state_t_danger - s_mean) / (s_std + 1e-8)
        z_norm = (z_intent_fn - a_mean) / (a_std + 1e-8)
        u_safe_norm = safeguard(s_norm, z_norm)
        u_safe = (u_safe_norm * (a_std + 1e-8)) + a_mean
        u_phys = u_safe.numpy().flatten()

    _, _, _, info = env.step(u_phys)
    print(f"Current Nitrate: {info['nitrate']:.2f} (Limit: 180.0)")
    print(f"Safeguard Feed (Fn): {(u_phys[1]+1)/2 * 20.0:.4f}")
    
    if info['is_safe']:
        print("✅ ALIGNED: Safeguard correctly throttled Feed near the limit.")
    else:
        print("❌ MISMATCHED: Safeguard allowed Feed that caused a Nitrate violation.")
        print(f"   Safety Penalty: {info['penalties']['safety']:.4f}")

if __name__ == "__main__":
    check_manifold_alignment()