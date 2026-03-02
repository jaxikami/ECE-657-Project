import os
# Fix for OpenMP runtime conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from data_gen import get_fresh_batch_dataset

# 1. High-Capacity Smart Steering Wheel with LayerNorm
class SmartSteeringWheel(nn.Module):
    def __init__(self, state_dim=3, action_dim=2, latent_dim=1024):
        super(SmartSteeringWheel, self).__init__()
        # LayerNorm stabilizes the deep mapping of the photoproduction manifold
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.LeakyReLU(0.2),
            
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.LeakyReLU(0.2),
            
            nn.Linear(latent_dim, latent_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim // 2, action_dim) 
        )

    def forward(self, state_norm, nom_act_norm):
        """Residual Bridge: Safe = Nominal + Delta"""
        x = torch.cat([state_norm, nom_act_norm], dim=1)
        delta_norm = self.net(x)
        return nom_act_norm + delta_norm

def run_pretraining(epochs=3000, batch_size=512, samples_per_epoch=20000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmartSteeringWheel().to(device)
    
    # Pre-calculate normalization constants from a large sample
    print("📊 Calculating manifold normalization constants...")
    s_raw, a_raw, _ = get_fresh_batch_dataset(100000)
    s_mean, s_std = s_raw.mean(0), s_raw.std(0)
    a_mean, a_std = a_raw.mean(0), a_raw.std(0)
    
    np.savez("norm_constants.npz", 
             s_mean=s_mean.numpy(), s_std=s_std.numpy(), 
             a_mean=a_mean.numpy(), a_std=a_std.numpy())

    s_m, s_s = s_mean.to(device), s_std.to(device)
    a_m, a_s = a_mean.to(device), a_std.to(device)

    # Optimization Setup: AdamW and HuberLoss for robustness
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=30)
    criterion = nn.HuberLoss(delta=0.1) 

    history = []
    print(f"🚀 Refinement Training on {device}...")
    pbar = tqdm(range(epochs), desc="Refining Safety Manifold")
    
    for epoch in pbar:
        # 1. Fresh Data Generation
        s_data, a_data, y_data = get_fresh_batch_dataset(samples_per_epoch)
        
        # 2. Importance Sampling: Duplicate 'Hard' cases (large safety corrections)
        delta_physical = torch.abs(y_data - a_data).sum(dim=1)
        hard_indices = (delta_physical > 10.0).nonzero(as_tuple=True)[0]
        
        if len(hard_indices) > 0:
            s_hard, a_hard, y_hard = s_data[hard_indices], a_data[hard_indices], y_data[hard_indices]
            s_data = torch.cat([s_data, s_hard], dim=0)
            a_data = torch.cat([a_data, a_hard], dim=0)
            y_data = torch.cat([y_data, y_hard], dim=0)

        # 3. Normalization
        s_norm = (s_data.to(device) - s_m) / (s_s + 1e-8)
        a_norm = (a_data.to(device) - a_m) / (a_s + 1e-8)
        y_norm = (y_data.to(device) - a_m) / (a_s + 1e-8) 
        
        loader = DataLoader(TensorDataset(s_norm, a_norm, y_norm), batch_size=batch_size, shuffle=True)
        
        model.train()
        epoch_loss = 0
        for b_s, b_a, b_y in loader:
            optimizer.zero_grad()
            pred_y_norm = model(b_s, b_a)
            loss = criterion(pred_y_norm, b_y)
            loss.backward()
            
            # Strict gradient clipping for the final refinement stage
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(loader)
        history.append(avg_loss)
        scheduler.step(avg_loss)

        # Update tqdm status with Loss and Learning Rate
        pbar.set_postfix({'Loss': f'{avg_loss:.2e}', 'LR': f"{optimizer.param_groups[0]['lr']:.2e}"})

    # Save final weights
    torch.save(model.state_dict(), "smart_steering_wheel.pth")
    print("\n✅ Pretraining Finalized. Generating Plots...")

    # --- FINAL PLOTTING ---
    plt.style.use('seaborn-v0_8-muted')
    plt.figure(figsize=(10, 6))
    
    # Main Loss Curve
    plt.plot(history, color='#2c3e50', linewidth=1.5, label='Huber Loss (Normalized)')
    
    # Formatting
    plt.yscale('log')
    plt.title("ResNet Convergence: Breaking the Safety Manifold Plateau")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (Log Scale)")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    
    # Annotations
    final_loss = history[-1]
    plt.annotate(f'Final Loss: {final_loss:.2e}', 
                 xy=(len(history)-1, final_loss), 
                 xytext=(len(history)*0.7, final_loss*10),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

    plt.legend()
    plt.tight_layout()
    plt.savefig("pretrain_loss_curve.png", dpi=300)
    print("📈 Plot saved as 'pretrain_loss_curve.png'")
    plt.show()

if __name__ == "__main__":
    run_pretraining()