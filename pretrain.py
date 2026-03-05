import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from data_gen import get_fresh_batch_dataset
from torch.amp import autocast, GradScaler

class ActionProjectionNetwork(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, latent_dim=512):
        super(ActionProjectionNetwork, self).__init__()
        self.input_layer = nn.Linear(state_dim + action_dim, latent_dim)
        
        self.res_block1 = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        self.final_layer = nn.Linear(latent_dim, action_dim)
        
        nn.init.uniform_(self.final_layer.weight, -1e-4, 1e-4)
        nn.init.constant_(self.final_layer.bias, 0)
        self.elu = nn.ELU()

    def forward(self, state_norm, nom_act_norm):
        x_in = torch.cat([state_norm, nom_act_norm], dim=1)
        x = self.elu(self.input_layer(x_in))
        
        identity = x
        x = self.res_block1(x)
        x = self.elu(x + identity) 
        
        # u = z - delta
        return nom_act_norm - torch.relu(self.final_layer(x))

def static_normalize(states):
    # Fixed physical limits to prevent vanishing features and coordinate shocks
    max_vals = torch.tensor([6.0, 170.0, 25.0, 1.0], device=states.device)
    return (states / max_vals) * 2.0 - 1.0

def run_pretraining(epochs=30000, batch_size=65536, buffer_size=2000000, refresh_interval=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActionProjectionNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=refresh_interval, T_mult=1, eta_min=1e-5
    )

    # Logging histories
    raw_history = []      # Total Loss (Identity + Weighted Safety)
    unbiased_history = [] # Pure MSE (Closeness to Target)
    
    early_stop_threshold = 1e-5
    required_success_per_buffer = 80
    start_monitoring_epoch = 4000
    buffer_success_count = 0
    
    pbar = tqdm(range(epochs), desc="Training Safety Manifold")
    scaler = GradScaler('cuda')

    for epoch in pbar:
        buffer_age = epoch % refresh_interval
        current_bias = 0.5 + (0.4 * (epoch / epochs)) 
        
        if buffer_age == 0:
            buffer_success_count = 0
            s_raw, a_raw, y_target = get_fresh_batch_dataset(buffer_size, bias=current_bias)
            s_norm = static_normalize(s_raw)
            a_norm = a_raw 
            y_norm = y_target

        model.train()
        epoch_raw_loss = 0
        epoch_unbiased_loss = 0
        indices = torch.randperm(buffer_size, device=device)
        
        for i in range(0, buffer_size, batch_size):
            batch_idx = indices[i : i + batch_size]
            b_s, b_a, b_y = s_norm[batch_idx], a_norm[batch_idx], y_norm[batch_idx]

            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                pred_y = model(b_s, b_a)
                is_safe = torch.all(torch.abs(b_a - b_y) < 1e-7, dim=1, keepdim=True)

                # 1. Identity Loss (u should equal z if already safe)
                identity_weight = 5.0 - (epoch / epochs)
                loss_identity = identity_weight * torch.mean((pred_y - b_a)**2)

                # 2. Safety Loss (High pressure on violations)
                safety_weight = 5.0 + (15.0 * (epoch / epochs))
                violation = torch.clamp(pred_y - b_y, min=0.0)
                loss_safety = torch.mean(safety_weight * (violation**2))

                # 3. Combined Raw Loss
                
                total_loss = torch.where(is_safe, loss_identity, loss_safety).mean()
                
                # 4. Unbiased MSE for convergence monitoring
                unbiased_mse = torch.mean((pred_y - b_y)**2)
                
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_raw_loss += total_loss.item()
            epoch_unbiased_loss += unbiased_mse.item()
            
        avg_raw = epoch_raw_loss / (buffer_size / batch_size)
        avg_unbiased = epoch_unbiased_loss / (buffer_size / batch_size)
        
        raw_history.append(avg_raw)
        unbiased_history.append(avg_unbiased)
        scheduler.step()

        if epoch >= start_monitoring_epoch and avg_unbiased < early_stop_threshold:
            buffer_success_count += 1

        pbar.set_postfix({
            'Raw': f'{avg_raw:.2e}', 
            'MSE': f'{avg_unbiased:.2e}', 
            'Success': f"{buffer_success_count}/{required_success_per_buffer}"
        })

        if buffer_success_count >= required_success_per_buffer:
            print(f"\n[Success] Converged at epoch {epoch}")
            break

    torch.save(model.state_dict(), "action_projection_network.pth")
    
    # Plotting both losses
    plt.figure(figsize=(10, 6))
    plt.plot(raw_history, label='Total Raw Loss (Weighted)', alpha=0.4)
    plt.yscale('log')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Manifold Convergence: Raw vs Unbiased Loss")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.savefig("manifold_convergence.png")
    plt.show()

if __name__ == "__main__":
    run_pretraining()