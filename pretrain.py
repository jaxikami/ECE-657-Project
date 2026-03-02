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

class ActionProjectionNetwork(nn.Module):
    """
    Neural safety layer for projecting nominal actions onto 
    the constrained manifold of the photoproduction process.
    """
    def __init__(self, state_dim=3, action_dim=2, latent_dim=1024):
        super(ActionProjectionNetwork, self).__init__()
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
        x = torch.cat([state_norm, nom_act_norm], dim=1)
        delta_norm = self.net(x)
        return nom_act_norm + delta_norm

def run_pretraining(epochs=5000, batch_size=1024, buffer_size=200000, refresh_interval=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActionProjectionNetwork().to(device)
    
    # 1. Normalization Setup
    print("Initializing manifold normalization constants...")
    s_init, a_init, _ = get_fresh_batch_dataset(500000)
    s_mean, s_std = s_init.mean(0), s_init.std(0)
    a_mean, a_std = a_init.mean(0), a_init.std(0)
    
    np.savez("norm_constants.npz", 
             s_mean=s_mean.numpy(), s_std=s_std.numpy(), 
             a_mean=a_mean.numpy(), a_std=a_std.numpy())

    s_m, s_s = s_mean.to(device), s_std.to(device)
    a_m, a_s = a_mean.to(device), a_std.to(device)

    # 2. Optimization Config
    initial_lr = 5e-4
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    decay_end_epoch = (2 / 3) * epochs
    criterion = nn.SmoothL1Loss(beta=0.1)

    history = []
    lr_history = []
    
    # --- Per-Buffer Success Config ---
    early_stop_threshold = 5e-4
    required_success_per_buffer = 90
    start_monitoring_epoch = 1500
    buffer_success_count = 0 # This resets every refresh
    
    # 3. Training Loop
    pbar = tqdm(range(epochs), desc="Training Safety Manifold")
    
    for epoch in pbar:
        buffer_age = epoch % refresh_interval
        
        # PERIODIC REFRESH
        noise_scale = max(1e-3, 1e-2 * (1.0 - (epoch / start_monitoring_epoch)))
        
        if buffer_age == 0:
            # RESET SUCCESS COUNT FOR THE NEW BUFFER
            buffer_success_count = 0
            
            s_raw, a_raw, y_target = get_fresh_batch_dataset(buffer_size)
            
            # --- Blurring with Noise Decay ---
            # As noise_scale approaches 0, the manifold becomes "sharper" for the agent.
            if noise_scale > 0:
                s_raw = s_raw + torch.randn_like(s_raw) * (s_raw.std(0) * noise_scale)
                a_raw = a_raw + torch.randn_like(a_raw) * (a_raw.std(0) * noise_scale)
            
            s_norm = (s_raw.to(device) - s_m) / (s_s + 1e-8)
            a_norm = (a_raw.to(device) - a_m) / (a_s + 1e-8)
            y_norm = (y_target.to(device) - a_m) / (a_s + 1e-8)
            
            train_ds = TensorDataset(s_norm, a_norm, y_norm)
            loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            
            if epoch > 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = initial_lr * 0.8

        # LR Decay logic
        if buffer_age != 0:
            lr_coeff = max(0.0, 1.0 - (epoch / decay_end_epoch))
            current_lr = max(1e-7, initial_lr * lr_coeff)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        else:
            current_lr = optimizer.param_groups[0]['lr']

        # Training Execution
        model.train()
        epoch_loss = 0
        for b_s, b_a, b_y in loader:
            optimizer.zero_grad()
            pred_y = model(b_s, b_a)
            loss = criterion(pred_y, b_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(loader)
        history.append(avg_loss)
        lr_history.append(current_lr)

        # --- Per-Buffer Early Exit Logic ---
        if epoch >= start_monitoring_epoch:
            if avg_loss < early_stop_threshold:
                buffer_success_count += 1

        pbar.set_postfix({
            'Loss': f'{avg_loss:.2e}', 
            'BufSuccess': f"{buffer_success_count}/{required_success_per_buffer}",
            'Age': buffer_age
        })

        if buffer_success_count >= required_success_per_buffer:
            print(f"\n[Success] Mastery! {buffer_success_count} epochs below {early_stop_threshold} in current buffer.")
            break

    # Save weights
    torch.save(model.state_dict(), "action_projection_network.pth")
    
    # --- Plotting ---
    plt.style.use('seaborn-v0_8-muted')
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Calculate Moving Average (200 episodes)
    window = 200
    if len(history) >= window:
        moving_avg = np.convolve(history, np.ones(window)/window, mode='valid')
        # Pad with NaNs so the moving average line aligns with the history index
        moving_avg_padded = np.concatenate([np.full(window-1, np.nan), moving_avg])
    else:
        moving_avg_padded = np.full(len(history), np.nan)

    # 1. Raw Training Loss
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('SmoothL1 Loss')
    ax1.plot(history, color='#2c3e50', alpha=0.3, linewidth=1, label='Raw Epoch Loss')
    
    # 2. Moving Average
    ax1.plot(moving_avg_padded, color='#e74c3c', linewidth=2, label=f'{window}-Epoch Moving Average')
    
    # 3. Success Threshold
    ax1.axhline(y=early_stop_threshold, color='green', linestyle='--', alpha=0.6, label=f'Target ({early_stop_threshold:.1e})')

    # Styling
    ax1.set_yscale('log')
    ax1.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.title(f"Loss vs Epochs")
    plt.legend(loc='upper right')
    
    fig.tight_layout()
    plt.savefig("manifold_convergence.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    run_pretraining()