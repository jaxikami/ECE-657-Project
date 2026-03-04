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
from torch.amp import autocast, GradScaler
class ActionProjectionNetwork(nn.Module):
    def __init__(self, state_dim=3, action_dim=2, latent_dim=512):
        super(ActionProjectionNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ELU(),
            
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ELU(),
            
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ELU(),
            nn.Linear(latent_dim // 2, action_dim) 
        )

    def forward(self, state_norm, nom_act_norm):
        x = torch.cat([state_norm, nom_act_norm], dim=1)
        delta_norm = self.net(x)
        return nom_act_norm + delta_norm

def run_pretraining(epochs=40000, batch_size=65536, buffer_size=2000000, refresh_interval=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActionProjectionNetwork().to(device)
    a_mean = torch.tensor([0.0, 0.0], device=device)
    a_std = torch.tensor([1.0, 1.0], device=device)
    # Hyperparameters with lr decay
    initial_lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, 
    T_0=refresh_interval, 
    T_mult=1, 
    eta_min=1e-5
)
    # decay_end_epoch = epochs
    criterion = nn.SmoothL1Loss(beta=0.05)
    history = []
    lr_history = []
    ma_window = 500  # The window for the moving average
    check_duration = 100     # How many epochs to monitor for lack of change
    min_relative_change = 0.001  # 0.1% threshold
    stagnation_counter = 0
    best_ma_value = None
    relative_change = None
    # Early Stopping Logic
# 1. LATENCY PERIOD (start_monitoring_epoch): 
#    We wait 3,000 epochs before checking for success. This prevents the 
#    model from "tripping" the exit early while the loss is still volatile 
#    during the initial high-learning-rate phase.
#
# 2. PRECISION TARGET (early_stop_threshold): 
#    The model must achieve an Average SmoothL1 Loss of 1e-4 or lower. 
#    This defines the mathematical "closeness" required for the projected 
#    action to be considered safe.
#
# 3. CONSISTENCY REQUIREMENT (required_success_per_buffer): 
#    Achieving the threshold once isn't enough. The model must maintain 
#    that precision for 40 epochs within a single data refresh cycle. 
#    This proves the model isn't just "passing through" a lucky local 
#    minimum but has actually converged.
#
# 4. BUFFER RESET (buffer_success_count): 
#    Because get_fresh_batch_dataset() generates new data every 100 epochs (refresh_interval), we reset the success count to zero. 
#    The model must "prove its mastery" all over again on the new data 
#    to ensure it hasn't overfit to the previous buffer.

    early_stop_threshold = 5e-4
    required_success_per_buffer = 40
    start_monitoring_epoch = 3000
    buffer_success_count = 0
    
    pbar = tqdm(range(epochs), desc="Training Safety Manifold")
    scaler = GradScaler('cuda')
    for epoch in pbar:
        buffer_age = epoch % refresh_interval
        current_bias = 0.5 + (0.4 * (epoch / epochs))
        noise_scale = max(5e-3, 1e-2 * (1.0 - (epoch / start_monitoring_epoch)))
        
        # This block executes every 'refresh_interval' (e.g., 100 epochs) to prevent 
        # the model from over-specializing on a single static dataset.
        #
        # 1. RESET SUCCESS COUNT: 
        #    We zero out 'buffer_success_count'. The model must prove its 
        #    accuracy from scratch on the new data to ensure it hasn't overfit 
        #    to the previous samples.
        #
        # 2. WHY WE ADD NOISE (Gaussian Blurring): 
        #    We inject random noise scaled by 'noise_scale' into the raw states 
        #    and actions. This serves three vital purposes:
        #    - ROBUSTNESS: In the real world, sensors are noisy. Training on 
        #      "blurred" data forces the network to learn the general shape of 
        #      the safety manifold rather than just memorizing exact points.
        #    - MANIFOLD "SHARPENING": Early in training, high noise helps the 
        #      network find the broad "safe zone." As noise_scale decays over 
        #      epochs, the model "sharpens" its focus, eventually learning the 
        #      precise boundary of the manifold.
        #    - COVERAGE: It ensures that the model sees data slightly outside 
        #      the standard distribution, helping it handle "out-of-bounds" 
        #      scenarios during real-time safety projection.
        #
        # 3. ON-THE-FLY NORMALIZATION: 
        #    Raw data is immediately shifted and scaled using the global 
        #    constants (s_m, s_s, etc.). This keeps all inputs within 
        #    a range (typically -1 to 1) that the neural network can process 
        #    without numerical instability.
        #
        # 4. LEARNING RATE BUMP: 
        #    When new data is introduced (if epoch > 0), we slightly reset 
        #    the LR to 80% of its initial value. This "jolt" provides 
        #    enough momentum for the optimizer to adapt to the new data 
        #    distribution before the standard decay logic takes back over.
        if buffer_age == 0:
            buffer_success_count = 0
            s_raw, a_raw, y_target = get_fresh_batch_dataset(buffer_size, bias=current_bias)
            
            # NEW: Update the "ruler" to match the current bias/distribution
            # This ensures the 'Danger Zone' is always spread across the network's input range
            s_m_local = s_raw.mean(dim=0)
            s_s_local = torch.clamp(s_raw.std(dim=0), min=1e-8)
            
            # Apply noise to the raw data BEFORE normalizing
            s_raw = s_raw + torch.randn_like(s_raw) * (s_s_local * noise_scale)
            
            # Normalize using the stats of the CURRENT buffer
            s_norm = (s_raw - s_m_local) / (s_s_local + 1e-8)
            a_norm = a_raw # Actions are already -1 to 1
            y_norm = y_target
            
            
 
            
            # if epoch > 0:
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = initial_lr * 0.8

        # LR Decay
        # if buffer_age != 0:
        #     lr_coeff = max(0.0, 1.0 - (epoch / decay_end_epoch))
        #     current_lr = max(1e-5, initial_lr * lr_coeff)
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = current_lr
        # else:
        #     current_lr = optimizer.param_groups[0]['lr']

        # Training Execution
        model.train()
        epoch_loss = 0
        indices = torch.randperm(buffer_size, device=device)
        
        # 2. Iterate through the buffer using direct VRAM slicing 
        for i in range(0, buffer_size, batch_size):
            # Grab a slice of indices
            batch_idx = indices[i : i + batch_size]
            
            # Direct slicing: No memory copying, just pointer offsets in VRAM
            b_s = s_norm[batch_idx]
            b_a = a_norm[batch_idx]
            b_y = y_norm[batch_idx]

            optimizer.zero_grad()
            
            # Forward pass with mixed precision 
            with autocast(device_type='cuda'):
                pred_y = model(b_s, b_a)
                is_unsafe = torch.any(torch.abs(b_a - b_y) > 1e-4, dim=1).float()
                raw_loss = criterion(pred_y, b_y)
                weights = 1.0 + (is_unsafe * 4.0) 
                loss = (raw_loss * weights).mean()

            # High-throughput backprop
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / (buffer_size / batch_size)
        history.append(avg_loss)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        lr_history.append(current_lr)

        # --- Per-Buffer Early Exit Logic ---
        if epoch >= start_monitoring_epoch:
            if avg_loss < early_stop_threshold:
                buffer_success_count += 1
            if len(history) >= ma_window:
                current_ma = np.mean(history[-ma_window:])
                if best_ma_value is not None:
                    relative_change = abs(current_ma - best_ma_value) / (best_ma_value + 1e-8)
                else:
                    best_ma_value = current_ma  # Initialize on the first calculation
                    if relative_change < min_relative_change:
                        stagnation_counter += 1
                    else:
                        stagnation_counter = 0
                        best_ma_value = current_ma
        if stagnation_counter >= check_duration:
                print(f"\n⚠️ WARNING: Training concluded early due to STAGNATION.")
                print(f"Loss MA changed by less than {min_relative_change*100}% for {check_duration} epochs.")
                
                # Still perform the Option B save so we don't lose progress
                np.savez("norm_constants.npz", 
                         s_mean=s_m_local.cpu().numpy(), 
                         s_std=s_s_local.cpu().numpy(), 
                         a_mean=a_mean.cpu().numpy(), 
                         a_std=a_std.cpu().numpy())
                break        

        pbar.set_postfix({
            'Loss': f'{avg_loss:.2e}', 
            'BufSuccess': f"{buffer_success_count}/{required_success_per_buffer}",
            'Age': buffer_age
        })

        if buffer_success_count >= required_success_per_buffer:
            print(f"\n[Success] {buffer_success_count} epochs below {early_stop_threshold}.")
            
            # --- CRITICAL FIX FOR OPTION B ---
            # Save the FINAL local stats that the model actually learned
            print(f"Saving FINAL manifold normalization constants (Bias: {current_bias:.2f})...")
            np.savez("norm_constants.npz", 
                     s_mean=s_m_local.cpu().numpy(), 
                     s_std=s_s_local.cpu().numpy(), 
                     a_mean=a_mean.cpu().numpy(), 
                     a_std=a_std.cpu().numpy())
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