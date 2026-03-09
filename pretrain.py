import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np # Added for moving average calculation
from tqdm import tqdm
from data_gen import get_fresh_batch_dataset
from torch.amp import autocast, GradScaler

# Hardware compatibility
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class ActionProjectionNetwork(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, latent_dim=512):
        super(ActionProjectionNetwork, self).__init__()
        
        # Internal Normalization Constants (Article Case Study Limits)
        # Order: [Biomass (cx), Nitrate (cN), Product (cq), Time (t_norm)]
        self.register_buffer("max_vals", torch.tensor([6.0, 800.0, 0.1, 1.0]))
        
        self.input_layer = nn.Linear(state_dim + action_dim, latent_dim)
        
        # Residual Block with Internal LayerNorm
        self.res_block1 = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
        self.final_layer = nn.Linear(latent_dim, action_dim)
        
        # Initialize final layer near zero to encourage identity mapping initially
        nn.init.uniform_(self.final_layer.weight, -1e-2, 1e-2)
        nn.init.constant_(self.final_layer.bias, 0)
        self.elu = nn.ELU()

    def forward(self, state_phys, nom_act_norm, apply_override=True):
        """
        Args:
            state_phys: Raw physical states [batch, 4]
            nom_act_norm: Normalized nominal actions [-1, 1]
            apply_override: If True, applies the explicit analytical G2 protection override.
        """
        # 1. Integrated Static Normalization
        state_norm = (state_phys / self.max_vals) * 2.0 - 1.0
        
        # 2. Initial Projection
        x_in = torch.cat([state_norm, nom_act_norm], dim=1)
        x_proj = self.elu(self.input_layer(x_in))
        
        # 3. Skip Connection (Residual Connection) for efficiency
        x_res = self.res_block1(x_proj)
        x_combined = self.elu(x_res + x_proj) # Global skip connection
        # 4. Final Residual Projection (u = z - delta)
        # ReLU ensures we only move the action in the safety-correction direction
        delta = torch.relu(self.final_layer(x_combined))
        u_nn = nom_act_norm - delta
        
        if not apply_override:
            return u_nn
        
        # 5. Explicit Analytical Override for G2 (Instantaneous)
        cx = state_phys[:, 0]
        cq = state_phys[:, 2]
        
        # 0.011 is the RATIO_LIMIT, 0.98 is the SAFE_BUFFER from data_gen
        g2_violation = cq > (cx * 0.011 * 0.98)
        
        u_safe = u_nn.clone()
        # Force Light Intent (idx 0) to maximum physical limit (+1.0 in normalized space)
        u_safe[:, 0] = torch.where(g2_violation, torch.full_like(u_safe[:, 0], 1.0), u_nn[:, 0])
        
        return u_safe

def run_pretraining(epochs=50000, batch_size=65536, buffer_size=2000000, refresh_interval=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActionProjectionNetwork(state_dim=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=500, min_lr=1e-5
    )

    raw_history = []      
    unbiased_history = [] 
    early_stop_threshold = 5e-6
    required_success_per_buffer = 90
    buffer_success_count = 0
    
    # --- PLATEAU EXIT SETUP ---
    best_moving_avg = float('inf')
    plateau_counter = 0
    patience = 1000
    window_size = 200
    improvement_threshold = 0.001 # 0.1% improvement
    
    pbar = tqdm(range(epochs), desc="Training Safeguard Manifold")
    scaler = GradScaler('cuda')

    for epoch in pbar:
        if epoch % refresh_interval == 0:
            buffer_success_count = 0
            # Fetch fresh batch with 10h budget logic
            s_raw, a_norm, y_target = get_fresh_batch_dataset(buffer_size, bias=0.5)

        model.train()
        epoch_raw_loss = 0
        epoch_unbiased_loss = 0
        indices = torch.randperm(buffer_size, device=device)
        
        for i in range(0, buffer_size, batch_size):
            batch_idx = indices[i : i + batch_size]
            b_s, b_a, b_y = s_raw[batch_idx], a_norm[batch_idx], y_target[batch_idx]

            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                pred_y = model(b_s, b_a, apply_override=False)
                
                # Check if the original intent was already safe
                is_safe = torch.all(torch.abs(b_a - b_y) < 1e-7, dim=1, keepdim=True)

                # 1. Identity Penalty: Punished for deviating if already safe
                identity_weight = 10
                loss_identity = torch.mean((pred_y - b_a)**2)

                # 2. Safety Penalty: Punished for violating boundaries (g1, g2)
                safety_weight = 100
                violation = torch.clamp(pred_y - b_y, min=1e-6) 
                loss_safety = torch.mean(safety_weight * (violation**2))

                total_loss = torch.where(is_safe, loss_identity, loss_safety).mean()
                unbiased_mse = torch.mean((pred_y - b_y)**2)
                
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_raw_loss += total_loss.item()
            epoch_unbiased_loss += unbiased_mse.item()
            
        avg_unbiased = epoch_unbiased_loss / (buffer_size / batch_size)
        raw_history.append(epoch_raw_loss / (buffer_size / batch_size))
        unbiased_history.append(avg_unbiased)
        scheduler.step(raw_history[-1])

        # --- PLATEAU LOGIC ---
        if len(raw_history) >= window_size and epoch >= 10000:
            current_moving_avg = np.mean(raw_history[-window_size:])
            
            # Check for at least 0.1% improvement over the best-ever moving average
            if current_moving_avg < best_moving_avg * (1 - improvement_threshold):
                best_moving_avg = current_moving_avg
                plateau_counter = 0
            else:
                plateau_counter += 1

        # Check for convergence success (low MSE)
        if epoch >= 5000 and avg_unbiased < early_stop_threshold:
            buffer_success_count += 1

        pbar.set_postfix({
            'MSE': f'{avg_unbiased:.2e}', 
            'Total Loss': f'{raw_history[-1]:.2e}',
            'Patience': f'{plateau_counter}/{patience}',
            'Stable': f"{buffer_success_count}/{required_success_per_buffer}"
        })

        # --- EARLY EXIT CONDITIONS ---
        if buffer_success_count >= required_success_per_buffer:
            print(f"\n[Success] Safety Manifold Converged (MSE threshold) at epoch {epoch}")
            break
        
        if epoch >= 5000 and plateau_counter >= patience:
            print(f"\n[Plateau] No 0.1% improvement for {patience} epochs. Terminating.")
            break

    torch.save(model.state_dict(), "action_projection_network.pth")
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()  # Create secondary axis
    
    if len(raw_history) >= window_size:
        # Calculate moving averages
        moving_avg_raw = np.convolve(raw_history, np.ones(window_size)/window_size, mode='valid')
        moving_avg_mse = np.convolve(unbiased_history, np.ones(window_size)/window_size, mode='valid')
        epochs_range = range(window_size - 1, len(raw_history))
        
        # Plot Total Loss on left axis (Blue)
        line1 = ax1.plot(epochs_range, moving_avg_raw, color='blue', label=f'Total Loss ({window_size}-Ep MA)')
        # Plot MSE on right axis (Red)
        line2 = ax2.plot(epochs_range, moving_avg_mse, color='red', label=f'Unbiased MSE ({window_size}-Ep MA)')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right')
    else:
        # Fallback if training ended extremely early
        line1 = ax1.plot(raw_history, color='blue', label='Total Weighted Loss')
        line2 = ax2.plot(unbiased_history, color='red', label='Unbiased MSE')
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right')

    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Total Loss", color='blue')
    ax2.set_ylabel("Unbiased MSE", color='red')
    
    # Use log scale for both axes as loss spans multiple orders of magnitude
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    
    plt.title("Safeguard Convergence: 200-Epoch Moving Averages")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("manifold_convergence.png")
    plt.close() # Close figure instead of show() to avoid blocking if run unsupervised

if __name__ == "__main__":
    run_pretraining()