import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from data_gen import get_fresh_batch_dataset
from torch.amp import autocast, GradScaler

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class ActionProjectionNetwork(nn.Module):
    """
    A Neural Network that projects unsafe actions back into the safe operational region.
    
    Architecture:
    It uses a Residual Network architecture. Instead of predicting the absolute safe action,
    it predicts a 'delta' (correction) and subtracts it from the nominal action. This
    encourages an identity mapping (doing nothing) when the nominal action is already safe.
    """
    def __init__(self, state_dim=4, action_dim=2, latent_dim=512):
        super(ActionProjectionNetwork, self).__init__()
        
        # Internal Static Normalization Constants [Biomass (cx), Nitrate (cN), Product (cq), Time (t_norm)]
        # This ensures the network always scales raw state inputs to [-1, 1] internally.
        self.register_buffer("max_vals", torch.tensor([6.0, 800.0, 0.2, 1.0]))
        
        # Input layer combines the 4D state space and 2D nominal action 
        self.input_layer = nn.Linear(state_dim + action_dim, latent_dim)
        
        # Residual Block with Internal LayerNorm for training stability
        self.res_block1 = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ELU(),  # ELU used to avoid dead neurons present in ReLU
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
        # Output layer maps back to the 2D action correction space
        self.final_layer = nn.Linear(latent_dim, action_dim)
        
        # Initialize final layer near zero. This ensures that at the start of training,
        # the predicted 'delta' is roughly 0, acting nearly as a perfect Identity function.
        nn.init.uniform_(self.final_layer.weight, -1e-2, 1e-2)
        nn.init.constant_(self.final_layer.bias, 0)
        self.elu = nn.ELU()

    def forward(self, state_phys, nom_act_norm, apply_override=True):
        """
        Forward pass to project nominal actions to safe target actions.
        
        Args:
            state_phys (Tensor): Raw physical states [batch, 4].
            nom_act_norm (Tensor): Normalized nominal (intended) actions [-1, 1].
            apply_override (bool): Whether to enforce hard analytical overrides (G3).
                                   (Typically False during training to let the NN learn it,
                                   and True during inference/RL for absolute guarantee).
                                   
        Returns:
            Tensor: Safe projected actions, normalized to [-1, 1].
        """
        # 1. Integrated Static Normalization
        # Converts physical state bounds to [-1, 1] based on cached maximums.
        state_norm = (state_phys / self.max_vals) * 2.0 - 1.0
        
        # 2. Initial Projection Layer
        # Concatenate normalized state and normalized nominal action
        x_in = torch.cat([state_norm, nom_act_norm], dim=1)
        x_proj = self.elu(self.input_layer(x_in))
        
        # 3. Skip Connection (Residual Connection)
        # The input is passed through the residual block and the original representation is added back.
        # This architecture is critical for the safety filter because the network must be biased
        # toward an "identity mapping" (doing nothing) if the action is already safe.
        # Deep networks without residual connections struggle to learn a perfect identity
        # function from scratch, but by adding x_proj back in here (and initializing the final 
        # linear layer's weights near zero), the network initially acts as a pass-through 
        # and only needs to learn the minor correction "delta" when the nominal action is unsafe.
        x_res = self.res_block1(x_proj)
        x_combined = self.elu(x_res + x_proj) # Global skip connection
        
        # 4. Final Residual Projection (u = z - delta)
        # ReLU is used here to ensure the action correction (delta) can only push 
        # the feed rate downward (reducing feed) or safely modify light intensity, 
        # aligning with the directional nature of the specific physical constraints.
        delta = torch.relu(self.final_layer(x_combined))
        u_nn = nom_act_norm - delta
        
        # 5. Explicit Analytical Override for G3 (Terminal Nitrate)
        # As a hard safety net, the neural network's intended response is explicitly bounded
        # to ensure the end-of-episode requirement is never violated.
        cN = state_phys[:, 1]
        t_norm = state_phys[:, 3]
        
        TOTAL_TIME = 240.0
        CONTROL_INTERVAL = 20.0
        N_LIMIT_TERM = 150.0
        FN_MAX = 40.0
        
        # Convert normalized time to absolute physical hours
        t_phys = t_norm * TOTAL_TIME
        time_remaining = TOTAL_TIME - t_phys
        
        # Flag if we are within the final control interval window
        near_end = time_remaining <= CONTROL_INTERVAL
        safe_time_remaining = torch.clamp(time_remaining, min=1.0) # Prevent division by zero
        
        # Denormalize the neural network's chosen nitrate feed rate [-1, 1] -> [0, 40]
        fn_nn_phys = ((u_nn[:, 1] + 1.0) / 2.0) * FN_MAX
        
        # Calculate maximum mathematically allowable feed rate to safely hit 150 mg/L * 0.98
        fn_max_term = (N_LIMIT_TERM * 0.98 - cN) / safe_time_remaining
        u_max_term = ((fn_max_term / FN_MAX) * 2.0) - 1.0
        
        # Ensure calculated maximum bound doesn't violently break the [-1, 1] output space
        u_max_term = torch.clamp(u_max_term, min=-1.0, max=1.0)
        
        u_safe = u_nn.clone()
        # Direct intervention: The feed action is capped at the hard limit if near the end.
        u_safe[:, 1] = torch.where(near_end, torch.minimum(u_nn[:, 1], u_max_term), u_nn[:, 1])
        
        return u_safe

def run_pretraining(epochs=50000, batch_size=65536, buffer_size=2000000, refresh_interval=100):
    """
    Main training loop for the Behavioral Cloning process.
    
    Generates data using `get_fresh_batch_dataset` and trains the
    ActionProjectionNetwork using an asymmetric loss function to penalize 
    safety violations more heavily than deviations from nominal actions.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActionProjectionNetwork(state_dim=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Decays learning rate when the loss plateaus to fine-tune the filter manifold
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=500, min_lr=1e-5
    )

    raw_history = []      
    unbiased_history = [] 
    early_stop_threshold = 2e-4
    required_success_per_buffer = 90
    buffer_success_count = 0
    
    # --- PLATEAU EXIT SETUP ---
    # Used to terminate training cleanly when the neural net stops improving.
    best_moving_avg = float('inf')
    plateau_counter = 0
    patience = 1000
    window_size = 200
    improvement_threshold = 0.001 # 0.1% improvement required to reset patience
    
    pbar = tqdm(range(epochs), desc="Training Safeguard Manifold")
    # Using Automatic Mixed Precision (AMP) to speed up large batch training
    scaler = GradScaler('cuda')

    for epoch in pbar:
        # Periodically regenerate the dataset to prevent overfitting to a single batch 
        # and provide continuous exposure to edge-case states.
        if epoch % refresh_interval == 0:
            buffer_success_count = 0
            # Fetch fresh batch combining uniform and boundary-focused sampling
            s_raw, a_norm, y_target = get_fresh_batch_dataset(buffer_size, bias=0.5)

        model.train()
        epoch_raw_loss = 0
        epoch_unbiased_loss = 0
        
        # Shuffle the loaded dataset
        indices = torch.randperm(buffer_size, device=device)
        
        # Mini-batch gradient descent loop
        for i in range(0, buffer_size, batch_size):
            batch_idx = indices[i : i + batch_size]
            b_s, b_a, b_y = s_raw[batch_idx], a_norm[batch_idx], y_target[batch_idx]

            optimizer.zero_grad()
            
            # Forward pass wrapped in AMP autocast
            with autocast(device_type='cuda'):
                # Note: Override is disabled during training to force the NN to learn the rules 
                # rather than relying exclusively on the analytical hard-cap.
                pred_y = model(b_s, b_a, apply_override=False)
                
                # Identify if the original nominal action was already safe.
                # If true, the NN should NOT alter the action (delta = 0).
                is_safe = torch.all(torch.abs(b_a - b_y) < 1e-7, dim=1, keepdim=True)

                # =====================================================================
                # Asymmetric Loss Function
                # =====================================================================
                
                # 1. Identity Penalty: Punish the network if it alters an action 
                # that was already perfectly safe. (Minimizes invasiveness)
                identity_weight = 10
                loss_identity = torch.mean((pred_y - b_a)**2)

                # 2. Safety Penalty: The network is punished heavily for failing to match
                # the analytically determined maximally safe target action.
                # A Relu clamp ensures 'overly safe' actions are not punished, only unsafe violations.
                safety_weight = 100
                violation = torch.clamp(pred_y - b_y, min=1e-6) 
                loss_safety = torch.mean(safety_weight * (violation**2))

                # Dynamically apply Identity loss to already-safe queries, and 
                # Safety loss to unsafe queries.
                total_loss = torch.where(is_safe, loss_identity, loss_safety).mean()
                
                # Track standard MSE purely for convergence evaluation metrics
                unbiased_mse = torch.mean((pred_y - b_y)**2)
                
            # Backward pass using the Gradient Scaler for AMP
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_raw_loss += total_loss.item()
            epoch_unbiased_loss += unbiased_mse.item()
            
        # Logging & Scheduling
        avg_unbiased = epoch_unbiased_loss / (buffer_size / batch_size)
        raw_history.append(epoch_raw_loss / (buffer_size / batch_size))
        unbiased_history.append(avg_unbiased)
        scheduler.step(raw_history[-1])

        # --- PLATEAU LOGIC ---
        if len(raw_history) >= window_size and epoch >= 7000:
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