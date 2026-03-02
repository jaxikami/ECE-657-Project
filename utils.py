import os
# 1. FIX OPENMP RUNTIME ERROR: Prevents the "Error #15" initialization crash 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch

class Logger:
    """
    Handles real-time tracking of rewards and constraint violations during 
    the 50,000 episode training run.
    """
    def __init__(self, experiment_name):
        self.name = experiment_name
        self.rewards = []
        self.violations = []
        self.avg_rewards = []
        self.violation_rates = []
        
    def log_episode(self, reward, violated):
        """Logs the total reward and safety status of a completed episode."""
        self.rewards.append(reward)
        self.violations.append(1 if violated else 0)
        
        # Calculate 100-episode moving averages for smoother training curves
        window = 100
        if len(self.rewards) >= window:
            self.avg_rewards.append(np.mean(self.rewards[-window:]))
            self.violation_rates.append(np.mean(self.violations[-window:]) * 100)
        else:
            # Fallback for the first 99 episodes
            self.avg_rewards.append(np.mean(self.rewards))
            self.violation_rates.append(np.mean(self.violations) * 100)

    def print_status(self, episode, total):
        """Prints current performance metrics to the console."""
        print(f"[{self.name}] Ep {episode}/{total} | "
              f"Avg Rew: {self.avg_rewards[-1]:.2f} | "
              f"Viol Rate: {self.violation_rates[-1]:.2f}%")

    def save_logs(self):
        """Exports the training history to a CSV for external analysis."""
        df = pd.DataFrame({
            'reward': self.rewards,
            'avg_reward': self.avg_rewards,
            'violation_rate': self.violation_rates
        })
        df.to_csv(f"{self.name}_logs.csv", index=False)

def plot_training_results(baseline_logger, resnet_logger):
    """
    Generates high-resolution comparison plots between the Baseline PPO 
    and the ResNet-Guided PPO.
    """
    plt.style.use('seaborn-v0_8-muted')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # --- Plot 1: Reward Convergence ---
    ax1.plot(baseline_logger.avg_rewards, label='Baseline (Lagrangian Penalty)', color='tab:red', alpha=0.8)
    ax1.plot(resnet_logger.avg_rewards, label='Proposed (ResNet Projection)', color='tab:blue', linewidth=2)
    ax1.set_title("Training Convergence: Cumulative Reward")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("100-Ep Moving Average Reward")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Violation Rates ---
    ax2.plot(baseline_logger.violation_rates, label='Baseline Violations', color='tab:red', linestyle='--')
    ax2.plot(resnet_logger.violation_rates, label='ResNet Violations', color='tab:blue', linewidth=2)
    ax2.set_title("Safety Performance: Constraint Violation Rate")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Violation Frequency (%)")
    ax2.set_ylim(-5, 105) # Keeps the y-axis standard for percentage comparison
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("research_results_comparison.png", dpi=300)
    print("📈 Comparison plots saved as 'research_results_comparison.png'")
    plt.show()

def plot_manifold_slice(resnet, s_mean, s_std, a_mean, a_std):
    """
    Optional visualization of the learned safety manifold. 
    Shows how Light Intensity is capped as Biomass (cx) increases.
    """
    # Create a grid of cx (Biomass) and i_nom (Intended Light)
    cx_grid = np.linspace(0.1, 5.0, 50)
    i_nom_grid = np.linspace(0, 3000, 50)
    CX, INOM = np.meshgrid(cx_grid, i_nom_grid)
    
    # Flatten for batch processing
    states = np.zeros((CX.size, 3)) # [cx, cN, cq]
    states[:, 0] = CX.flatten()
    states[:, 1] = 50.0 # Fixed Nitrate for the slice
    
    nom_actions = np.zeros((INOM.size, 2)) # [I, Fn]
    nom_actions[:, 0] = INOM.flatten()
    nom_actions[:, 1] = 5.0 # Fixed Feed for the slice
    
    # Convert to Tensors and Normalize
    device = next(resnet.parameters()).device
    s_tensor = (torch.FloatTensor(states).to(device) - s_mean.to(device)) / s_std.to(device)
    a_tensor = torch.FloatTensor(nom_actions).to(device)
    
    with torch.no_grad():
        safe_actions = resnet(s_tensor, a_tensor)
        i_safe = safe_actions[:, 0].cpu().numpy().reshape(CX.shape)

    # Plot the heatmap of corrections
    plt.figure(figsize=(8, 6))
    plt.contourf(CX, INOM, i_safe, levels=50, cmap='viridis')
    plt.colorbar(label='Safe Light Intensity ($\mu mol/m^2s$)')
    plt.title("ResNet Learned Safety Boundary (Fixed $C_N=50, F_n=5$)")
    plt.xlabel("Biomass ($C_x$)")
    plt.ylabel("Intended Light Intensity ($I_{nom}$)")
    plt.savefig("resnet_manifold_slice.png")
    plt.show()