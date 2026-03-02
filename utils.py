import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

# Fix for OpenMP runtime conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class Logger:
    """
    Handles real-time tracking of rewards and constraint violations 
    during the 50,000 episode training run[cite: 16].
    """
    def __init__(self, experiment_name):
        self.name = experiment_name
        self.rewards = []
        self.violations = []
        self.avg_rewards = []
        self.violation_rates = []
        
    def log_episode(self, reward, violated):
        """Logs the total reward and safety status of a completed episode[cite: 19]."""
        self.rewards.append(reward)
        self.violations.append(1 if violated else 0)
        
        # Calculate 100-episode moving averages for smooth visualization [cite: 20]
        window = 100
        if len(self.rewards) >= window:
            self.avg_rewards.append(np.mean(self.rewards[-window:]))
            self.violation_rates.append(np.mean(self.violations[-window:]) * 100)
        else:
            self.avg_rewards.append(np.mean(self.rewards))
            self.violation_rates.append(np.mean(self.violations) * 100)

    def save_data(self):
        """Saves training logs to CSV for later analysis[cite: 19]."""
        df = pd.DataFrame({
            'reward': self.rewards,
            'violation': self.violations,
            'avg_reward': self.avg_rewards,
            'violation_rate': self.violation_rates
        })
        df.to_csv(f"{self.name}_logs.csv", index=False)

def plot_training_results(loggers):
    """
    Generates Learning Curves (Requirement 1): 
    Cumulative reward and violation rates vs. episode[cite: 20].
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    for logger in loggers:
        ax1.plot(logger.avg_rewards, label=f"{logger.name} Reward")
        ax2.plot(logger.violation_rates, label=f"{logger.name} Violation Rate (%)")
    
    ax1.set_ylabel("Average Reward (100-ep window)")
    ax1.set_title("Learning Curves: Convergence & Safety")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_ylabel("Violation Rate (%)")
    ax2.set_xlabel("Episode")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("training_benchmarks.png")
    plt.show()

def plot_evaluation_comparison(results):
    """
    Generates Comparative Histograms (Requirement 2):
    Benchmarks ResNet vs. Non-ResNet vs. MPC on Reward and Safety[cite: 21, 23, 24].
    'results' should be a dict: {'AgentName': {'reward': val, 'violations': val}}
    """
    # Attempt to load MPC baseline if not already in results
    if os.path.exists("mpc_results.csv"):
        mpc_df = pd.read_csv("mpc_results.csv")
        results['MPC Oracle'] = {
            'reward': mpc_df['reward'].iloc[0],
            'violations': mpc_df['violations'].iloc[0]
        }

    labels = list(results.keys())
    rewards = [results[l]['reward'] for l in labels]
    violations = [results[l]['violations'] for l in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Reward Bar 
    rects1 = ax1.bar(x - width/2, rewards, width, label='Avg Reward', color='#2ecc71')
    ax1.set_ylabel('Cumulative Reward')
    ax1.set_title('Final Evaluation Benchmarks (RL vs. MPC Oracle)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    
    # Violation Bar (Secondary Axis) 
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, violations, width, label='Violations', color='#e74c3c')
    ax2.set_ylabel('Avg Violations per Episode')
    
    # Combined legend
    lines, labels_lg = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels_lg + labels2, loc='upper left')

    fig.tight_layout()
    plt.savefig("evaluation_comparison.png")
    plt.show()

def plot_temporal_dynamics(rl_trajectories):
    """
    Generates Temporal Dynamics (Requirement 3):
    Phycocyanin production vs. time comparing RL to MPC baseline.
    'rl_trajectories' should be a dict of lists containing 'cq' over time.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot RL Variants
    for name, data in rl_trajectories.items():
        plt.plot(data, label=name, linewidth=2, alpha=0.8)
    
    # Load and Plot MPC Baseline 
    if os.path.exists("mpc_trajectory.npy"):
        mpc_data = np.load("mpc_trajectory.npy")
        plt.plot(mpc_data, label="MPC Oracle (Baseline)", color='black', linestyle='--', linewidth=2.5)
    
    plt.title("Temporal Dynamics: Phycocyanin Production (cq) Over Time")
    plt.xlabel("Time Step (Total: 7,200 Steps)")
    plt.ylabel("Phycocyanin Concentration (cq)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("production_dynamics.png")
    plt.show()