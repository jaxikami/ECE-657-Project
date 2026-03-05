import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

class DataLogger:
    def __init__(self):
        self.training_log = {"NonResNet": [], "SPRL": []}
        self.eval_data = {"NonResNet": None, "SPRL": None}

    def log_training_episode(self, agent_name, total_reward):
        """Logs the total reward of a training episode."""
        self.training_log[agent_name].append(total_reward)

    def log_evaluation_trajectory(self, agent_name, states, actions, rewards, info_list):
        """Logs a full episode trajectory for state/action plotting."""
        self.eval_data[agent_name] = {
            "states": np.array(states),   # [T, 3]
            "actions": np.array(actions), # [T, 2]
            "rewards": np.array(rewards), # [T]
            "is_safe": np.array([i["is_safe"] for i in info_list]),
            "breakdown": pd.DataFrame([i["reward_breakdown"] for i in info_list]) if "reward_breakdown" in info_list[0] else None
        }

class Plotter:
    @staticmethod
    def plot_training_results(training_log, agent_name, window=50):
        """Saves Reward vs Episodes for a specific agent (moving average)."""
        rewards = training_log.get(agent_name, [])
        if len(rewards) < window: 
            return

        plt.figure(figsize=(10, 6))
        # Calculate Moving Average
        mv_avg = pd.Series(rewards).rolling(window=window).mean()
        plt.plot(mv_avg, label=f"{agent_name} (MA {window})", color='tab:blue')
        
        # Add semi-transparent raw data
        plt.plot(rewards, alpha=0.15, color='tab:blue')

        plt.title(f"Training Convergence: {agent_name}")
        plt.xlabel("Episodes")
        plt.ylabel("Cumulative Reward")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        filename = f"training_{agent_name}.png"
        plt.savefig(filename)
        plt.close() # Close to free memory and prevent blocking
        print(f"Training plot saved as {filename}")

    @staticmethod
    def plot_evaluation_trajectories(eval_data, agent_name):
        """Saves states, actions, and safety plots for a specific agent."""
        data = eval_data.get(agent_name)
        if data is None:
            return

        fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
        time = np.arange(len(data["states"]))

        # --- Subplot 1: States [cx, cN, cq] ---
        labels = ["Biomass ($c_x$)", "Nitrate ($c_N$)", "Product ($c_q$)"]
        for i, label in enumerate(labels):
            ax = axes[i//2, i%2]
            ax.plot(time, data["states"][:, i], label=f"{agent_name}", color='tab:orange')
            ax.set_title(f"{label} over Time")
            ax.legend()
            ax.grid(True, alpha=0.2)

        # --- Subplot 2: Actions [I, F_N] ---
        # Light Intensity (I)
        I_phys = (data["actions"][:, 0] + 1) * 1500.0 / 2.0 # Adjusted based on common scaling
        axes[1, 1].plot(time, I_phys, label=f"{agent_name}", color='tab:green')
        axes[1, 1].set_title("Light Intensity ($I$) over Time")
        axes[1, 1].axhline(y=450, color='r', linestyle='--', alpha=0.5, label="I_crit")
        axes[1, 1].legend()

        # Nitrate Feed (Fn)
        Fn_phys = (data["actions"][:, 1] + 1) * 10.0
        axes[2, 0].plot(time, Fn_phys, label=f"{agent_name}", color='tab:purple')
        axes[2, 0].set_title("Nitrate Feed ($F_N$) over Time")
        axes[2, 0].legend()

        # --- Subplot 3: Constraint Violations ---
        violation_mask = data["is_safe"].astype(int)
        axes[2, 1].step(time, violation_mask, label=f"{agent_name}", where='post', color='tab:red')
        axes[2, 1].set_title("Safety Integrity (1=Safe, 0=Violation)")
        axes[2, 1].set_ylim([-0.1, 1.1])
        axes[2, 1].legend()

        plt.tight_layout()
        filename = f"eval_trajectory_{agent_name}.png"
        plt.savefig(filename)
        plt.close()
        print(f"Evaluation plot saved as {filename}")