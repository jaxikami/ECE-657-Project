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
        # Convert lists to numpy for easier indexing
        self.eval_data[agent_name] = {
            "states": np.array(states),   # [T, 3]
            "actions": np.array(actions), # [T, 2]
            "rewards": np.array(rewards), # [T]
            "is_safe": np.array([i["is_safe"] for i in info_list]),
            "breakdown": pd.DataFrame([i["reward_breakdown"] for i in info_list])
        }

class Plotter:
    @staticmethod
    def plot_training_results(training_log, window=50):
        """1. Reward vs Episodes (50-episode moving average)"""
        plt.figure(figsize=(10, 6))
        for agent_name, rewards in training_log.items():
            if len(rewards) < window: continue
            
            # Calculate Moving Average
            mv_avg = pd.Series(rewards).rolling(window=window).mean()
            plt.plot(mv_avg, label=f"{agent_name} (MA {window})")
            
            # Add semi-transparent raw data
            plt.plot(rewards, alpha=0.15, color=plt.gca().lines[-1].get_color())

        plt.title("Training Convergence: Reward vs Episodes")
        plt.xlabel("Episodes")
        plt.ylabel("Cumulative Reward")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("training_comparison.png")
        plt.show()

    @staticmethod
    def plot_evaluation_trajectories(eval_data):
        """
        2. Average [I, Fn] vs Time
        3. [cx, cN, cq] vs Time
        4. Constraint Violations
        """
        fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
        time = np.arange(len(eval_data["SPRL"]["states"]))

        # --- Subplot 1: States [cx, cN, cq] ---
        for i, label in enumerate(["Biomass ($c_x$)", "Nitrate ($c_N$)", "Product ($c_q$)"]):
            for agent in ["NonResNet", "SPRL"]:
                axes[i//2, i%2].plot(time, eval_data[agent]["states"][:, i], label=f"{agent}")
            axes[i//2, i%2].set_title(f"{label} over Time")
            axes[i//2, i%2].legend()
            axes[i//2, i%2].grid(True, alpha=0.2)

        # --- Subplot 2: Actions [I, F_N] ---
        # Light Intensity (I)
        for agent in ["NonResNet", "SPRL"]:
            # Note: Actions from agent are [-1, 1], we map to physical for plotting
            I_phys = (eval_data[agent]["actions"][:, 0] + 1) * 1500.0
            axes[1, 1].plot(time, I_phys, label=f"{agent}")
        axes[1, 1].set_title("Light Intensity ($I$) over Time")
        axes[1, 1].axhline(y=450, color='r', linestyle='--', alpha=0.5, label="I_crit (Unshaded)")
        axes[1, 1].legend()

        # Nitrate Feed (Fn)
        for agent in ["NonResNet", "SPRL"]:
            Fn_phys = (eval_data[agent]["actions"][:, 1] + 1) * 10.0
            axes[2, 0].plot(time, Fn_phys, label=f"{agent}")
        axes[2, 0].set_title("Nitrate Feed ($F_N$) over Time")
        axes[2, 0].legend()

        # --- Subplot 3: Constraint Violations ---
        for agent in ["NonResNet", "SPRL"]:
            # 1 = Safe, 0 = Violation
            violation_mask = eval_data[agent]["is_safe"].astype(int)
            axes[2, 1].step(time, violation_mask, label=f"{agent}", where='post')
        
        axes[2, 1].set_title("Safety Integrity (1=Safe, 0=Violation)")
        axes[2, 1].set_ylim([-0.1, 1.1])
        axes[2, 1].legend()

        plt.tight_layout()
        plt.savefig("evaluation_trajectories.png")
        plt.show()