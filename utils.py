import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

class DataLogger:
    def __init__(self):
        self.training_log = {"NonResNet": [], "SPRL": []}
        self.eval_data = {"NonResNet": None, "SPRL": None}

    def log_training_episode(self, agent_name, total_reward):
        self.training_log[agent_name].append(total_reward)

    def log_evaluation_trajectory(self, agent_name, states, actions, rewards, info_list):
        # Updated to handle the flat info dictionary from env.py
        self.eval_data[agent_name] = {
            "states": np.array(states),   
            "actions": np.array(actions), 
            "rewards": np.array(rewards), 
            "is_safe": np.array([1 if i["violation_count"] == 0 else 0 for i in info_list]), # Map violations to safety
            "metrics": pd.DataFrame(info_list) 
        }

class Plotter:
    @staticmethod
    def plot_training_results(training_log, agent_name, window=50):
        rewards = training_log.get(agent_name, [])
        if len(rewards) < window: return

        plt.figure(figsize=(10, 6))
        mv_avg = pd.Series(rewards).rolling(window=window).mean()
        plt.plot(mv_avg, label=f"{agent_name} (MA {window})", color='tab:blue')
        plt.plot(rewards, alpha=0.15, color='tab:blue')
        plt.title(f"Training Convergence: {agent_name}")
        plt.xlabel("Episodes")
        plt.ylabel("Cumulative Reward")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"training_{agent_name}.png")
        plt.close()

    @staticmethod
    def plot_evaluation_trajectories(eval_data, agent_name):
        data = eval_data.get(agent_name)
        if data is None: return

        # Constants from env.py
        I_MIN, I_MAX = 120.0, 400.0
        FN_MAX = 40.0

        fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
        time = np.arange(len(data["states"]))

        # --- States: Biomass, Nitrate, Product ---
        labels = ["Biomass ($c_x$)", "Nitrate ($c_N$)", "Product ($c_q$)"]
        for i, label in enumerate(labels):
            ax = axes[i//2, i%2]
            ax.plot(time, data["states"][:, i], label=f"{agent_name}", color='tab:orange')
            ax.set_title(f"{label} over Time")
            ax.grid(True, alpha=0.2)

        # --- Actions: Light Intensity (I) ---
        I_phys = I_MIN + ((data["actions"][:, 0] + 1.0) / 2.0) * (I_MAX - I_MIN)
        axes[1, 1].plot(time, I_phys, label="Light Intensity", color='tab:green')
        axes[1, 1].set_title("Light Intensity ($I$) [$\mu mol/m^2/s$]")
        axes[1, 1].axhline(y=I_MAX, color='r', linestyle='--', alpha=0.3)
        
        # --- Actions: Nitrate Feed (Fn) ---
        Fn_phys = ((data["actions"][:, 1] + 1.0) / 2.0) * FN_MAX
        axes[2, 0].plot(time, Fn_phys, label="Nitrate Feed", color='tab:purple')
        axes[2, 0].set_title("Nitrate Feed ($F_N$) [$mg/L/h$]")

        # --- Safety Integrity ---
        axes[2, 1].step(time, data["is_safe"], where='post', color='tab:red')
        axes[2, 1].set_title("Safety Integrity (1=Safe, 0=Violation)")
        axes[2, 1].set_ylim([-0.1, 1.1])

        plt.tight_layout()
        plt.savefig(f"eval_trajectory_{agent_name}.png")
        plt.close()