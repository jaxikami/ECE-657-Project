import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

class DataLogger:
    def __init__(self):
        self.training_log = {"NonResNet": [], "SPRL": []}
        self.training_violations = {"NonResNet": [], "SPRL": []}
        self.eval_data = {"NonResNet": None, "SPRL": None}
        self.eval_violations = {"NonResNet": [], "SPRL": []}
        self.eval_violations_details = {"NonResNet": {"g1": [], "g2": [], "g3": []}, "SPRL": {"g1": [], "g2": [], "g3": []}}

    def log_training_episode(self, agent_name, total_reward, violation_count=0):
        self.training_log[agent_name].append(total_reward)
        self.training_violations[agent_name].append(violation_count)

    def log_evaluation_trajectory(self, agent_name, states, actions, rewards, info_list):
        # Updated to handle the flat info dictionary from env.py
        self.eval_data[agent_name] = {
            "states": np.array(states),   
            "actions": np.array(actions), 
            "rewards": np.array(rewards), 
            "is_safe": np.array([1 if i["violation_count"] == 0 else 0 for i in info_list]), # Map violations to safety
            "metrics": pd.DataFrame(info_list) 
        }
        
    def log_evaluation_episode_violations(self, agent_name, violation_count, g1_count=0, g2_count=0, g3_count=0):
        self.eval_violations[agent_name].append(violation_count)
        self.eval_violations_details[agent_name]["g1"].append(g1_count)
        self.eval_violations_details[agent_name]["g2"].append(g2_count)
        self.eval_violations_details[agent_name]["g3"].append(g3_count)

class Plotter:
    @staticmethod
    def plot_training_results(training_log, agent_name, window=50):
        rewards = training_log.get(agent_name, [])
        if len(rewards) < window: return

        plt.figure(figsize=(10, 6))
        mv_avg = pd.Series(rewards).rolling(window=window).mean()
        plt.plot(mv_avg, label=f"{agent_name} (MA {window})", color='tab:blue')
        plt.plot(rewards, alpha=0.15, color='tab:blue')
        
        # Scale y-axis based on the moving average to ignore extreme raw outliers
        valid_mv_avg = mv_avg.dropna()
        if len(valid_mv_avg) > 0:
            y_max = valid_mv_avg.max()
            y_min = -2.0 # Fixed lower bound as requested
            y_range = y_max - y_min
            # Add a 10% buffer on top, keep bottom at -2
            plt.ylim(y_min, y_max + 0.1 * y_range)
            
        plt.title(f"Training Convergence: {agent_name}")
        plt.xlabel("Episodes")
        plt.ylabel("Cumulative Reward")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"training_{agent_name}.png")
        plt.close()

    @staticmethod
    def plot_training_violations(logger):
        agents = ["NonResNet", "SPRL"]
        train_viols = []
        
        for agent in agents:
            t_viol = sum(logger.training_violations.get(agent, []))
            train_viols.append(t_viol)
            
        fig, ax1 = plt.subplots(figsize=(6, 5))
        ax1.bar(agents, train_viols, color=['tab:blue', 'tab:orange'])
        ax1.set_title("Total Training Violations")
        ax1.set_ylabel("Number of Violations")
        ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("training_violations.png")
        plt.close()
        
    @staticmethod
    def plot_evaluation_violations(logger, noise_level):
        agents = ["NonResNet", "SPRL"]
        eval_viols = []
        
        for agent in agents:
            e_viol = sum(logger.eval_violations.get(agent, []))
            eval_viols.append(e_viol)
            
        fig, ax2 = plt.subplots(figsize=(6, 5))
        ax2.bar(agents, eval_viols, color=['tab:blue', 'tab:orange'])
        ax2.set_title(f"Total Evaluation Violations (1000 noisy eps @ {noise_level})")
        ax2.set_ylabel("Number of Violations")
        ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("evaluation_violations.png")
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
        labels = [r"Biomass ($c_x$)", r"Nitrate ($c_N$)", r"Product ($c_q$)"]
        for i, label in enumerate(labels):
            ax = axes[i//2, i%2]
            ax.plot(time, data["states"][:, i], label=f"{agent_name}", color='tab:orange')
            ax.set_title(f"{label} over Time")
            ax.grid(True, alpha=0.2)

        # --- Actions: Light Intensity (I) ---
        I_phys = I_MIN + ((data["actions"][:, 0] + 1.0) / 2.0) * (I_MAX - I_MIN)
        axes[1, 1].plot(time, I_phys, label="Light Intensity", color='tab:green')
        axes[1, 1].set_title(r"Light Intensity ($I$) [$\mu mol/m^2/s$]")
        axes[1, 1].axhline(y=I_MAX, color='r', linestyle='--', alpha=0.3)
        
        # --- Actions: Nitrate Feed (Fn) ---
        Fn_phys = ((data["actions"][:, 1] + 1.0) / 2.0) * FN_MAX
        axes[2, 0].plot(time, Fn_phys, label="Nitrate Feed", color='tab:purple')
        axes[2, 0].set_title(r"Nitrate Feed ($F_N$) [$mg/L/h$]")

        # --- Safety Integrity ---
        axes[2, 1].step(time, data["is_safe"], where='post', color='tab:red')
        axes[2, 1].set_title("Safety Integrity (1=Safe, 0=Violation)")
        axes[2, 1].set_ylim([-0.1, 1.1])

        plt.tight_layout()
        plt.savefig(f"eval_trajectory_{agent_name}.png")
        plt.close()