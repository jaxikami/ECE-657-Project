"""
Utility module for metrics tracking and visualization.

This module provides the `DataLogger` for recording episode trajectories,
rewards, and constraint violations during both training and evaluation phases.
It also includes the `Plotter` class for generating comparative visualizations
of agent performance and safety integrity.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Configure matplotlib for research paper quality plots
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'lines.linewidth': 2
})

class DataLogger:
    """
    Centralized logging architecture for recording agent metrics.
    Maintains separate dictionaries for standard continuous tracking.
    """
    def __init__(self):
        # 1. Training metrics
        self.training_log = {"NonResNet": [], "SPRL": []}
        self.training_violations = {"NonResNet": [], "SPRL": []}
        
        # 2. Evaluation metrics
        self.eval_data = {"NonResNet": None, "SPRL": None}
        self.eval_violations = {"NonResNet": [], "SPRL": []}
        self.eval_violations_details = {"NonResNet": {"g1": [], "g2": [], "g3": []}, "SPRL": {"g1": [], "g2": [], "g3": []}}

    def log_training_episode(self, agent_name, total_reward, violation_count=0):
        self.training_log[agent_name].append(total_reward)
        self.training_violations[agent_name].append(violation_count)

    def log_evaluation_trajectory(self, agent_name, states, actions, rewards, info_list, agg_data=None):
        """
        Records a full environmental trajectory from the evaluation phase.
        The flat info dictionary returned by the environment is parsed and stored.
        """
        self.eval_data[agent_name] = {
            "states": np.array(states),   
            "actions": np.array(actions), 
            "rewards": np.array(rewards), 
            "is_safe": np.array([1 if i["violation_count"] == 0 else 0 for i in info_list]), # Map physical violations to Boolean safety
            "metrics": pd.DataFrame(info_list),
            "agg_data": agg_data
        }
        
    def log_evaluation_episode_violations(self, agent_name, violation_count, g1_count=0, g2_count=0, g3_count=0):
        self.eval_violations[agent_name].append(violation_count)
        self.eval_violations_details[agent_name]["g1"].append(g1_count)
        self.eval_violations_details[agent_name]["g2"].append(g2_count)
        self.eval_violations_details[agent_name]["g3"].append(g3_count)

class Plotter:
    """
    Static utility class responsible for generating and saving matplotlib visualizations.
    """
    @staticmethod
    def plot_training_results(training_log, agent_name, window=50):
        """
        Generates a moving-average convergence plot of the cumulative training rewards.
        """
        rewards = training_log.get(agent_name, [])
        if len(rewards) < window: return

        plt.figure(figsize=(10, 6))
        mv_avg = pd.Series(rewards).rolling(window=window).mean()
        plt.plot(mv_avg, label=f"{agent_name} (MA {window})", color='tab:blue')
        plt.plot(rewards, alpha=0.15, color='tab:blue')
        
        # The y-axis is dynamically scaled based on the moving average to ignore extreme initial raw outliers
        valid_mv_avg = mv_avg.dropna()
        if len(valid_mv_avg) > 0:
            y_max = valid_mv_avg.max()
            y_min = -2.0 # A fixed lower bound is explicitly applied to maintain visual context
            y_range = y_max - y_min
            # A 10% visual buffer is added to the upper bound
            plt.ylim(y_min, y_max + 0.1 * y_range)
            
        plt.title(f"Training Convergence: {agent_name}")
        plt.xlabel("Episodes")
        plt.ylabel("Cumulative Reward")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"training_{agent_name}.png", dpi=300, bbox_inches='tight')
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
        ax1.set_xlabel("Agent")
        ax1.set_ylabel("Number of Violations")
        ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("training_violations.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    @staticmethod
    def plot_comprehensive_evaluation(eval_data, eval_violations):
        """
        Generates a 6-panel subplot comparing NonResNet and SPRL agents.
        1. NonResNet Nitrate levels with shaded min/max bounds
        2. SPRL Nitrate levels with shaded min/max bounds
        3. Total evaluation violations bar chart
        4. Average Phycocyanin production over time
        5. Best case Light Intensity step plot
        6. Best case Nitrate Feed step plot
        """
        if "NonResNet" not in eval_data or "SPRL" not in eval_data:
            print("Evaluation data missing for one or both agents. Cannot plot comprehensive evaluation.")
            return
            
        nr_data = eval_data["NonResNet"]
        sprl_data = eval_data["SPRL"]
        
        # Constants
        I_MIN, I_MAX = 120.0, 400.0
        FN_MAX = 40.0
        N_LIMIT_PATH = 800.0
        N_LIMIT_TERM = 150.0

        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        if nr_data is not None and len(nr_data["states"]) > 0:
            time = np.arange(len(nr_data["states"]))
        elif sprl_data is not None and len(sprl_data["states"]) > 0:
            time = np.arange(len(sprl_data["states"]))
        else:
            return

        # ---------------------------------------------------------
        # 1. NonResNet Nitrate Level
        # ---------------------------------------------------------
        ax = axes[0, 0]
        if nr_data is not None:
            nr_n_best = nr_data["states"][:, 1] * 800.0
            nr_agg = nr_data.get("agg_data", {})
            ax.plot(time, nr_n_best, label="Best Run", color='tab:blue')
            if nr_agg:
                ax.fill_between(time, nr_agg["nitrate_min"] * 800.0, nr_agg["nitrate_max"] * 800.0, color='tab:blue', alpha=0.2, label="Min/Max Bounds")
            
        ax.axhline(y=N_LIMIT_PATH, color='r', linestyle='--', alpha=0.5, label="Max Path Limit ($g_1$)")
        ax.axhline(y=N_LIMIT_TERM, color='darkred', linestyle='--', alpha=0.5, label="Max Exit Limit ($g_3$)")
        ax.set_title(r"Nitrate Level ($c_N$) - NonResNet")
        ax.set_ylabel("mg/L")
        ax.set_xlabel("Time Step")
        ax.grid(True, alpha=0.2)
        ax.legend()

        # ---------------------------------------------------------
        # 2. SPRL Nitrate Level
        # ---------------------------------------------------------
        ax = axes[0, 1]
        if sprl_data is not None:
            s_n_best = sprl_data["states"][:, 1] * 800.0
            s_agg = sprl_data.get("agg_data", {})
            ax.plot(time, s_n_best, label="Best Run", color='tab:orange')
            if s_agg:
                ax.fill_between(time, s_agg["nitrate_min"] * 800.0, s_agg["nitrate_max"] * 800.0, color='tab:orange', alpha=0.2, label="Min/Max Bounds")
            
        ax.axhline(y=N_LIMIT_PATH, color='r', linestyle='--', alpha=0.5, label="Max Path Limit ($g_1$)")
        ax.axhline(y=N_LIMIT_TERM, color='darkred', linestyle='--', alpha=0.5, label="Max Exit Limit ($g_3$)")
        ax.set_title(r"Nitrate Level ($c_N$) - SPRL")
        ax.set_ylabel("mg/L")
        ax.set_xlabel("Time Step")
        ax.grid(True, alpha=0.2)
        ax.legend()
        
        # ---------------------------------------------------------
        # 3. Constraint Violations Bar Chart
        # ---------------------------------------------------------
        ax = axes[1, 0]
        agents = ["NonResNet", "SPRL"]
        viols = [sum(eval_violations.get(a, [])) for a in agents]
        bars = ax.bar(agents, viols, color=['tab:blue', 'tab:orange'])
        ax.set_title("Total Evaluation Violations")
        ax.set_ylabel("Count")
        ax.set_xlabel("Agent")
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        # Add labels on top of bars
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')

        # ---------------------------------------------------------
        # 4. Average Production over Time
        # ---------------------------------------------------------
        ax = axes[1, 1]
        if nr_data is not None and nr_data.get("agg_data"):
            ax.plot(time, nr_data["agg_data"]["production_avg"] * 0.2, label="NonResNet Avg", color='tab:blue')
        if sprl_data is not None and sprl_data.get("agg_data"):
            ax.plot(time, sprl_data["agg_data"]["production_avg"] * 0.2, label="SPRL Avg", color='tab:orange')
        ax.set_title(r"Average Phycocyanin Production ($c_q$)")
        ax.set_ylabel("g/L")
        ax.set_xlabel("Time Step")
        ax.grid(True, alpha=0.2)
        ax.legend()

        # ---------------------------------------------------------
        # 5. Best Case Light Intensity (I)
        # ---------------------------------------------------------
        ax = axes[2, 0]
        if nr_data is not None:
            nr_I_phys = I_MIN + ((nr_data["actions"][:, 0] + 1.0) / 2.0) * (I_MAX - I_MIN)
            t_act = np.arange(len(nr_I_phys) + 1)
            ax.step(t_act, np.append(nr_I_phys, nr_I_phys[-1]), where='post', label="NonResNet", color='tab:blue')
        if sprl_data is not None:
            s_I_phys = I_MIN + ((sprl_data["actions"][:, 0] + 1.0) / 2.0) * (I_MAX - I_MIN)
            t_act_s = np.arange(len(s_I_phys) + 1)
            ax.step(t_act_s, np.append(s_I_phys, s_I_phys[-1]), where='post', label="SPRL", color='tab:orange')
            
        ax.axhline(y=I_MAX, color='r', linestyle='--', alpha=0.3)
        ax.set_title(r"Light Intensity ($I$)")
        ax.set_ylabel(r"$\mu mol/m^2/s$")
        ax.set_xlabel("Time Step")
        ax.grid(True, alpha=0.2)
        ax.legend()

        # ---------------------------------------------------------
        # 6. Best Case Nitrate Feed (Fn)
        # ---------------------------------------------------------
        ax = axes[2, 1]
        if nr_data is not None:
            nr_Fn_phys = ((nr_data["actions"][:, 1] + 1.0) / 2.0) * FN_MAX
            t_act = np.arange(len(nr_Fn_phys) + 1)
            ax.step(t_act, np.append(nr_Fn_phys, nr_Fn_phys[-1]), where='post', label="NonResNet", color='tab:blue')
        if sprl_data is not None:
            s_Fn_phys = ((sprl_data["actions"][:, 1] + 1.0) / 2.0) * FN_MAX
            t_act_s = np.arange(len(s_Fn_phys) + 1)
            ax.step(t_act_s, np.append(s_Fn_phys, s_Fn_phys[-1]), where='post', label="SPRL", color='tab:orange')

        ax.set_title(r"Nitrate Feed ($F_N$)")
        ax.set_ylabel(r"$mg/L/h$")
        ax.set_xlabel("Time Step")
        ax.grid(True, alpha=0.2)
        ax.legend()

        plt.tight_layout()
        plt.savefig("comprehensive_evaluation.png", dpi=300, bbox_inches='tight')
        plt.close()