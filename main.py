import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
from env import PhycocyaninEnv
from lag_agent import NonResNet_Agent
from res_net_agent import SPRL_Agent
from utils import DataLogger, Plotter
from torch.optim.lr_scheduler import LinearLR

# =============================================================================
# HYPERPARAMETERS
# =============================================================================
STATE_DIM = 4      # State space: [Biomass (cx), Nitrate (cN), Phycocyanin (cq), Normalized Time (t_norm)]
ACTION_DIM = 2     # Action space: [Light Intensity (I), Nitrate Feed (Fn)]
MAX_EPISODES = 50000
UPDATE_TIMESTEP = 1000 # Number of steps collected before triggering a PPO update
K_EPOCHS = 40          # Number of optimization epochs per PPO update
EPS_CLIP = 0.3         # PPO surrogate objective clipping parameter
GAMMA = 0.99           # Discount factor for future rewards
LR_ACTOR = 3e-4        # Initial learning rate for the Actor network
LR_CRITIC = 1e-3       # Initial learning rate for the Critic network
MIN_LR = 1e-5          # Minimum learning rate bound for the linear scheduler
ENTROPY_COEFF = 0.05   # Exploration coefficient for the SPRL agent
EVALUATE_ONLY = False  # Set to False to run the full training loop before evaluation
NOISE_STD = 0.05       # Standard deviation of the Gaussian noise applied to agent intent during evaluation

class Memory:
    """
    Buffer for storing environment trajectories generated during policy rollouts.
    """
    def __init__(self):
        # The unbounded `raw_actions` must be explicitly stored to accurately recalculate 
        # log probabilities during the PPO evaluate() step.
        self.states, self.actions, self.raw_actions, self.logprobs, self.rewards, self.is_terminals = [], [], [], [], [], []
        
    def clear(self):
        del self.states[:], self.actions[:], self.raw_actions[:], self.logprobs[:], self.rewards[:], self.is_terminals[:]

def train_agent(agent_name, agent, logger):
    """
    Executes the primary training loop for a specified RL agent.
    
    The environment is stepped iteratively, and trajectory data is accumulated in memory.
    Upon reaching the defined `UPDATE_TIMESTEP`, the agent's PPO learning algorithm is triggered.
    A linear learning rate scheduler and an early stopping mechanism based on performance plateaus
    are utilized to optimize training efficiency.
    """
    print(f"\n--- Starting Training: {agent_name} ---")
    env = PhycocyaninEnv()
    memory = Memory()
    scheduler = LinearLR(agent.optimizer, start_factor=1.0, end_factor=0.01, total_iters=MAX_EPISODES)
    
    time_step = 0
    rewards_history = []
    best_moving_avg = -float('inf')
    plateau_counter = 0
    
    # Early Exit Parameters
    WINDOW_SIZE, PATIENCE, EARLY_EXIT_START = 150, 1500, 15000    
    
    pbar = tqdm(range(1, MAX_EPISODES + 1), desc=f"Training {agent_name}")
    for i_episode in pbar:
        state = env.reset()
        current_ep_reward = 0
        
        while True:
            time_step += 1
            
            # Action Selection
            # The agent generates an intent and processes it through its specific architecture.
            # Both agents return the squashed action, its log probability, and the raw unbounded sample.
            action, log_prob, raw_act = agent.select_action(state)
            
            # Memory Storage
            memory.states.append(torch.tensor(state, dtype=torch.float32))
            memory.actions.append(torch.tensor(action, dtype=torch.float32))
            memory.raw_actions.append(torch.tensor(raw_act, dtype=torch.float32))
            memory.logprobs.append(torch.tensor(log_prob, dtype=torch.float32))
            
            # Environment Step
            state, reward, done, info = env.step(action)
            
            current_ep_reward += reward
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            # PPO Update Trigger
            if time_step % UPDATE_TIMESTEP == 0:
                agent.learn(memory)                
                memory.clear()
                time_step = 0
                
            if done: break
            
        scheduler.step()
        logger.log_training_episode(agent_name, current_ep_reward, info["violation_count"])
        rewards_history.append(current_ep_reward)
        
        # --- Plateau Detection and Early Stopping ---
        # Evaluated only after a minimum number of episodes to ensure training stability.
        if i_episode >= EARLY_EXIT_START:
            recent_avg = np.mean(rewards_history[-WINDOW_SIZE:])
            if recent_avg > best_moving_avg:
                best_moving_avg = recent_avg
                plateau_counter = 0 
            else:
                plateau_counter += 1
                
            if plateau_counter >= PATIENCE: 
                print(f"Early stopping triggered at episode {i_episode} due to performance plateau.")
                break

        if i_episode % 10 == 0:
            pbar.set_postfix({
                "AvgR": f"{info['avg_reward']:.3f}",
                "G1P": f"{info['avg_g1_penalty']:.3f}",
                "G2P": f"{info['avg_g2_penalty']:.3f}",
                "G3P": f"{info['avg_g3_penalty']:.3f}",
                "SmP": f"{info['avg_smooth_penalty']:.3f}",
                "NUsP": f"{info['avg_nitrate_usage_penalty']:.3f}",
                "Vio": f"{info['violation_count']}",
                "LR": f"{agent.optimizer.param_groups[0]['lr']:.1e}"
            })

    torch.save(agent.policy.state_dict(), f"{agent_name}_final_weights.pth")
    Plotter.plot_training_results(logger.training_log, agent_name=agent_name)

def evaluate_agent(agent_name, agent, logger, eval_episodes=5000, noise_std=0.05):
    """
    Evaluates a trained agent's robustness against both initial state variations
    and continuous intent observation noise.
    
    The agent's nominal intent is corrupted with Gaussian noise before being processed
    by the architecture (e.g., intercepted by the Safety Filter for SPRL). Constraint violations 
    and overall performance are rigorously logged to quantify the safety efficacy of the models.
    """
    print(f"\n--- Evaluating: {agent_name} with Randomized Initial State + N(0, {noise_std}) Intent Noise ---")
    
    load_path = f"{agent_name}_final_weights.pth"
    if os.path.exists(load_path):
        agent.policy.load_state_dict(torch.load(load_path))
        agent.policy.eval()

    env = PhycocyaninEnv()
    best_states, best_actions, best_rewards, best_infos = [], [], [], []
    best_ep_reward = -float('inf')
    total_g1, total_g2, total_g3 = 0, 0, 0
    
    for _ in range(eval_episodes):
        # The environment is reset with randomized initial conditions for robust evaluation.
        state = env.reset(randomize=True)
        
        ep_states, ep_actions, ep_rewards, ep_infos = [], [], [], []
        
        while True:
            # Deterministic Intent Generation
            with torch.no_grad():
                state_t = torch.FloatTensor(state).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).unsqueeze(0)
                intent = torch.tanh(agent.policy.actor(state_t)).cpu().numpy().flatten()
            
            # Intent Corruption
            # Gaussian variation is applied to the calculated intent, simulating
            # actuator imprecision or policy drift.
            if noise_std > 0:
                noise = np.random.normal(0, noise_std, size=intent.shape)
                noisy_intent = np.clip(intent + noise, -1.0, 1.0)
            else:
                noisy_intent = intent
                
            # SPRL Safeguard Application
            # If the architecture includes an active safeguard, the corrupted intent is intercepted
            # and corrected prior to environmental execution.
            with torch.no_grad():
                if agent_name == "SPRL":
                    noisy_intent_t = torch.FloatTensor(noisy_intent).to(state_t.device).unsqueeze(0)
                    phys_scale = torch.tensor([6.0, 800.0, 0.1, 1.0], device=state_t.device)
                    action = agent.safeguard(state_t * phys_scale, noisy_intent_t).cpu().numpy().flatten()
                else:
                    action = noisy_intent
                
            next_state, reward, done, info = env.step(action)
            ep_states.append(state)
            ep_actions.append(action)
            ep_rewards.append(reward)
            ep_infos.append(info)
            state = next_state
            if done: break
        
        # Best Trajectory Storage
        # The episode achieving the highest total reward is retained for downstream visualization.
        ep_total_reward = sum(ep_rewards)
        if ep_total_reward > best_ep_reward:
            best_ep_reward = ep_total_reward
            best_states, best_actions, best_rewards, best_infos = ep_states, ep_actions, ep_rewards, ep_infos
        
        # Violation Archiving
        if len(ep_infos) > 0:
            last_info = ep_infos[-1]
            logger.log_evaluation_episode_violations(
                agent_name, 
                last_info["violation_count"],
                last_info.get("g1_violation_count", 0),
                last_info.get("g2_violation_count", 0),
                last_info.get("g3_violation_count", 0)
            )
            total_g1 += last_info.get("g1_violation_count", 0)
            total_g2 += last_info.get("g2_violation_count", 0)
            total_g3 += last_info.get("g3_violation_count", 0)

    print(f"{agent_name} Evaluation Violations - G1 (Path Nitrate): {total_g1}, G2 (Ratio): {total_g2}, G3 (Terminal Nitrate): {total_g3}")

    # Compilation and Visual Output
    for i in best_infos: i["is_safe"] = 1 if i["violation_count"] == 0 else 0
    
    logger.log_evaluation_trajectory(agent_name, best_states, best_actions, best_rewards, best_infos)
    print(f"Best Episode Plotted Reward: {best_ep_reward:.2f}")
    Plotter.plot_evaluation_trajectories(logger.eval_data, agent_name=agent_name)

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================
if __name__ == "__main__":
    logger = DataLogger()
    lag_agent = NonResNet_Agent(STATE_DIM, ACTION_DIM, LR_ACTOR, LR_CRITIC, GAMMA, K_EPOCHS, EPS_CLIP)
    sprl_agent = SPRL_Agent(STATE_DIM, ACTION_DIM, LR_ACTOR, LR_CRITIC, GAMMA, K_EPOCHS, EPS_CLIP, ENTROPY_COEFF)
    
    # Training
    if not EVALUATE_ONLY:
        train_agent("NonResNet", lag_agent, logger)
        train_agent("SPRL", sprl_agent, logger)
        
    # Evaluation (with aggressive 20% intent noise)
    evaluate_agent("NonResNet", lag_agent, logger, noise_std=NOISE_STD)
    evaluate_agent("SPRL", sprl_agent, logger, noise_std=NOISE_STD)
    
    if not EVALUATE_ONLY:
        Plotter.plot_training_violations(logger)
        
    # Plot Evaluation Violations
    Plotter.plot_evaluation_violations(logger, noise_level=NOISE_STD)