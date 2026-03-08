import torch
import numpy as np
import os
from tqdm import tqdm
from env import PhycocyaninEnv  # 1. Updated Class Name
from lag_agent import NonResNet_Agent
from res_net_agent import SPRL_Agent
from utils import DataLogger, Plotter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# --- Hyperparameters Updated ---
STATE_DIM = 4      # 2. Updated to 4 (cx, cN, cq, t_norm)
ACTION_DIM = 2
MAX_EPISODES = 50000 
UPDATE_TIMESTEP = 2000 # Adjusted for stability 
K_EPOCHS = 40
EPS_CLIP = 0.2
GAMMA = 0.99
LR_ACTOR = 3e-4
LR_CRITIC = 1e-3
MIN_LR = 1e-5 
ENTROPY_COEFF = 0.05

class Memory:
    def __init__(self):
        # 3. Added raw_actions to store the pre-tanh/unsquashed samples for PPO evaluate()
        self.states, self.actions, self.raw_actions, self.logprobs, self.rewards, self.is_terminals = [], [], [], [], [], []
    def clear(self):
        del self.states[:], self.actions[:], self.raw_actions[:], self.logprobs[:], self.rewards[:], self.is_terminals[:]

def train_agent(agent_name, agent, logger):
    print(f"\n--- Starting Training: {agent_name} ---")
    env = PhycocyaninEnv() # 4. Matches env.py
    memory = Memory()
    scheduler = CosineAnnealingWarmRestarts(agent.optimizer, T_0=5000, T_mult=2, eta_min=MIN_LR)
    
    time_step = 0
    rewards_history = []
    best_moving_avg = -float('inf')
    plateau_counter = 0
    
    # Early Exit Params
    WINDOW_SIZE, PATIENCE, EARLY_EXIT_START = 500, 1000, 10000    
    
    pbar = tqdm(range(1, MAX_EPISODES + 1), desc=f"Training {agent_name}")
    for i_episode in pbar:
        state = env.reset()
        current_ep_reward = 0
        
        while True: # 5. Use while loop to respect env's 'done' signal
            time_step += 1
            
            # Select action
            if agent_name == "SPRL":
                # Returns: safe_action, log_prob, z_raw
                action, log_prob, raw_act = agent.select_action(state)
            else:
                # Returns: action, log_prob, raw_action
                action, log_prob, raw_act = agent.select_action(state)
            
            memory.states.append(torch.tensor(state))
            memory.actions.append(torch.tensor(action))
            memory.raw_actions.append(torch.tensor(raw_act))
            memory.logprobs.append(torch.tensor(log_prob))
            
            state, reward, done, info = env.step(action)
            
            current_ep_reward += reward
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            if time_step % UPDATE_TIMESTEP == 0:
                agent.learn(memory)
                memory.clear()
                time_step = 0
            if done: break
            
        logger.log_training_episode(agent_name, current_ep_reward)
        rewards_history.append(current_ep_reward)
        scheduler.step(i_episode)
        
        # --- Simplified Plateau Logic ---
        if i_episode >= EARLY_EXIT_START:
            recent_avg = np.mean(rewards_history[-WINDOW_SIZE:])
            if recent_avg > best_moving_avg:
                best_moving_avg = recent_avg
                plateau_counter = 0 
            else:
                plateau_counter += 1
            if plateau_counter >= PATIENCE: break

        if i_episode % 10 == 0:
            pbar.set_postfix({
                "Reward": f"{current_ep_reward:.1f}",
                "Vio": f"{info['violation_count']}",
                "LR": f"{agent.optimizer.param_groups[0]['lr']:.1e}"
            })

    torch.save(agent.policy.state_dict(), f"{agent_name}_final_weights.pth")
    Plotter.plot_training_results(logger.training_log, agent_name=agent_name)

def evaluate_agent(agent_name, agent, logger, eval_episodes=10):
    print(f"\n--- Evaluating: {agent_name} ---")
    load_path = f"{agent_name}_final_weights.pth"
    if os.path.exists(load_path):
        agent.policy.load_state_dict(torch.load(load_path))
        agent.policy.eval()

    env = PhycocyaninEnv()
    all_states, all_actions, all_rewards, all_infos = [], [], [], []
    
    for _ in range(eval_episodes):
        state = env.reset()
        ep_states, ep_actions, ep_rewards, ep_infos = [], [], [], []
        
        while True:
            # Deterministic Action Selection
            with torch.no_grad():
                state_t = torch.FloatTensor(state).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).unsqueeze(0)
                if agent_name == "SPRL":
                    # Intent z from actor
                    z = torch.tanh(agent.policy.actor(state_t))
                    # Safeguard projection (using physical units)
                    phys_scale = torch.tensor([6.0, 800.0, 0.1, 1.0], device=state_t.device)
                    action = agent.safeguard(state_t * phys_scale, z).cpu().numpy().flatten()
                else:
                    action = torch.tanh(agent.policy.actor(state_t)).cpu().numpy().flatten()
            
            next_state, reward, done, info = env.step(action)
            ep_states.append(state)
            ep_actions.append(action)
            ep_rewards.append(reward)
            ep_infos.append(info)
            state = next_state
            if done: break
        
        # Store last trajectory for Plotter
        all_states, all_actions, all_rewards, all_infos = ep_states, ep_actions, ep_rewards, ep_infos

    for i in all_infos: i["is_safe"] = 1 if i["violation_count"] == 0 else 0
    
    logger.log_evaluation_trajectory(agent_name, all_states, all_actions, all_rewards, all_infos)
    Plotter.plot_evaluation_trajectories(logger.eval_data, agent_name=agent_name)

if __name__ == "__main__":
    logger = DataLogger()
    lag_agent = NonResNet_Agent(STATE_DIM, ACTION_DIM, LR_ACTOR, LR_CRITIC, GAMMA, K_EPOCHS, EPS_CLIP)
    sprl_agent = SPRL_Agent(STATE_DIM, ACTION_DIM, LR_ACTOR, LR_CRITIC, GAMMA, K_EPOCHS, EPS_CLIP, ENTROPY_COEFF)
    
    # Training
    train_agent("NonResNet", lag_agent, logger)
    train_agent("SPRL", sprl_agent, logger)
        
    # Evaluation
    evaluate_agent("NonResNet", lag_agent, logger)
    evaluate_agent("SPRL", sprl_agent, logger)