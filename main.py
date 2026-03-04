import torch
import numpy as np
import os
from tqdm import tqdm
from env import PhotoProductionEnv
from lag_agent import NonResNet_Agent
from res_net_agent import SPRL_Agent
from utils import DataLogger, Plotter

# --- Hyperparameters remain the same ---
STATE_DIM = 3
ACTION_DIM = 2
LATENT_DIM = 2
MAX_EPISODES = 50000 
MAX_STEPS = 7200      
UPDATE_TIMESTEP = 14400 
K_EPOCHS = 40
EPS_CLIP = 0.2
GAMMA = 0.99
LR_ACTOR = 3e-4
LR_CRITIC = 1e-3
MIN_LR = 1e-7  
ENTROPY_COEFF = 0.01

class Memory:
    def __init__(self):
        self.states, self.actions, self.logprobs, self.rewards, self.is_terminals = [], [], [], [], []
    def clear(self):
        del self.states[:], self.actions[:], self.logprobs[:], self.rewards[:], self.is_terminals[:]

def apply_linear_decay(agent, episode, total_episodes, base_lr_actor, base_lr_critic):
    lr_coeff = max(0.0, 1.0 - (episode / total_episodes))
    current_lr_actor = max(MIN_LR, base_lr_actor * lr_coeff)
    current_lr_critic = max(MIN_LR, base_lr_critic * lr_coeff)
    for i in range(len(agent.optimizer.param_groups) - 1):
        agent.optimizer.param_groups[i]['lr'] = current_lr_actor
    agent.optimizer.param_groups[-1]['lr'] = current_lr_critic
    return current_lr_actor

def train_agent(agent_name, agent, logger):
    """Trains agent with plateau-based early exit and saves weights."""
    print(f"\n--- Starting Training: {agent_name} ---")
    env = PhotoProductionEnv(train_mode=True)
    memory = Memory()
    time_step = 0
    
    # Plateau Tracking Variables
    rewards_history = []
    best_moving_avg = -float('inf')
    plateau_counter = 0
    WINDOW_SIZE = 500
    PATIENCE = 1000
    IMPROVEMENT_THRESHOLD = 1e-3
    early_exit_start = 10000
    pbar = tqdm(range(1, MAX_EPISODES + 1), desc=f"Training {agent_name}")
    for i_episode in pbar:
        current_lr = apply_linear_decay(agent, i_episode, MAX_EPISODES, LR_ACTOR, LR_CRITIC)
        state = env.reset()
        current_ep_reward = 0
        last_info = {}
        
        for t in range(MAX_STEPS):
            time_step += 1
            if agent_name == "SPRL":
                action, latent, log_prob = agent.select_action(state)
                memory.actions.append(torch.tensor(latent))
            else:
                action, log_prob = agent.select_action(state)
                memory.actions.append(torch.tensor(action))
            
            memory.states.append(torch.tensor(state))
            memory.logprobs.append(torch.tensor(log_prob))
            
            state, reward, done, info = env.step(action)
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            current_ep_reward += reward
            last_info = info
            
            if time_step % UPDATE_TIMESTEP == 0:
                agent.learn(memory)
                memory.clear()
                time_step = 0
            
            if done: break
            
        logger.log_training_episode(agent_name, current_ep_reward)
        rewards_history.append(current_ep_reward)

        # --- Early Exit Check ---
        
        if len(rewards_history) >= early_exit_start:
            # Calculate current 500-episode moving average
            current_moving_avg = np.mean(rewards_history[-WINDOW_SIZE:])
            
            # Check for improvement relative to the best moving average
            if current_moving_avg > (best_moving_avg * (1 + IMPROVEMENT_THRESHOLD)):
                best_moving_avg = current_moving_avg
                plateau_counter = 0  # Reset patience if we improved
            else:
                plateau_counter += 1 # Increment if no significant gain

            # Check if we've reached the plateau limit
            if plateau_counter >= PATIENCE:
                print(f"\n[Early Exit] Plateau reached. No improvement > {IMPROVEMENT_THRESHOLD} in {PATIENCE} episodes.")
                break
        
        if i_episode % 5 == 0:
            # Extract specific components for clarity
            prod_rwd = last_info['reward']['production']
            saf_pnlty = last_info['penalties']['safety']
            smth_pnlty = last_info['penalties']['smoothing']
            bio_pnlty = last_info['penalties']['biomass_efficiency']
            
            pbar.set_postfix({
                "Prod": f"{prod_rwd:.3f}",
                "Saf": f"{saf_pnlty:.3f}",
                "Smth": f"{smth_pnlty:.3f}",
                "Bio": f"{bio_pnlty:.3f}",
                "Plat": f"{plateau_counter}/{PATIENCE}"
            })

    # --- SAVE WEIGHTS AFTER TRAINING OR EARLY EXIT ---
    save_path = f"{agent_name}_final_weights.pth"
    torch.save(agent.policy.state_dict(), save_path)
    print(f"Successfully saved weights to {save_path}")

def evaluate_agent(agent_name, agent, logger, eval_episodes=100):
    """Loads weights from .pth file and runs deterministic evaluation."""
    print(f"\n--- Evaluating: {agent_name} ---")
    
    # --- LOAD WEIGHTS BEFORE EVALUATION ---
    load_path = f"{agent_name}_final_weights.pth"
    if os.path.exists(load_path):
        agent.policy.load_state_dict(torch.load(load_path))
        agent.policy.eval()
        print(f"Loaded trained weights from {load_path}")
    else:
        print(f"Warning: No weights found at {load_path}. Evaluating untrained model.")

    env = PhotoProductionEnv(train_mode=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_states, all_actions, all_rewards, all_infos = [], [], [], []
    
    for _ in tqdm(range(eval_episodes), desc=f"Eval {agent_name}"):
        state = env.reset()
        ep_states, ep_actions, ep_rewards, ep_infos = [], [], [], []
        
        for t in range(MAX_STEPS):
            with torch.no_grad():
                state_t = torch.FloatTensor(state).to(device)
                if agent_name == "SPRL":
                    z = agent.policy.actor_latent_mean(agent.policy.actor_base(state_t))
                    s_norm = (state_t.unsqueeze(0) - agent.s_mean) / (agent.s_std + 1e-8)
                    z_norm = (z.unsqueeze(0) - agent.a_mean) / (agent.a_std + 1e-8)
                    action_norm = agent.safeguard(s_norm, z_norm)
                    action = (action_norm * (agent.a_std + 1e-8)) + agent.a_mean
                    action = action.cpu().numpy().flatten()
                else:
                    action = agent.policy.actor(state_t).cpu().numpy().flatten()
            
            next_state, reward, done, info = env.step(action)
            ep_states.append(state)
            ep_actions.append(action)
            ep_rewards.append(reward)
            ep_infos.append(info)
            state = next_state
            if done: break
            
        all_states, all_actions, all_rewards, all_infos = ep_states, ep_actions, ep_rewards, ep_infos

    logger.log_evaluation_trajectory(agent_name, all_states, all_actions, all_rewards, all_infos)

if __name__ == "__main__":
    logger = DataLogger()
    
    lag_agent = NonResNet_Agent(STATE_DIM, ACTION_DIM, LR_ACTOR, LR_CRITIC, GAMMA, K_EPOCHS, EPS_CLIP)
    sprl_agent = SPRL_Agent(STATE_DIM, ACTION_DIM, LR_ACTOR, LR_CRITIC, GAMMA, K_EPOCHS, EPS_CLIP, ENTROPY_COEFF, LATENT_DIM)
    
    train_agent("SPRL", sprl_agent, logger)
    train_agent("NonResNet", lag_agent, logger)
    
    evaluate_agent("NonResNet", lag_agent, logger)
    evaluate_agent("SPRL", sprl_agent, logger)
    
    Plotter.plot_training_results(logger.training_log)
    Plotter.plot_evaluation_trajectories(logger.eval_data)