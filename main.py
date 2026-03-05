import torch
import numpy as np
import os
from tqdm import tqdm
from env import PhotoProductionEnv
from lag_agent import NonResNet_Agent
from res_net_agent import SPRL_Agent
from utils import DataLogger, Plotter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

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
MIN_LR = 1e-5 
ENTROPY_COEFF = 0.01

class Memory:
    def __init__(self):
        self.states, self.actions, self.logprobs, self.rewards, self.is_terminals = [], [], [], [], []
    def clear(self):
        del self.states[:], self.actions[:], self.logprobs[:], self.rewards[:], self.is_terminals[:]

def train_agent(agent_name, agent, logger):
    print(f"\n--- Starting Training: {agent_name} ---")
    env = PhotoProductionEnv(train_mode=True)
    memory = Memory()
    scheduler = CosineAnnealingWarmRestarts(
        agent.optimizer, 
        T_0=5000, 
        T_mult=2, 
        eta_min=MIN_LR
    )
    time_step = 0
    
    rewards_history = []
    best_moving_avg = -float('inf')
    plateau_counter = 0
    
    # --- EARLY EXIT HYPERPARAMETERS ---
    WINDOW_SIZE = 500            # Episodes to average for smoothing
    PATIENCE = 1000             # Max episodes to wait for improvement
    IMPROVEMENT_THRESHOLD = 1e-3 # Minimum change to count as improvement
    EARLY_EXIT_START = 10000    # Don't check for plateaus before this episode
    
    pbar = tqdm(range(1, MAX_EPISODES + 1), desc=f"Training {agent_name}")
    for i_episode in pbar:
        state = env.reset()
        
        current_ep_reward = 0
        ep_prod, ep_saf, ep_smth, ep_bio = 0, 0, 0, 0
        steps_taken = 0
        
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
            
            current_ep_reward += reward
            ep_prod += info['reward']['production']
            ep_saf += info['penalties']['safety']
            ep_smth += info['penalties']['smoothing']
            ep_bio += info['penalties']['biomass_efficiency']
            steps_taken += 1
            
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
        # --- PLATEAU LOGIC IMPLEMENTATION ---
        if i_episode >= EARLY_EXIT_START:
            # Calculate moving average of the last WINDOW_SIZE rewards
            recent_avg = np.mean(rewards_history[-WINDOW_SIZE:])
            
            # Check if recent performance is significantly better than the best recorded
            if recent_avg > (best_moving_avg * (1 + IMPROVEMENT_THRESHOLD)):
                best_moving_avg = recent_avg
                plateau_counter = 0 # Reset patience
            else:
                plateau_counter += 1
            
            # Trigger Early Exit
            if plateau_counter >= PATIENCE:
                print(f"\n[Early Exit] {agent_name} reached a plateau at episode {i_episode}.")
                break

        if i_episode % 5 == 0:
            pbar.set_postfix({
                "Total": f"{current_ep_reward:.2f}",
                "Avg_Prod": f"{ep_prod/steps_taken:.3f}",
                "Avg_Saf": f"{ep_saf/steps_taken:.3f}",
                "Avg_Smth": f"{ep_smth/steps_taken:.3f}",
                "Avg_Bio": f"{ep_bio/steps_taken:.3f}",
                "Plat": f"{plateau_counter}/{PATIENCE}",
                "LR": f"{agent.optimizer.param_groups[0]['lr']:.3e}"
            })

    # Save weights after training completes or early exit triggers
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
    
    # Initialize pbar similar to training
    pbar = tqdm(range(eval_episodes), desc=f"Eval {agent_name}")
    
    for _ in pbar:
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
        
        # --- Update pbar with metrics from the completed episode ---
        last_info = ep_infos[-1]
        total_ep_reward = sum(ep_rewards)
        
        pbar.set_postfix({
            "Total": f"{total_ep_reward:.3f}",
            "Prod": f"{last_info['reward']['production']:.3f}",
            "Saf": f"{last_info['penalties']['safety']:.3f}",
            "Smth": f"{last_info['penalties']['smoothing']:.3f}",
            "Bio": f"{last_info['penalties']['biomass_efficiency']:.3f}"
        })
            
        all_states, all_actions, all_rewards, all_infos = ep_states, ep_actions, ep_rewards, ep_infos

    logger.log_evaluation_trajectory(agent_name, all_states, all_actions, all_rewards, all_infos)

if __name__ == "__main__":
    logger = DataLogger()
    
    lag_agent = NonResNet_Agent(STATE_DIM, ACTION_DIM, LR_ACTOR, LR_CRITIC, GAMMA, K_EPOCHS, EPS_CLIP)
    sprl_agent = SPRL_Agent(STATE_DIM, ACTION_DIM, LR_ACTOR, LR_CRITIC, GAMMA, K_EPOCHS, EPS_CLIP, ENTROPY_COEFF, LATENT_DIM)
    train_active = False
    if train_active:
        train_agent("NonResNet", lag_agent, logger)
    train_agent("SPRL", sprl_agent, logger)
        
    
    evaluate_agent("NonResNet", lag_agent, logger)
    evaluate_agent("SPRL", sprl_agent, logger)
    
    Plotter.plot_training_results(logger.training_log)
    Plotter.plot_evaluation_trajectories(logger.eval_data)