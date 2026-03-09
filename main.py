import torch
import numpy as np
import os
from tqdm import tqdm
from env import PhycocyaninEnv  # 1. Updated Class Name
from lag_agent import NonResNet_Agent
from res_net_agent import SPRL_Agent
from utils import DataLogger, Plotter
from torch.optim.lr_scheduler import LinearLR

# --- Hyperparameters Updated ---
STATE_DIM = 4      # 2. Updated to 4 (cx, cN, cq, t_norm)
ACTION_DIM = 2
MAX_EPISODES = 50000 
UPDATE_TIMESTEP = 1000 # Adjusted for stability 
K_EPOCHS = 40
EPS_CLIP = 0.2
GAMMA = 0.99
LR_ACTOR = 3e-4
LR_CRITIC = 1e-3
MIN_LR = 1e-5 
ENTROPY_COEFF = 0.05
EVALUATE_ONLY = True  # Set to False to run the 100k episode training loop
NOISE_STD = 0.05

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
    scheduler = LinearLR(agent.optimizer, start_factor=1.0, end_factor=0.01, total_iters=MAX_EPISODES)
    
    time_step = 0
    rewards_history = []
    best_moving_avg = -float('inf')
    plateau_counter = 0
    
    # Early Exit Params
    WINDOW_SIZE, PATIENCE, EARLY_EXIT_START = 150, 1500, 15000    
    
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
            
            memory.states.append(torch.tensor(state, dtype=torch.float32))
            memory.actions.append(torch.tensor(action, dtype=torch.float32))
            memory.raw_actions.append(torch.tensor(raw_act, dtype=torch.float32))
            memory.logprobs.append(torch.tensor(log_prob, dtype=torch.float32))
            
            state, reward, done, info = env.step(action)
            
            current_ep_reward += reward
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            if time_step % UPDATE_TIMESTEP == 0:
                agent.learn(memory)                
                memory.clear()
                time_step = 0
            if done: break
        scheduler.step()
        logger.log_training_episode(agent_name, current_ep_reward, info["violation_count"])
        rewards_history.append(current_ep_reward)
        
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
    print(f"\n--- Evaluating: {agent_name} with Randomized Initial State + N(0, {noise_std}) Intent Noise ---")
    load_path = f"{agent_name}_final_weights.pth"
    if os.path.exists(load_path):
        agent.policy.load_state_dict(torch.load(load_path))
        agent.policy.eval()

    env = PhycocyaninEnv()
    all_states, all_actions, all_rewards, all_infos = [], [], [], []
    total_g1, total_g2, total_g3 = 0, 0, 0
    
    for _ in range(eval_episodes):
        # 1. Base reset with standard normal variance
        state = env.reset(randomize=True)
        
        ep_states, ep_actions, ep_rewards, ep_infos = [], [], [], []
        
        while True:
            # Deterministic Intent Generation
            with torch.no_grad():
                state_t = torch.FloatTensor(state).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).unsqueeze(0)
                intent = torch.tanh(agent.policy.actor(state_t)).cpu().numpy().flatten()
            
            # Apply severe Gaussian variation (e.g. 20%) to the INTENT, simulating 
            # actor confusion or controller imprecision
            if noise_std > 0:
                noise = np.random.normal(0, noise_std, size=intent.shape)
                noisy_intent = np.clip(intent + noise, -1.0, 1.0)
            else:
                noisy_intent = intent
                
            # SPRL Safeguard Intercepts the Noisy Intent
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
        
        # Store last trajectory for Plotter
        all_states, all_actions, all_rewards, all_infos = ep_states, ep_actions, ep_rewards, ep_infos
        
        # Log episode violations
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

    for i in all_infos: i["is_safe"] = 1 if i["violation_count"] == 0 else 0
    
    logger.log_evaluation_trajectory(agent_name, all_states, all_actions, all_rewards, all_infos)
    Plotter.plot_evaluation_trajectories(logger.eval_data, agent_name=agent_name)

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