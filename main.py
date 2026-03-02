import os
import torch
import numpy as np
import pandas as pd
from env import PhycocyaninSafeEnv
from lag_agent import NonResNet_Agent
from res_net_agent import SPRL_Agent
from mpc import evaluate_mpc
from utils import Logger, plot_training_results, plot_evaluation_comparison, plot_temporal_dynamics

# Fix for OpenMP runtime conflict 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def train_agent(use_resnet=False, total_episodes=50000, max_steps=7200):
    """
    Executes a 50,000 episode training run for a specific agent type.
    """
    experiment_name = "ResNet_Guided_PPO" if use_resnet else "Baseline_Lagrangian_PPO"
    print(f"\n🚀 Starting Training: {experiment_name}")
    
    env = PhycocyaninSafeEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Hyperparameters based on project architecture [cite: 14, 15]
    params = {
        'state_dim': state_dim,
        'action_dim': action_dim,
        'lr_actor': 3e-4,
        'lr_critic': 1e-3,
        'gamma': 0.99,
        'K_epochs': 40,
        'eps_clip': 0.2
    }

    if use_resnet:
        agent = SPRL_Agent(**params, entropy_coeff=0.01, latent_dim=action_dim)
    else:
        agent = NonResNet_Agent(**params)

    logger = Logger(experiment_name)
    
    class Memory:
        def __init__(self):
            self.actions, self.states, self.logprobs, self.rewards, self.is_terminals = [], [], [], [], []
        def clear(self):
            del self.actions[:], self.states[:], self.logprobs[:], self.rewards[:], self.is_terminals[:]

    memory = Memory()
    
    for episode in range(1, total_episodes + 1):
        state = env.reset()
        episode_reward = 0
        violated = False
        
        for t in range(max_steps):
            # Select action (SP-RL handles internal projection) [cite: 15]
            action, latent_or_nom, logprob = agent.select_action(state)
            
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            memory.states.append(torch.FloatTensor(state))
            memory.actions.append(torch.FloatTensor(latent_or_nom))
            memory.logprobs.append(torch.FloatTensor(logprob))
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            state = next_state
            episode_reward += reward
            if info.get('constraint_violated', False):
                violated = True
            
            if done: break
            
        # Update PPO policy
        agent.learn(memory)
        memory.clear()
        
        logger.log_episode(episode_reward, violated)
        
        if episode % 100 == 0:
            avg_r = logger.avg_rewards[-1]
            v_rate = logger.violation_rates[-1]
            print(f"Ep {episode} | Avg Reward: {avg_r:.2f} | Violation Rate: {v_rate:.1f}%")

    # Save final model 
    torch.save(agent.policy.state_dict(), f"{experiment_name}_final.pth")
    logger.save_data()
    return agent, logger

def run_evaluation(agent, name, episodes=100, max_steps=7200):
    """
    Deterministic evaluation phase.
    """
    print(f"🧪 Evaluating {name}...")
    env = PhycocyaninSafeEnv()
    total_rewards = []
    total_violations = 0
    trajectory = []

    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0
        for t in range(max_steps):
            # Deterministic selection (exploitation)
            action, _, _ = agent.select_action(state)
            state, reward, done, info = env.step(action)
            ep_reward += reward
            if info.get('constraint_violated', False):
                total_violations += 1
            if ep == 0: trajectory.append(state[2]) # Track cq [cite: 26]
            if done: break
        total_rewards.append(ep_reward)

    return {
        'reward': np.mean(total_rewards),
        'violations': total_violations / episodes,
        'trajectory': trajectory
    }

if __name__ == "__main__":
    # 1. Train both agents 
    lag_agent, lag_logger = train_agent(use_resnet=False)
    res_agent, res_logger = train_agent(use_resnet=True)

    # 2. Benchmark against MPC Oracle [cite: 17, 26]
    print("\n--- Final Benchmarking ---")
    mpc_results = evaluate_mpc() # From mpc.py
    
    lag_eval = run_evaluation(lag_agent, "Baseline Lagrangian")
    res_eval = run_evaluation(res_agent, "ResNet Guided")

    # 3. Visualization and Metrics [cite: 19]
    # Learning Curves (Requirement 1)
    plot_training_results([lag_logger, res_logger])

    # Comparative Histograms (Requirement 2)
    eval_metrics = {
        'Baseline PPO': {'reward': lag_eval['reward'], 'violations': lag_eval['violations']},
        'ResNet PPO': {'reward': res_eval['reward'], 'violations': res_eval['violations']},
        'MPC Oracle': {'reward': mpc_results['reward'], 'violations': mpc_results['violations_per_ep']}
    }
    plot_evaluation_comparison(eval_metrics)

    # Temporal Dynamics (Requirement 3)
    trajectories = {
        'Baseline PPO': lag_eval['trajectory'],
        'ResNet PPO': res_eval['trajectory']
    }
    plot_temporal_dynamics(trajectories)
    
    print("\n✅ Benchmarking complete. All plots saved.")