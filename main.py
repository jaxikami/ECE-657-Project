import os
# 1. FIX OPENMP RUNTIME ERROR: Prevents the "Error #15" initialization crash 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
from env import create_env
from agent import PPOAgent, PPOBuffer
from pretrain import SmartSteeringWheel
from utils import Logger, plot_training_results

def run_experiment(use_resnet=False, total_episodes=50000, max_steps=7200):
    """
    Runs a full training experiment. 
    If use_resnet=True, the agent's actions are projected by the Smart Steering Wheel.
    """
    experiment_name = "ResNet_Guided_PPO" if use_resnet else "Baseline_Lagrangian_PPO"
    print(f"\n🚀 Starting Experiment: {experiment_name}")
    
    env = create_env()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[1] 
    
    # 1. Initialize ResNet if required
    resnet = None
    if use_resnet:
        resnet = SmartSteeringWheel(state_dim=state_dim, action_dim=action_dim).to('cuda' if torch.cuda.is_available() else 'cpu')
        # Load the pretrained weights from pretrain.py
        try:
            resnet.load_state_dict(torch.load("smart_steering_wheel.pth", weights_only=True))
            resnet.eval()
            print("✅ Pretrained ResNet loaded successfully.")
        except FileNotFoundError:
            print("❌ Error: smart_steering_wheel.pth not found. Run pretrain.py first.")
            return None

    # 2. Initialize Agent and Buffer
    agent = PPOAgent(state_dim, action_dim, resnet_model=resnet)
    buffer = PPOBuffer()
    logger = Logger(experiment_name)

    # 3. Training Loop
    for episode in range(1, total_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        
        for t in range(max_steps):
            # Agent outputs nominal action; internal logic handles ResNet projection
            nom_act, safe_act, logprob, val = agent.select_action(state)
            
            # Step environment with the SAFE action
            next_state, reward, done, trunc, info = env.step(safe_act)
            
            # Store data (PPO learns mapping from state to the INTENDED action)
            buffer.states.append(state)
            buffer.actions.append(nom_act)
            buffer.logprobs.append(logprob)
            buffer.rewards.append(reward)
            buffer.is_terminals.append(done or trunc)
            
            state = next_state
            episode_reward += reward
            
            # Update PPO every 2000 steps
            if (t + 1) % 2000 == 0:
                agent.update(buffer)
                buffer.clear()

            if done or trunc:
                break
        
        logger.log_episode(episode_reward, info.get('is_violated', False))
        
        if episode % 100 == 0:
            logger.print_status(episode, total_episodes)
            
        # Periodic Checkpointing
        if episode % 5000 == 0:
            torch.save(agent.policy.state_dict(), f"weights/{experiment_name}_ep{episode}.pth")

    # Save final model
    save_path = f"{experiment_name}_final.pth"
    torch.save(agent.policy.state_dict(), save_path)
    print(f"✅ Training complete. Saved to {save_path}")
    return logger

def evaluate_agent(model_path, use_resnet=False, episodes=100, noise_std=0.05):
    """
    Evaluation phase: 100 episodes with Gaussian noise and NO exploration.
    """
    print(f"\n🧪 Evaluating {model_path}...")
    env = create_env()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[1]
    
    resnet = None
    if use_resnet:
        resnet = SmartSteeringWheel(state_dim, action_dim).to('cuda' if torch.cuda.is_available() else 'cpu')
        resnet.load_state_dict(torch.load("smart_steering_wheel.pth", weights_only=True))
        resnet.eval()

    agent = PPOAgent(state_dim, action_dim, resnet_model=resnet)
    agent.policy.load_state_dict(torch.load(model_path, weights_only=True))
    
    violations = 0
    total_rewards = []

    for _ in range(episodes):
        state, _ = env.reset()
        ep_reward = 0
        for _ in range(7200):
            # evaluate=True uses the mean (exploitation)
            _, safe_act, _, _ = agent.select_action(state, evaluate=True)
            
            # Add Gaussian noise to simulate sensor/actuator uncertainty
            noise = np.random.normal(0, noise_std, size=safe_act.shape)
            noisy_action = safe_act + noise
            
            state, reward, done, trunc, info = env.step(noisy_action)
            ep_reward += reward
            if info.get('is_violated', False):
                violations += 1
            if done or trunc: break
        total_rewards.append(ep_reward)
    
    avg_reward = np.mean(total_rewards)
    violation_rate = (violations / (episodes * 7200)) * 100
    return avg_reward, violation_rate

if __name__ == "__main__":
    if not os.path.exists("weights"): os.makedirs("weights")
    
    # 1. Run Baseline (Lagrangian/Penalty only)
    baseline_logger = run_experiment(use_resnet=False)
    
    # 2. Run Proposed (With ResNet Steering)
    resnet_logger = run_experiment(use_resnet=True)
    
    # 3. Final Evaluation Comparison
    b_rew, b_viol = evaluate_agent("Baseline_Lagrangian_PPO_final.pth", use_resnet=False)
    r_rew, r_viol = evaluate_agent("ResNet_Guided_PPO_final.pth", use_resnet=True)
    
    print("\n" + "="*40)
    print("      FINAL RESEARCH RESULTS")
    print("="*40)
    print(f"Baseline PPO | Reward: {b_rew:8.2f} | Violations: {b_viol:.4f}%")
    print(f"ResNet PPO   | Reward: {r_rew:8.2f} | Violations: {r_viol:.4f}%")
    print("="*40)
    
    # Generate Comparison Plots
    plot_training_results(baseline_logger, resnet_logger)