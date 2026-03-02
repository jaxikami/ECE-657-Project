import numpy as np
import torch
from env import PhycocyaninSafeEnv
import pandas as pd
import os

class MPCOracle:
    """
    A simplified MPC Oracle for the photoproduction process.
    Provides a perfect safety baseline by utilizing the 
    known physical constraints of the system.
    """
    def __init__(self):
        # Physical Constants
        self.I_crit = 450.0
        self.alpha = 0.25
        self.L = 0.5
        self.cN_threshold = 180.0
        self.Fn_max = 20.0

    def _get_i_limit(self, cx):
        """Calculates the dynamic photoinhibition threshold."""
        shading_factor = np.exp(-self.alpha * cx * self.L)
        return self.I_crit / (shading_factor + 1e-6)

    def select_action(self, state):
        """
        Greedy Optimal Control:
        1. Maximize Light (I) up to the photoinhibition manifold.
        2. Binary Feed (Fn): Stop if nitrate > threshold.
        """
        cx, cN, cq = state
        
        # 1. Calculate max safe light intensity
        i_safe = self._get_i_limit(cx)
        
        # 2. Control feed rate based on nitrate toxicity limit
        fn_safe = 0.0 if cN >= self.cN_threshold else self.Fn_max
        
        return np.array([i_safe, fn_safe], dtype=np.float32)

def evaluate_mpc(episodes=100, max_steps=7200):
    """Runs 100 evaluation episodes to set the benchmark."""
    print(f"🧪 Evaluating MPC Oracle baseline...")
    env = PhycocyaninSafeEnv()
    oracle = MPCOracle()
    
    total_rewards = []
    total_violations = 0
    trajectories = [] # To store cq (production) over time

    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0
        ep_trajectory = []
        
        for t in range(max_steps):
            # The oracle picks the 'perfect' action
            action = oracle.select_action(state)
            
            # Simulate sensor/actuator noise (0.05 std as per main.py)
            noise = np.random.normal(0, 0.05, size=action.shape)
            noisy_action = action + noise
            
            state, reward, done, info = env.step(noisy_action)
            
            ep_reward += reward
            ep_trajectory.append(float(state[2])) # Log Phycocyanin concentration
            
            if info.get('constraint_violated', False):
                total_violations += 1
            
            if done:
                break
        
        total_rewards.append(ep_reward)
        if ep == 0: 
            trajectories = ep_trajectory

    avg_reward = np.mean(total_rewards)
    # Violation Rate % (total violations / total possible steps)
    violation_rate = (total_violations / (episodes * max_steps)) * 100
    
    print("-" * 35)
    print(f"MPC Oracle Benchmark Results:")
    print(f"Avg Reward:      {avg_reward:.2f}")
    print(f"Violation Rate:  {violation_rate:.4f}%")
    print("-" * 35)
    
    return {
        'reward': avg_reward,
        'violations_per_ep': total_violations / episodes,
        'violation_rate_percent': violation_rate,
        'trajectory': trajectories
    }

if __name__ == "__main__":
    results = evaluate_mpc()
    
    # Save results to be loaded by plot_temporal_dynamics in utils.py
    df = pd.DataFrame({
        'reward': [results['reward']],
        'violations': [results['violations_per_ep']],
        'violation_rate': [results['violation_rate_percent']]
    })
    df.to_csv("mpc_results.csv", index=False)
    
    # Save the trajectory for the temporal dynamics plot 
    np.save("mpc_trajectory.npy", np.array(results['trajectory']))