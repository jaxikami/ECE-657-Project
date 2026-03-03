import numpy as np
import torch
from pcgym import make_env

class PhycocyaninSafeEnv:
    def __init__(self):
        # Initialize the core pc-gym environment
        self.env = make_env("photoproduction")
        
        # Physical Constants
        self.I_crit = 450.0
        self.alpha = 0.25
        self.L = 0.5
        
        # Normalization Constants for scaling violations to [0, 1]
        self.max_cN_violation = 20.0 # (200 limit - 180 threshold)
        self.max_I_violation = 3000.0 # Max possible light intensity
        
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def _get_i_limit(self, cx):
        """Calculates the dynamic photoinhibition threshold"""
        shading_factor = np.exp(-self.alpha * cx * self.L)
        return self.I_crit / (shading_factor + 1e-6)

    def step(self, action):
        """
        Executes step with a scaled Lagrangian penalty (0-1 range * 10)
        and early stopping.
        """
        i_nom, fn_nom = action
        
        # 1. Execute step in pc-gym simulation
        next_state, reward, terminated, truncated, info = self.env.step(action)
        cx, cN, cq = next_state
        production_reward = cq * 0.1
        # 2. Calculate Photoinhibition Limit
        i_limit = self._get_i_limit(cx)
        
        # 3. Calculate Raw Violations
        v_n = max(0, cN - 180.0) 
        v_i = max(0, i_nom - i_limit)
        
        # 4. Scale violations to [0, 1]
        # We divide by the max expected violation to keep the penalty proportional
        scaled_v_n = min(1.0, v_n / self.max_cN_violation)
        scaled_v_i = min(1.0, v_i / self.max_I_violation)
        
        # Combined scaled violation (0 to 1 range)
        total_scaled_violation = (scaled_v_n + scaled_v_i) / 2.0
        
        # 5. Apply Lagrangian Penalty (Scaled 0-1 then * 10)
        # This creates a penalty of up to -10.0
        lagrangian_penalty = total_scaled_violation * 10.0
        total_reward = reward - lagrangian_penalty + production_reward
        
        # 6. Early Stopping: End run immediately if any constraint is hit
        violated = (v_n > 0 or v_i > 0)
        done = terminated or truncated or violated
        
        info.update({
            'constraint_violated': violated,
            'lagrangian_penalty': lagrangian_penalty,
            'is_early_stop': violated
        })
        
        return np.array(next_state, dtype=np.float32), total_reward, done, info

    def reset(self, seed=None, options=None):
        state, info = self.env.reset(seed=seed, options=options)
        return np.array(state, dtype=np.float32)