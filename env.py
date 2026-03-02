import numpy as np
import gymnasium as gym
from pcgym import make_env

class SimplePID:
    """
    Native PID implementation to provide nominal 'driver' intent 
    without external dependencies.
    """
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, setpoint, measurement, dt=1.0):
        error = setpoint - measurement
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

class PhotoproductionEnv(gym.Wrapper):
    def __init__(self, render_mode=None):
        # 1. Initialize pc-gym Photoproduction
        # States: [cx (Biomass), cN (Nitrate), cq (Product)]
        # Actions: [I_in (Light), Fn (Nitrate Feed)]
        env = make_env('photoproduction', render_mode=render_mode)
        super().__init__(env)
        
        # 2. PID Controllers for the 'Nominal Intent'
        # Light PID tracks Biomass; Nitrate PID tracks Nitrate level
        self.pid_light = SimplePID(kp=1.2, ki=0.1, kd=0.01)
        self.pid_nitrate = SimplePID(kp=0.5, ki=0.05, kd=0.01)

        # 3. Research Constraint Thresholds
        self.I_crit = 450.0  # Critical average light intensity
        self.cx_min = 0.05   # Washout threshold (Biomass)
        
        self.current_action = np.zeros(self.action_space.shape[1])

    def get_nominal_control(self, obs, setpoints=[2.0, 50.0]):
        """
        Calculates the 'Intent' of a standard controller (the nominal action).
        setpoints: [target_biomass (cx), target_nitrate (cN)]
        """
        cx, cN, _ = obs
        target_cx, target_cN = setpoints
        
        # Calculate PID outputs
        i_nom = self.pid_light.compute(target_cx, cx)
        fn_nom = self.pid_nitrate.compute(target_cN, cN)
        
        # Clip to the environment's physical action space limits
        nom_action = np.array([i_nom, fn_nom])
        return np.clip(nom_action, self.action_space.low, self.action_space.high)

    def compute_reward(self, state, action, prev_action, info):
        """
        R = Production + Survival - Violation_Penalty - Unsmooth_Penalty
        """
        cx, _, cq = state
        
        # 1. Production Reward (Phycocyanin yield)
        reward = cq * 0.5 
        
        # 2. Survival Bonus/Penalty (Washout protection)
        if cx > self.cx_min:
            reward += 2.0
        else:
            reward -= 50.0 # Heavy penalty for biomass washout
            
        # 3. Constraint Violation Penalty (Photo-inhibition)
        # We use I_av (Average Light) from pc-gym info
        I_av = info.get('I_av', 0)
        if I_av > self.I_crit:
            reward -= 10.0 * (I_av - self.I_crit)
            
        # 4. Control Smoothness Penalty (Action chattering)
        smoothness = np.sum(np.square(action - prev_action))
        reward -= 0.1 * smoothness
        
        return reward

    def step(self, action):
        """
        Standard Gymnasium step augmented with research reward logic.
        """
        prev_action = self.current_action
        self.current_action = action
        
        # Execute action in the underlying pc-gym environment
        obs, _, terminated, truncated, info = self.env.step(action)
        
        # Calculate our custom research reward
        reward = self.compute_reward(obs, action, prev_action, info)
        
        # Explicitly flag violations for the logger
        info['is_violated'] = info.get('I_av', 0) > self.I_crit or obs[0] < self.cx_min
        
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Resets the environment and the action history."""
        obs, info = self.env.reset(seed=seed)
        self.current_action = np.zeros(self.action_space.shape[1])
        return obs, info

def create_env():
    return PhotoproductionEnv()