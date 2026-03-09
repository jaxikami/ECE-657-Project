import numpy as np
from numba import njit

# --- KINETIC ENGINE (Arthrospira platensis) ---
@njit
def calculate_rates_numba(C, I, Fn):
    """
    Kinetics based on Table 1.
    All rates are in per-hour units.
    """
    # Indices: 0: Biomass (Cx), 1: Nitrate (CN), 2: Phycocyanin (Cq)
    i_X, i_N, i_q = 0, 1, 2
    
    # Parameters 
    um = 0.0572    # h^-1
    ud = 0.0       # h^-1 (Death rate)
    KN = 393.1     # mg/L
    YNX = 504.5    # mg/g
    km = 0.00016   # mg/g/h
    kd = 0.281     # h^-1
    ks = 178.9     # umol/m^2/s
    ki = 447.1     # umol/m^2/s
    ksq = 23.51    # umol/m^2/s
    kiq = 800.0    # umol/m^2/s
    KNp = 16.89    # mg/L

    # 1. Growth photolimitation (Aiba model) [cite: 647, 649]
    phi_I = I / (I + ks + (I**2 / ki))
    phi_N = C[i_N] / (C[i_N] + KN)
    
    # 2. Product photolimitation 
    phi_Iq = I / (I + ksq + (I**2 / kiq))
    
    R = np.zeros(3)
    # dCx/dt [cite: 649]
    R[i_X] = um * phi_I * C[i_X] * phi_N - ud * C[i_X]
    # dCN/dt [cite: 649]
    R[i_N] = -YNX * (um * phi_I * C[i_X] * phi_N) + Fn
    # dCq/dt 
    R[i_q] = km * phi_Iq * C[i_X] - (kd * C[i_q]) / (C[i_N] + KNp)
    
    return R

@njit
def integrate_rk4(c_init, I, Fn, dt, n_steps):
    c = c_init.copy()
    for _ in range(n_steps):
        # RK4 integration for stability 
        k1 = calculate_rates_numba(c, I, Fn)
        k2 = calculate_rates_numba(c + 0.5 * dt * k1, I, Fn)
        k3 = calculate_rates_numba(c + 0.5 * dt * k2, I, Fn)
        k4 = calculate_rates_numba(c + dt * k3, I, Fn)
        
        c += (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        c = np.maximum(c, 0.0) 
    return c

class PhycocyaninEnv:
    def __init__(self):
        # --- Batch Config  ---
        self.total_time = 240.0        
        self.control_freq = 20.0       
        self.max_steps = int(self.total_time / self.control_freq) # 12 steps of 20h
        
        # --- Limits [cite: 679, 687, 691, 694, 695] ---
        self.I_MIN, self.I_MAX = 120.0, 400.0 
        self.FN_MAX = 40.0                    
        self.N_LIMIT_PATH = 800.0      # g1
        self.RATIO_LIMIT = 0.011       # g2
        self.N_LIMIT_TERM = 150.0      # g3
        
        # --- Integration Config ---
        self.dt = 10.0 / 60.0          # 10 minutes (0.1667 h)
        self.n_inner_steps = int(self.control_freq / self.dt) # 120 steps for 20h

        self.reset()

    def reset(self, randomize=False):
        self.time = 0.0
        self.time_step_count = 0
        self.violation_count = 0
        self.g1_violation_count = 0
        self.g2_violation_count = 0
        self.g3_violation_count = 0
        
        # Initial State Base: [1.0 g/L, 150 mg/L, 0.0 mg/L]
        self.state = np.array([1.0, 150.0, 0.0], dtype=np.float64)
        
        # Add Initial State Randomization for Robustness Evaluation
        if randomize:
            # Vary Biomass between 0.8 and 1.2 g/L
            self.state[0] = np.random.uniform(0.8, 1.2)
            # Vary initial Nitrate between 100.0 and 200.0 mg/L
            self.state[1] = np.random.uniform(100.0, 200.0)
            
        self.prev_action = np.zeros(2)
        
        # Metric Tracking
        self.ep_total_reward = 0.0
        self.ep_rewards = []
        self.ep_prod_rewards = []
        self.ep_smooth_penalties = []
        self.ep_n_usage_penalties = []
        self.ep_violation_penalties = []
        self.ep_g1_penalties = []
        self.ep_g2_penalties = []
        self.ep_g3_penalties = []
        
        return self.get_state_norm()

    def get_state_norm(self):
        norm_obs = np.zeros(4, dtype=np.float64)
        norm_obs[0] = self.state[0] / 6.0     
        norm_obs[1] = self.state[1] / 800.0   
        norm_obs[2] = self.state[2] / 0.2     
        norm_obs[3] = self.time / self.total_time 
        return norm_obs

    def step(self, action):
        a_clipped = np.clip(action, -1.0, 1.0)
        I_phys = self.I_MIN + ((a_clipped[0] + 1.0) / 2.0) * (self.I_MAX - self.I_MIN)
        Fn_phys = ((a_clipped[1] + 1.0) / 2.0) * self.FN_MAX
        
        # Internal integration every 10 minutes for stability
        self.state = integrate_rk4(self.state, I_phys, Fn_phys, self.dt, self.n_inner_steps)
        self.time += self.control_freq
        self.time_step_count += 1
        minor_coef = 0 if getattr(self, 'is_training', False) else 2
        severe_coef = 200
        # --- Penalties and Rewards ---
        # g1: Path Nitrate violation penalty
        n_vio_p = 0.0
        n_ratio = self.state[1] / self.N_LIMIT_PATH
        # Reverse log barrier approaching infinity as n_ratio -> 1, centered near 0.95
        if n_ratio > 0.95 and n_ratio < 1.0:
            # Shift ratio such that evaluating at 0.95 gives roughly log(1.0) = 0
            n_vio_p -= minor_coef * np.log(1.05 - n_ratio) 
        if n_ratio > 1.0:
            n_vio_p += severe_coef * (n_ratio - 1.0) 
            self.violation_count += 1
            self.g1_violation_count += 1
            
        # g2: Ratio violation penalty
        q_vio_p = 0.0
        ratio = self.state[2] / (self.state[0] + 1e-8)
        q_ratio = ratio / self.RATIO_LIMIT
        if q_ratio > 0.95 and q_ratio < 1.0:
            q_vio_p -= minor_coef * np.log(1.05 - q_ratio)
        if ratio > self.RATIO_LIMIT:
            q_vio_p += severe_coef * (q_ratio - 1.0)
            self.violation_count += 1
            self.g2_violation_count += 1
            
        # Smoothing
        smooth_p = 0.05 * np.mean(np.square(a_clipped - self.prev_action))
        self.prev_action = a_clipped.copy()
        
        # Nitrate usage (encourages metabolism efficiency)
        n_use_p = 0.005 * Fn_phys
        
        # Production Reward
        prod_r = self.state[2] * 10
        
        total_v_p = n_vio_p + q_vio_p
        step_reward = prod_r - (total_v_p + smooth_p + n_use_p)
        
        # Update metrics
        self.ep_total_reward += step_reward
        self.ep_rewards.append(step_reward)
        self.ep_prod_rewards.append(prod_r)
        self.ep_smooth_penalties.append(smooth_p)
        self.ep_n_usage_penalties.append(n_use_p)
        self.ep_violation_penalties.append(total_v_p)
        self.ep_g1_penalties.append(n_vio_p)
        self.ep_g2_penalties.append(q_vio_p)
        self.ep_g3_penalties.append(0.0)
        
        # --- Termination & Terminal Penalty ---
        done = self.time_step_count >= self.max_steps
        if done:
            # g3: Terminal Nitrate check [cite: 691]
            t_ratio = self.state[1] / self.N_LIMIT_TERM
            t_penalty = 0.0
            
            if t_ratio > 0.95 and t_ratio < 1.0:
                t_penalty -= minor_coef * np.log(1.05 - t_ratio) * 100
                
            if t_ratio > 1.0:
                t_penalty += severe_coef * (t_ratio - 1.0) * 100
                self.violation_count += 1
                self.g3_violation_count += 1
            else:
                self.ep_total_reward += 50.0 # Success bonus
                
            if t_penalty > 0:
                step_reward -= t_penalty
                self.ep_total_reward -= t_penalty
                self.ep_violation_penalties[-1] += t_penalty
                self.ep_g3_penalties[-1] += t_penalty
                self.ep_rewards[-1] -= t_penalty

        info = {
            "avg_reward": float(np.mean(self.ep_rewards)),
            "total_reward": self.ep_total_reward,
            "avg_prod_reward": float(np.mean(self.ep_prod_rewards)),
            "avg_smooth_penalty": float(np.mean(self.ep_smooth_penalties)),
            "avg_nitrate_usage_penalty": float(np.mean(self.ep_n_usage_penalties)),
            "avg_violation_penalty": float(np.mean(self.ep_violation_penalties)),
            "avg_g1_penalty": float(np.mean(self.ep_g1_penalties)),
            "avg_g2_penalty": float(np.mean(self.ep_g2_penalties)),
            "avg_g3_penalty": float(np.mean(self.ep_g3_penalties)),
            "violation_count": self.violation_count,
            "g1_violation_count": self.g1_violation_count,
            "g2_violation_count": self.g2_violation_count,
            "g3_violation_count": self.g3_violation_count
        }
        
        return self.get_state_norm(), step_reward, done, info