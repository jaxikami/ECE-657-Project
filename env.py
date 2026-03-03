import numpy as np
from numba import njit

# --- KINETIC ENGINE (Numba Accelerated) ---
@njit
def calculate_rates_numba(x, u, params):
    """
    State derivatives for the photo production model.
    x: [c_x, c_N, c_q] (Biomass, Nitrate, Phycocyanin)
    u: [I, F_N] (Light Intensity, Nitrate Feed Rate)
    """
    # Parameters from dataclass
    u_m, u_d, Y_NX = 0.0572, 0.0, 504.5
    k_m, k_d, k_sq, K_Nq, k_iq = 0.00016, 0.281, 23.51, 16.89, 800.0
    k_s, k_i, k_N = 178.9, 447.1, 393.1
    
    c_x, c_N, c_q = x[0], x[1], x[2]
    I, F_N = u[0], u[1]
    
    # Light intensity saturation/inhibition term
    phi_I = I / (I + k_s + (I**2 / k_i))
    phi_I_q = I / (I + k_sq + (I**2 / k_iq))
    
    # Nitrate saturation term
    phi_N = c_N / (c_N + k_N)
    
    # ODE System
    dc_x = u_m * phi_I * c_x * phi_N - u_d * c_x
    dc_N = -Y_NX * u_m * phi_I * c_x * phi_N + F_N
    dc_q = k_m * phi_I_q * c_x - (k_d * c_q) / (c_N + K_Nq)
    
    return np.array([dc_x, dc_N, dc_q])

@njit
def integrate_rk4(x_init, u, params, dt, n_steps):
    """
    Runge-Kutta 4th Order Integration.
    """
    x = x_init.copy()
    step_size = dt / n_steps
    
    for _ in range(n_steps):
        k1 = calculate_rates_numba(x, u, params)
        k2 = calculate_rates_numba(x + 0.5 * step_size * k1, u, params)
        k3 = calculate_rates_numba(x + 0.5 * step_size * k2, u, params)
        k4 = calculate_rates_numba(x + step_size * k3, u, params)
        
        x += (step_size / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        
        # Physical boundary clipping
        x = np.maximum(x, 1e-6) 
        
        if np.any(np.isnan(x)) or np.max(x) > 5000.0:
            return x, True # Instability detected
            
    return x, False

# --- ENVIRONMENT CLASS ---
class PhotoProductionEnv:
    def __init__(self):
        # State: [cx, cN, cq]
        self.stateDim = 3
        # Action: [I, F_N]
        self.actionDim = 2
        
        # Physical Constraints
        self.I_max = 3000.0
        self.Fn_max = 20.0
        self.I_crit = 450.0
        self.alpha = 0.25
        self.L = 0.5
        
        # Episode Config
        self.max_steps = 200
        self.dt = 1.0  # Hours per step
        self.n_inner_steps = 10
        self.params = np.zeros(1) # Placeholder for future uncertain params
        
        self.reset()

    def reset(self):
        """Resets to standard starting concentrations."""
        self.time_step_count = 0
        # cx=0.5, cN=100.0, cq=5.0
        self.state = np.array([0.5, 100.0, 5.0], dtype=np.float32)
        self.prev_action = np.zeros(self.actionDim)
        return self.get_state_norm()

    def get_state_norm(self):
        """Standardizes state for the ActionProjectionNetwork."""
        # Biomass max ~6.0, Nitrate max ~200.0, Phycocyanin max ~25.0
        norm_scales = np.array([6.0, 200.0, 25.0], dtype=np.float32)
        return (self.state / norm_scales).astype(np.float32)

    def step(self, action):
        """
        Executes one environment step with full reward tracking in info.
        """
        # 1. Action Denormalization [-1, 1] -> [Physical Range]
        a_clipped = np.clip(action, -1.0, 1.0)
        a_scaled = (a_clipped + 1.0) / 2.0
        
        I = a_scaled[0] * self.I_max
        Fn = a_scaled[1] * self.Fn_max
        u = np.array([I, Fn])

        # 2. Physics Integration (Numba Accelerated)
        new_state, unstable = integrate_rk4(
            self.state, u, self.params, self.dt, self.n_inner_steps
        )
        self.state = new_state

        # 3. Constraint Check (Photoinhibition & Toxicity)
        shading_factor = np.exp(-self.alpha * self.state[0] * self.L)
        i_limit = self.I_crit / (shading_factor + 1e-6)
        
        is_safe = True
        if I > i_limit: is_safe = False
        if self.state[1] > 180.0: is_safe = False

        # 4. Reward Calculation Breakdown
        # Primary Gain: Phycocyanin concentration
        reward_prod = self.state[2] 
        
        # Penalty 1: Safety Violation (Heavy penalty for breaking constraints)
        reward_safety_penalty = -5.0 if not is_safe else 0.0
        
        # Penalty 2: Smoothing (Cost of aggressive actuator changes)
        reward_smoothing_penalty = -0.01 * np.mean(np.square(a_clipped - self.prev_action))
        
        # Penalty 3: Biomass Usage (The "efficiency" cost)
        reward_biomass_penalty = -0.1 * self.state[0]
        
        total_reward = (reward_prod + 
                        reward_safety_penalty + 
                        reward_smoothing_penalty + 
                        reward_biomass_penalty)
        
        # 5. Housekeeping & Monitoring
        self.prev_action = a_clipped.copy()
        self.time_step_count += 1
        
        # Termination conditions
        done = (self.time_step_count >= self.max_steps or 
                unstable or 
                self.state[0] < 0.01)

        # Full info tracking for benchmarking
        info = {
            "biomass": self.state[0],
            "nitrate": self.state[1],
            "product": self.state[2],
            "is_safe": is_safe,
            "unstable": unstable,
            "reward_breakdown": {
                "production": reward_prod,
                "safety_penalty": reward_safety_penalty,
                "smoothing_penalty": reward_smoothing_penalty,
                "biomass_penalty": reward_biomass_penalty
            }
        }

        return self.get_state_norm(), total_reward, done, info