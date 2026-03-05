import numpy as np
from numba import njit

# --- KINETIC ENGINE (Numba Accelerated) ---
@njit
def calculate_rates_numba(x, u, params):
    u_m, u_d, Y_NX = 0.0572, 0.0, 504.5
    k_m, k_d, k_sq, K_Nq, k_iq = 0.00016, 0.281, 23.51, 16.89, 800.0
    k_s, k_i, k_N = 178.9, 447.1, 393.1
    
    c_x, c_N, c_q = x[0], x[1], x[2]
    I, F_N = u[0], u[1]
    
    phi_I = I / (I + k_s + (I**2 / k_i))
    phi_I_q = I / (I + k_sq + (I**2 / k_iq))
    phi_N = c_N / (c_N + k_N)
    
    dc_x = u_m * phi_I * c_x * phi_N - u_d * c_x
    dc_N = -Y_NX * u_m * phi_I * c_x * phi_N + F_N
    dc_q = k_m * phi_I_q * c_x - (k_d * c_q) / (c_N + K_Nq)
    
    return np.array([dc_x, dc_N, dc_q])

@njit
def integrate_rk4(x_init, u, params, dt, n_steps):
    x = x_init.astype(np.float64)
    step_size = dt / n_steps
    for _ in range(n_steps):
        k1 = calculate_rates_numba(x, u, params)
        k2 = calculate_rates_numba(x + 0.5 * step_size * k1, u, params)
        k3 = calculate_rates_numba(x + 0.5 * step_size * k2, u, params)
        k4 = calculate_rates_numba(x + step_size * k3, u, params)
        x += (step_size / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        x = np.maximum(x, 1e-6) 
        if np.any(np.isnan(x)) or np.max(x) > 5000.0:
            return x, True
    return x, False

# --- UPDATED ENVIRONMENT CLASS ---
class PhotoProductionEnv:
    def __init__(self, train_mode=True):
        """
        PhotoProduction Environment with explicit mode selection.
        :param train_mode: If True, adds 0.05 Gaussian noise to actions.
        """
        self.stateDim = 3
        self.actionDim = 2
        
        # Physical Constants
        self.I_max = 1000.0
        self.Fn_max = 20.0
        self.I_crit = 450.0
        self.alpha = 0.25
        self.L = 0.5
        
        # Mode Config
        self.train_mode = train_mode
        self.noise_std = 0.05
        
        # Episode Config
        self.max_steps = 200
        self.dt = 1.0  
        self.n_inner_steps = 10
        self.params = np.zeros(1)
        
        self.reset()

    def reset(self):
        self.time_step_count = 0
        # Change from float32 to float64 (default)
        self.state = np.array([0.5, 100.0, 5.0], dtype=np.float64) 
        self.prev_action = np.zeros(self.actionDim)
        return self.get_state_norm()

    def get_state_norm(self):
        # Ensure the normalization scales and return type are float32 for PyTorch compatibility
        norm_scales = np.array([6.0, 200.0, 25.0], dtype=np.float64)
        return (self.state / norm_scales).astype(np.float32)

    def step(self, action):
        """Executes one step with noise injection and detailed info logging."""
        # 1. Action Processing with Conditional Noise
        processed_action = action.copy()
        if not self.train_mode:
            noise = np.random.normal(0, self.noise_std, size=self.actionDim)
            processed_action += noise
        
        # 2. Denormalization & Scaling
        a_clipped = np.clip(processed_action, -1.0, 1.0)
        a_scaled = (a_clipped + 1.0) / 2.0
        
        I = a_scaled[0] * self.I_max
        Fn = a_scaled[1] * self.Fn_max
        u = np.array([I, Fn])

        # 3. Physics Integration
        new_state, unstable = integrate_rk4(
            self.state, u, self.params, self.dt, self.n_inner_steps
        )
        self.state = new_state

        # 4. Constraint Check
        shading_factor = np.exp(-self.alpha * self.state[0] * self.L)
        i_limit = self.I_crit / (shading_factor + 1e-6)
        
        i_violation = max(0.0, I - i_limit)
        n_violation = max(0.0, self.state[1] - 180.0)

        is_safe = (i_violation == 0.0 and n_violation == 0.0)

        # 5. Reward vs. Penalty Calculation
        # REWARDS (Positive Gains)
        reward_prod = self.state[2]/4  # Phycocyanin concentration
        
        # PENALTIES (Negative Values)
        if not is_safe:
            # Normalize the violation distance by the max possible value to keep scales consistent
            i_penalty = (i_violation / self.I_max) * 10.0 
            n_penalty = (n_violation / self.Fn_max) * 10.0
            penalty_safety = -(1.0 + i_penalty + n_penalty)/5 
        else:
            penalty_safety = 0.0
        penalty_smoothing = -0.2 * np.mean(np.square(a_clipped - self.prev_action))
        penalty_biomass = -0.05 * self.state[0]
        
        total_reward = reward_prod + penalty_safety + penalty_smoothing + penalty_biomass
        
        self.prev_action = a_clipped.copy()
        self.time_step_count += 1
        
        done = (self.time_step_count >= self.max_steps or 
                unstable or self.state[0] < 0.01)

        # Updated info dictionary containing full breakdown
        info = {
            "biomass": self.state[0],
            "nitrate": self.state[1],
            "product": self.state[2],
            "is_safe": is_safe,
            "unstable": unstable,
            "reward": {
                "production": reward_prod
            },
            "penalties": {
                "safety": penalty_safety,
                "smoothing": penalty_smoothing,
                "biomass_efficiency": penalty_biomass,
                "total_penalty": penalty_safety + penalty_smoothing + penalty_biomass
            }
        }

        return self.get_state_norm(), total_reward, done, info