import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import os

# Hardware acceleration setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActionProjectionNetwork(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, latent_dim=512):
        super(ActionProjectionNetwork, self).__init__()
        self.input_layer = nn.Linear(state_dim + action_dim, latent_dim)
        
        self.res_block1 = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        self.final_layer = nn.Linear(latent_dim, action_dim)
        
        nn.init.uniform_(self.final_layer.weight, -1e-4, 1e-4)
        nn.init.constant_(self.final_layer.bias, 0)
        
        self.elu = nn.ELU()

    def forward(self, state_norm, nom_act_norm):
        x_in = torch.cat([state_norm, nom_act_norm], dim=1)
        x = self.elu(self.input_layer(x_in))
        
        identity = x
        x = self.res_block1(x)
        x = self.elu(x + identity) 
        
        # CHANGE: Use a small LeakyReLU or no ReLU at all during RL training
        # This allows the Actor to "feel" the gradient even in safe regions.
        delta = self.final_layer(x) 
        return nom_act_norm - delta
    
def static_normalize(states_phys):
    """
    Synchronized with pretrain.py logic.
    Converts physical states to [-1, 1] range.
    """
    # Max values for [cx, cN, cq, n_dist]
    max_vals = torch.tensor([6.0, 170.0, 25.0, 1.0], device=states_phys.device)
    return (states_phys / max_vals) * 2.0 - 1.0

class ActorCritic(nn.Module):
    """
    Standard PPO Actor-Critic: Learns the latent intent 'z'.
    Uses the 3-dim physical state from the environment.
    """
    def __init__(self, state_dim=3, action_dim=2, latent_dim=2):
        super(ActorCritic, self).__init__()
        self.latent_dim = latent_dim
        self.LOG_STD_MIN = -4.0
        self.LOG_STD_MAX = 0.0

        self.actor_base = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU()
        )
        
        self.actor_latent_mean = nn.Linear(128, latent_dim) 
        self.log_std = nn.Parameter(torch.ones(latent_dim) * -1.0)

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def act(self, state):
        base_output = self.actor_base(state)
        mean = self.actor_latent_mean(base_output) # Unbounded mean
        std = torch.exp(torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX))
        dist = Normal(mean, std)
        
        z_raw = dist.sample()
        z = torch.tanh(z_raw)
        
        # CRITICAL: Tanh Squashing Correction for LogProb
        log_prob = dist.log_prob(z_raw).sum(dim=-1)
        log_prob -= torch.log(1 - z.pow(2) + 1e-6).sum(dim=-1)
        
        return z.detach(), log_prob.detach()

    def evaluate(self, state, z_sampled):
        base_output = self.actor_base(state)
        # REMOVED torch.tanh from mean here to match act()
        mean = self.actor_latent_mean(base_output) 
        std = torch.exp(torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX))
        dist = Normal(mean, std)
        
        # Since z_sampled is already in Tanh space, we need the inverse to evaluate
        # or evaluate in raw space. Usually, PPO stores 'z_raw'.
        # For simplicity, ensure your memory stores 'z_raw' from the act() function.
        log_probs = dist.log_prob(z_sampled).sum(dim=-1)
        
        # Apply the same Tanh correction during learning
        z_tanhed = torch.tanh(z_sampled)
        log_probs -= torch.log(1 - z_tanhed.pow(2) + 1e-6).sum(dim=-1)
        
        dist_entropy = dist.entropy().sum(dim=-1)
        state_values = self.critic(state)
        
        return log_probs, state_values, dist_entropy

class SPRL_Agent:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, entropy_coeff, latent_dim=2):
        self.gamma = gamma               
        self.eps_clip = eps_clip         
        self.K_epochs = K_epochs         
        self.entropy_coeff = entropy_coeff 
        self.latent_dim = latent_dim

        # 1. Actor-Critic (Uses 3-dim physical state)
        self.policy = ActorCritic(state_dim, action_dim, latent_dim).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor_base.parameters(), 'lr': lr_actor},
            {'params': self.policy.actor_latent_mean.parameters(), 'lr': lr_actor},
            {'params': [self.policy.log_std], 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, latent_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # 2. Safeguard (Uses 4-dim state: 3 phys + 1 budget distance)
        self.safeguard = ActionProjectionNetwork(state_dim + 1, action_dim).to(device)
        if os.path.exists("action_projection_network.pth"):
            self.safeguard.load_state_dict(torch.load("action_projection_network.pth", map_location=device))
            self.safeguard.eval()
            print("Successfully loaded Residual Budget-Aware Safeguard.")
        
        # Load normalization constants
        # if os.path.exists("norm_constants.npz"):
        #     norms = np.load("norm_constants.npz")
        #     self.s_mean = torch.tensor(norms['s_mean'], dtype=torch.float32).to(device)
        #     self.s_std = torch.tensor(norms['s_std'], dtype=torch.float32).to(device)
        #     self.a_mean = torch.tensor(norms['a_mean'], dtype=torch.float32).to(device)
        #     self.a_std = torch.tensor(norms['a_std'], dtype=torch.float32).to(device)

        # High-precision loss for the mapping penalty
        self.criterion = nn.SmoothL1Loss(beta=0.005)
        self.return_rms = RunningMeanStd(shape=())

    def select_action(self, state):
        with torch.no_grad():
            # 1. Convert to physical units (Actor still wants these)
            # Ensure cN is capped at 170 for normalization stability
            state_phys = state * np.array([6.0, 200.0, 25.0]) 
            cN = state_phys[1]
            
            # 2. Budget Distance
            n_limit_phys = 180.0 * 0.95
            n_dist = max(0.0, n_limit_phys - cN) / n_limit_phys
            
            # 3. Form tensors
            state_t_3d = torch.FloatTensor(state_phys).to(device)
            state_t_4d = torch.FloatTensor(np.append(state_phys, n_dist)).to(device).unsqueeze(0)
            
            # 4. Generate Intent 'z'
            z, log_prob = self.policy_old.act(state_t_3d)
            
            # 5. NEW Static Normalization for Safeguard
            s_norm = static_normalize(state_t_4d)
            
            # 6. Safeguard Projection (Skip a_mean/a_std if pretrain was identity)
            # In your new pretrain.py, a_mean=0 and a_std=1.
            u_phi_norm = self.safeguard(s_norm, z.unsqueeze(0))
            
        return u_phi_norm.cpu().numpy().flatten(), z.cpu().numpy(), log_prob.cpu().numpy()
    
    def learn(self, memory):
        # 1. Discounted Reward Calculation (Standard PPO)
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal: 
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # 2. Reward Normalization using Running Statistics
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(device)
        self.return_rms.update(rewards_tensor.cpu().numpy())
        scaled_rewards = rewards_tensor / (np.sqrt(self.return_rms.var) + 1e-8)

        # 3. Prepare Tensors (Convert environment [0,1] states to physical units)
        states_norm = torch.squeeze(torch.stack(memory.states, dim=0)).detach().to(device)
        # Ensure scaling factors [6.0, 200.0, 25.0] match env.py exactly
        old_states_phys = states_norm * torch.tensor([6.0, 200.0, 25.0], device=device)
        
        old_latents = torch.squeeze(torch.stack(memory.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs, dim=0)).detach().to(device)

        for _ in range(self.K_epochs):
            # 4. Evaluate current policy
            # Uses physical 3D states for the Actor-Critic
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states_phys, old_latents)
            
            # 5. PPO Surrogate Loss
            ratios = torch.exp(logprobs - old_logprobs)
            advantages = scaled_rewards - state_values.detach().squeeze()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            ppo_loss = -torch.min(surr1, surr2).mean()

            # 6. SP-RL Mapping Penalty (Residual Budget-Aware)
            with torch.no_grad():
                cN_batch = old_states_phys[:, 1]
                n_limit_phys = 180.0 * 0.95
                n_dist_batch = torch.clamp(n_limit_phys - cN_batch, min=0.0) / n_limit_phys
                
                # Stack to 4D for the Safeguard: [cx, cN, cq, n_dist]
                old_states_4d = torch.cat([old_states_phys, n_dist_batch.unsqueeze(1)], dim=1)
                
                # Apply Synchronized Static Normalization
                s_n = static_normalize(old_states_4d)
                
                # Project the actor's intent through the pre-trained safeguard
                u_safe = self.safeguard(s_n, old_latents)

            # 7. Calculate Differentiable Penalty
            # Penalty is only applied if the safeguard had to modify the action
            diff = old_latents - u_safe
            mask = (torch.abs(diff) > 0.005).float() 
            mapping_penalty = (self.criterion(old_latents, u_safe) * mask).mean()
            
            # 8. Combined Loss with Gradient Clipping
            # Reduced mapping_penalty_coeff to prevent early convergence to "safe but low" rewards
            mapping_penalty_coeff = 0.001 
            value_loss_coeff = 0.5
            
            loss = ppo_loss + \
                   value_loss_coeff * self.criterion(state_values.squeeze(), scaled_rewards) - \
                   self.entropy_coeff * dist_entropy.mean() + \
                   mapping_penalty_coeff * mapping_penalty

            # 9. Optimization Step
            self.optimizer.zero_grad()
            loss.backward()
            # Clip gradients to 0.5 to prevent numeric instability from the exponential rates
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

        # Update the old policy for the next iteration
        self.policy_old.load_state_dict(self.policy.state_dict())

class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        self.mean, self.var, self.count = new_mean, new_var, tot_count