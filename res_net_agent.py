import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import os

# Hardware acceleration setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActionProjectionNetwork(nn.Module):
    """
    Residual Safeguard: Predicts the 'reduction' (delta) to be subtracted from intent.
    Uses a 4-dim state including the Nitrogen Budget Distance.
    """
    def __init__(self, state_dim=4, action_dim=2, latent_dim=512): 
        super(ActionProjectionNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ELU(),
            nn.Linear(latent_dim, action_dim) 
        )

    def forward(self, state_norm, nom_act_norm):
        # Match the Residual Delta logic: Safe = Intent - ReLU(Delta)
        x = torch.cat([state_norm, nom_act_norm], dim=1)
        delta = torch.relu(self.net(x)) 
        return nom_act_norm - delta

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
        mean = self.actor_latent_mean(base_output)
        std = torch.exp(torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX))
        dist = Normal(mean, std)
        
        z_raw = dist.sample()
        z = torch.tanh(z_raw)
        log_prob = dist.log_prob(z_raw).sum(dim=-1)
        return z.detach(), log_prob.detach()

    def evaluate(self, state, z_sampled):
        base_output = self.actor_base(state)
        mean = torch.tanh(self.actor_latent_mean(base_output)) 
        std = torch.exp(torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX))
        dist = Normal(mean, std)
        
        log_probs = dist.log_prob(z_sampled).sum(dim=-1)
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
        if os.path.exists("norm_constants.npz"):
            norms = np.load("norm_constants.npz")
            self.s_mean = torch.tensor(norms['s_mean'], dtype=torch.float32).to(device)
            self.s_std = torch.tensor(norms['s_std'], dtype=torch.float32).to(device)
            self.a_mean = torch.tensor(norms['a_mean'], dtype=torch.float32).to(device)
            self.a_std = torch.tensor(norms['a_std'], dtype=torch.float32).to(device)

        # High-precision loss for the mapping penalty
        self.criterion = nn.SmoothL1Loss(beta=0.005)
        self.return_rms = RunningMeanStd(shape=())

    def select_action(self, state):
        with torch.no_grad():
            # 1. Physical Conversion
            state_phys = state * np.array([6.0, 200.0, 25.0])
            cN = state_phys[1]
            
            # 2. Budget Distance Calculation
            n_limit_phys = 180.0 * 0.95
            n_dist = max(0.0, n_limit_phys - cN) / n_limit_phys
            
            # 3. Form 4D state for Safeguard and 3D for Actor
            state_4d = np.append(state_phys, n_dist)
            state_t_4d = torch.FloatTensor(state_4d).to(device)
            state_t_3d = torch.FloatTensor(state_phys).to(device)
            
            # 4. Generate Latent Intent 'z'
            z, log_prob = self.policy_old.act(state_t_3d)
            
            # 5. Safeguard Projection
            s_norm = (state_t_4d.unsqueeze(0) - self.s_mean) / (self.s_std + 1e-8)
            z_norm = (z.unsqueeze(0) - self.a_mean) / (self.a_std + 1e-8)
            u_phi_norm = self.safeguard(s_norm, z_norm)
            
            # 6. Physical Renormalization
            u_phi = (u_phi_norm * (self.a_std + 1e-8)) + self.a_mean
            
        return u_phi.cpu().numpy().flatten(), z.cpu().numpy(), log_prob.cpu().numpy()

    def learn(self, memory):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal: discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(device)
        self.return_rms.update(rewards_tensor.cpu().numpy())
        scaled_rewards = rewards_tensor / (np.sqrt(self.return_rms.var) + 1e-8)

        # Agent states are [0, 1] normalized from env; convert to physical for mapping penalty
        states_norm = torch.squeeze(torch.stack(memory.states, dim=0)).detach().to(device)
        old_states_phys = states_norm * torch.tensor([6.0, 200.0, 25.0], device=device)
        
        old_latents = torch.squeeze(torch.stack(memory.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs, dim=0)).detach().to(device)

        for _ in range(self.K_epochs):
            # Actor uses the 3-dim physical state
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states_phys, old_latents)
            
            ratios = torch.exp(logprobs - old_logprobs)
            advantages = scaled_rewards - state_values.detach().squeeze()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # --- SP-RL MAPPING PENALTY ---
            with torch.no_grad():
                # 1. Re-calculate Budget Distance for batch
                cN_batch = old_states_phys[:, 1]
                n_limit_phys = 180.0 * 0.95
                n_dist_batch = torch.clamp(n_limit_phys - cN_batch, min=0.0) / n_limit_phys
                
                # 2. Re-stack to 4D for the safeguard
                old_states_4d = torch.cat([old_states_phys, n_dist_batch.unsqueeze(1)], dim=1)
                
                # 3. Projection
                s_n = (old_states_4d - self.s_mean) / (self.s_std + 1e-8)
                z_n = (old_latents - self.a_mean) / (self.a_std + 1e-8)
                u_safe_n = self.safeguard(s_n, z_n)
                u_safe = (u_safe_n * (self.a_std + 1e-8)) + self.a_mean

            # Penalty applies if safeguard throttles production
            diff = old_latents - u_safe
            mask = (torch.abs(diff) > 1e-4).float() 
            mapping_penalty = (self.criterion(old_latents, u_safe) * mask).mean()
            
            value_loss = 0.5 * self.criterion(state_values.squeeze(), scaled_rewards)
            mapping_penalty_coeff = 0.05 # Slightly lower to prioritize production reward

            loss = -torch.min(surr1, surr2) + \
                value_loss - \
                self.entropy_coeff * dist_entropy + \
                mapping_penalty_coeff * mapping_penalty

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

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