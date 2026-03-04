import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import os

# Hardware acceleration setup [cite: 14]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActionProjectionNetwork(nn.Module):
    def __init__(self, state_dim=3, action_dim=2, latent_dim=512):
        super(ActionProjectionNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ELU(), # Match pretrain.py
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ELU(), # Match pretrain.py
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ELU(), # Match pretrain.py
            nn.Linear(latent_dim // 2, action_dim) 
        )

    def forward(self, state_norm, nom_act_norm):
        x = torch.cat([state_norm, nom_act_norm], dim=1)
        delta_norm = self.net(x)
        return nom_act_norm + delta_norm

class ActorCritic(nn.Module):
    """
    SP-RL Actor-Critic: Learns the latent intent 'z' which is then 
    passed through the safeguard[cite: 15, 59].
    """
    def __init__(self, state_dim, action_dim, latent_dim):
        super(ActorCritic, self).__init__()
        self.latent_dim = latent_dim
        self.LOG_STD_MIN = -4.0
        self.LOG_STD_MAX = 0.0

        # Actor Base: Shared features [cite: 14]
        self.actor_base = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU()
        )
        
        # Mean of the latent intent (z)
        self.actor_latent_mean = nn.Linear(128, latent_dim) 
        self.log_std = nn.Parameter(torch.ones(latent_dim) * -1.0)

        # Critic: Estimating Value [cite: 156]
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def act(self, state):
        """Generates latent intent z[cite: 15]."""
        base_output = self.actor_base(state)
        mean = self.actor_latent_mean(base_output)
        std = torch.exp(torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX))
        dist = Normal(mean, std)
        
        z_raw = dist.sample()
        z = torch.tanh(z_raw)
        log_prob = dist.log_prob(z_raw).sum(dim=-1)
        return z.detach(), log_prob.detach()

    def evaluate(self, state, z_sampled):
        """Evaluates latents with matching squashing logic."""
        base_output = self.actor_base(state)
        mean = self.actor_latent_mean(base_output)
        
        # We use tanh on the mean to help the network converge towards 
        # the [-1, 1] range the safeguard expects.
        mean = torch.tanh(mean) 
        
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

        # 1. Initialize Primary Policy (Predicts Z) [cite: 15]
        self.policy = ActorCritic(state_dim, action_dim, latent_dim).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor_base.parameters(), 'lr': lr_actor},
            {'params': self.policy.actor_latent_mean.parameters(), 'lr': lr_actor},
            {'params': [self.policy.log_std], 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, latent_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # 2. Load the Safeguard (Action Projection Network) [cite: 12, 59]
        self.safeguard = ActionProjectionNetwork(state_dim, action_dim).to(device)
        if os.path.exists("action_projection_network.pth"):
            self.safeguard.load_state_dict(torch.load("action_projection_network.pth", map_location=device))
            self.safeguard.eval()
            print("Successfully loaded Action Projection Safeguard.")
        
        # Load normalization constants [cite: 12]
        if os.path.exists("norm_constants.npz"):
            norms = np.load("norm_constants.npz")
            self.s_mean = torch.tensor(norms['s_mean'], dtype=torch.float32).to(device)
            self.s_std = torch.tensor(norms['s_std'], dtype=torch.float32).to(device)
            self.a_mean = torch.tensor(norms['a_mean'], dtype=torch.float32).to(device)
            self.a_std = torch.tensor(norms['a_std'], dtype=torch.float32).to(device)

        self.criterion = nn.SmoothL1Loss(beta=0.1)
        self.return_rms = RunningMeanStd(shape=())

    def select_action(self, state):
        """SP-RL: Agent picks intent z, Safeguard projects to safe action u_phi[cite: 42, 236]."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).to(device)
            z, log_prob = self.policy_old.act(state_t)
            
            # Normalize for safeguard [cite: 12]
            s_norm = (state_t.unsqueeze(0) - self.s_mean) / (self.s_std + 1e-8)
            z_norm = (z.unsqueeze(0) - self.a_mean) / (self.a_std + 1e-8)
            
            # Project to safe manifold [cite: 13, 202]
            u_phi_norm = self.safeguard(s_norm, z_norm)
            u_phi = (u_phi_norm * (self.a_std + 1e-8)) + self.a_mean
            
        return u_phi.cpu().numpy().flatten(), z.cpu().numpy(), log_prob.cpu().numpy()

    def learn(self, memory):
        """PPO update with SP-RL Mapping Penalty[cite: 47, 78]."""
        # Standard reward discounting and normalization [cite: 180, 182]
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal: discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(device)
        self.return_rms.update(rewards_tensor.cpu().numpy())
        scaled_rewards = rewards_tensor / (np.sqrt(self.return_rms.var) + 1e-8)

        old_states = torch.squeeze(torch.stack(memory.states, dim=0)).detach().to(device)
        old_latents = torch.squeeze(torch.stack(memory.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs, dim=0)).detach().to(device)

        # Optimize policy
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_latents)
            
            # PPO Ratio [cite: 15]
            ratios = torch.exp(logprobs - old_logprobs)
            
            # Advantages [cite: 178]
            advantages = scaled_rewards - state_values.detach().squeeze()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # --- SP-RL MAPPING PENALTY (Regularization) [cite: 78, 427] ---
            # Penalize distance between intent and safe projection to prevent zero-gradients.
            with torch.no_grad():
                # 1. Normalize current state and sampled intent
                s_n = (old_states - self.s_mean) / (self.s_std + 1e-8)
                z_n = (old_latents - self.a_mean) / (self.a_std + 1e-8)
                
                # 2. Get the safe projection from the safeguard
                u_safe_n = self.safeguard(s_n, z_n)
                u_safe = (u_safe_n * (self.a_std + 1e-8)) + self.a_mean
            diff = old_latents - u_safe
            mask = (torch.abs(diff) > 1e-3).float()
            mapping_penalty = (self.criterion(old_latents, u_safe) * mask).mean()
            value_loss = 0.5 * self.criterion(state_values.squeeze(), scaled_rewards)
            mapping_penalty_coeff = 0.1  # Hyperparameter to balance the penalty
            # Combined Loss [cite: 411, 427]
            loss = -torch.min(surr1, surr2) + \
                value_loss - \
                self.entropy_coeff * dist_entropy + \
                mapping_penalty_coeff * mapping_penalty  # <--- SP-RL Action Aliasing Penalty

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

class RunningMeanStd:
    """Standard normalization utility for stable training."""
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
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count