import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import os

# Hardware acceleration setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActionProjectionNetwork(nn.Module):
    """
    The pre-trained Safeguard. 
    Expects 4D Physical State: [cx, cN, cq, t_norm]
    """
    def __init__(self, state_dim=4, action_dim=2, latent_dim=512):
        super(ActionProjectionNetwork, self).__init__()
        # Synchronized with pretrain.py
        self.register_buffer("max_vals", torch.tensor([6.0, 800.0, 0.1, 1.0]))
        
        self.input_layer = nn.Linear(state_dim + action_dim, latent_dim)
        self.res_block1 = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        self.final_layer = nn.Linear(latent_dim, action_dim)
        self.elu = nn.ELU()

    def forward(self, state_phys, nom_act_norm, apply_override=True):
        # 1. Internal Static Normalization matching Pretrain
        state_norm = (state_phys / self.max_vals) * 2.0 - 1.0
        
        x_in = torch.cat([state_norm, nom_act_norm], dim=1)
        x_proj = self.elu(self.input_layer(x_in))
        
        x_res = self.res_block1(x_proj)
        x_combined = self.elu(x_res + x_proj)
        # 2. Residual Correction
        delta = torch.relu(self.final_layer(x_combined))
        u_nn = nom_act_norm - delta
        
        if not apply_override:
            return u_nn
        
        # 3. Explicit Analytical Override for G2 (Instantaneous)
        cx = state_phys[:, 0]
        cq = state_phys[:, 2]
        
        # 0.011 is the RATIO_LIMIT, 0.98 is the SAFE_BUFFER from data_gen
        g2_violation = cq > (cx * 0.011 * 0.98)
        
        u_safe = u_nn.clone()
        # Force Light Intent (idx 0) to maximum physical limit (+1.0 in normalized space)
        # This differentiable operation ensures perfect gradients for the PPO actor mapping penalty.
        u_safe[:, 0] = torch.where(g2_violation, torch.full_like(u_safe[:, 0], 1.0), u_nn[:, 0])
        
        return u_safe

class ActorCritic(nn.Module):
    """
    Learns the latent intent 'z' based on the 4D state.
    """
    def __init__(self, state_dim=4, action_dim=2):
        super(ActorCritic, self).__init__()
        self.LOG_STD_MIN = -2.0
        self.LOG_STD_MAX = 0.5

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def act(self, state):
        mean = self.actor(state)
        std = torch.exp(torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX))
        dist = Normal(mean, std)
        
        z_raw = dist.sample()
        z = torch.tanh(z_raw)
        
        # Tanh Jacobian Correction
        log_prob = dist.log_prob(z_raw).sum(dim=-1)
        log_prob -= torch.log(1 - z.pow(2) + 1e-6).sum(dim=-1)
        
        return z.detach(), log_prob.detach(), z_raw.detach()

    def evaluate(self, state, z_raw):
        mean = self.actor(state)
        std = torch.exp(torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX))
        dist = Normal(mean, std)
        
        log_probs = dist.log_prob(z_raw).sum(dim=-1)
        z_tanhed = torch.tanh(z_raw)
        log_probs -= torch.log(1 - z_tanhed.pow(2) + 1e-6).sum(dim=-1)
        
        dist_entropy = dist.entropy().sum(dim=-1)
        state_values = self.critic(state)
        return log_probs, state_values, dist_entropy

class SPRL_Agent:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, entropy_coeff):
        self.gamma = gamma               
        self.eps_clip = eps_clip         
        self.K_epochs = K_epochs         
        self.entropy_coeff = entropy_coeff 

        # 1. Actor-Critic (Now 4D to include Time)
        self.policy = ActorCritic(state_dim=4, action_dim=2).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': [self.policy.log_std], 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim=4, action_dim=2).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # 2. Safeguard
        self.safeguard = ActionProjectionNetwork(state_dim=4, action_dim=2).to(device)
        if os.path.exists("action_projection_network.pth"):
            self.safeguard.load_state_dict(torch.load("action_projection_network.pth", map_location=device))
            self.safeguard.eval()

        self.MseLoss = nn.MSELoss()
        self.mapping_criterion = nn.SmoothL1Loss(beta=0.01)

    def select_action(self, state_norm):
        """
        Input: state_norm [cx, cN, cq, t_norm] from environment.
        """
        with torch.no_grad():
            state_t = torch.FloatTensor(state_norm).to(device).unsqueeze(0)
            
            # 1. Generate Intent 'z'
            z, log_prob, z_raw = self.policy_old.act(state_t)
            
            # 2. Denormalize to Physical Units for Safeguard
            # [6.0, 800.0, 0.1, 1.0] matches env.py and pretrain.py
            phys_scale = torch.tensor([6.0, 800.0, 0.1, 1.0], device=device)
            state_phys = state_t * phys_scale
            
            # 3. Project to Safety
            u_safe = self.safeguard(state_phys, z)
            
        return u_safe.cpu().numpy().flatten(), log_prob.cpu().numpy(), z_raw.cpu().numpy().flatten()

    def learn(self, memory):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal: discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.squeeze(torch.stack(memory.states, dim=0)).detach().to(device)
        old_z_raw = torch.squeeze(torch.stack(memory.raw_actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs, dim=0)).detach().to(device)

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_z_raw)
            
            # 1. PPO Loss
            ratios = torch.exp(logprobs - old_logprobs)
            advantages = rewards - state_values.detach().squeeze()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            ppo_loss = -torch.min(surr1, surr2).mean()

            # 2. SP-RL Mapping Penalty
            with torch.no_grad():
                phys_scale = torch.tensor([6.0, 800.0, 0.1, 1.0], device=device)
                states_phys = old_states * phys_scale
                z_intent = torch.tanh(old_z_raw)
                u_safe = self.safeguard(states_phys, z_intent)

            # Penalty: If safeguard modified the action, punish the actor
            mapping_penalty = self.mapping_criterion(z_intent, u_safe)
            
            # 3. Total Loss
            loss = ppo_loss + \
                   0.5 * self.MseLoss(state_values.squeeze(), rewards) - \
                   self.entropy_coeff * dist_entropy.mean() + \
                   0.1 * mapping_penalty # Coefficient for safety alignment

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())