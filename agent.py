import os
# Standard fix for OpenMP runtime conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

# 1. The Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()
        
        self.base = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        features = self.base(state)
        mu = self.actor(features)
        value = self.critic(features)
        
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        
        return dist, value

# 2. The PPO Agent with Residual ResNet Steering
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, K_epochs=10, eps_clip=0.2, resnet_model=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        
        # --- Load Normalization Constants from Pretraining ---
        if os.path.exists("norm_constants.npz"):
            norms = np.load("norm_constants.npz")
            self.s_mean = torch.FloatTensor(norms['s_mean']).to(self.device)
            self.s_std = torch.FloatTensor(norms['s_std']).to(self.device) + 1e-8
            self.a_mean = torch.FloatTensor(norms['a_mean']).to(self.device)
            self.a_std = torch.FloatTensor(norms['a_std']).to(self.device) + 1e-8
            print("✅ Agent loaded normalization constants from norm_constants.npz")
        else:
            print("⚠️ Warning: norm_constants.npz not found. Environment scaling may be incorrect.")
            self.s_mean, self.s_std = 0, 1
            self.a_mean, self.a_std = 0, 1

        self.resnet = resnet_model
        if self.resnet:
            self.resnet.eval()
            for param in self.resnet.parameters():
                param.requires_grad = False

    def select_action(self, state, evaluate=False):
        state_raw = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # 1. Normalize State
        state_norm = (state_raw - self.s_mean) / self.s_std
        
        with torch.no_grad():
            dist, value = self.policy_old(state_norm)
            # 2. Sample action in Normalized Space
            if evaluate:
                action_norm = dist.mean
            else:
                action_norm = dist.sample()
            
            action_logprob = dist.log_prob(action_norm).sum(dim=-1)
        
        # --- THE RESIDUAL SAFETY BRIDGE ---
        if self.resnet:
            # ResNet now expects [Normalized State, Normalized Action] 
            # and returns [Normalized Safe Action]
            safe_action_norm = self.resnet(state_norm, action_norm)
        else:
            safe_action_norm = action_norm
            
        # 3. Denormalize final safe action for the Environment
        safe_action_phys = safe_action_norm * self.a_std + self.a_mean
        nominal_action_phys = action_norm * self.a_std + self.a_mean
            
        return nominal_action_phys.cpu().numpy().flatten(), \
               safe_action_phys.cpu().numpy().flatten(), \
               action_logprob.cpu().item(), \
               value.cpu().item()

    def update(self, buffer):
        # Update loop remains consistent: Policy learns mapping from S_norm to A_norm
        old_states_raw = torch.FloatTensor(np.array(buffer.states)).to(self.device)
        old_states_norm = (old_states_raw - self.s_mean) / self.s_std
        
        # Buffer stores the original nominal action (normalized)
        old_actions_raw = torch.FloatTensor(np.array(buffer.actions)).to(self.device)
        old_actions_norm = (old_actions_raw - self.a_mean) / self.a_std
        
        old_logprobs = torch.FloatTensor(np.array(buffer.logprobs)).to(self.device)
        
        returns = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(buffer.rewards), reversed(buffer.is_terminals)):
            if is_terminal: discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
            
        returns = torch.FloatTensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        for _ in range(self.K_epochs):
            dist, state_values = self.policy(old_states_norm)
            logprobs = dist.log_prob(old_actions_norm).sum(dim=-1)
            dist_entropy = dist.entropy().sum(dim=-1)
            state_values = torch.squeeze(state_values)
            
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = returns - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, returns) - 0.01*dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())

class PPOBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]