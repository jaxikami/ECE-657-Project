import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np

# Hardware setup [cite: 14]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCriticNonResNet(nn.Module):
    """
    Standard PPO Actor-Critic: Learns actions directly.
    Relies on environment penalties to learn constraints.
    """
    def __init__(self, state_dim, action_dim):
        super(ActorCriticNonResNet, self).__init__()
        self.LOG_STD_MIN = -2.0
        self.LOG_STD_MAX = 0.5

        # Actor: Maps state directly to action mean [cite: 14]
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh() # Squeezes action to [-1, 1] range
        )
        
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic: Estimating Value 
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def act(self, state):
        action_mean = self.actor(state)
        std = torch.exp(torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX))
        dist = Normal(action_mean, std)
        
        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(dim=-1)
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state)
        std = torch.exp(torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX))
        dist = Normal(action_mean, std)
        
        action_logprobs = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        state_values = self.critic(state)
        return action_logprobs, state_values, dist_entropy

class NonResNet_Agent:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCriticNonResNet(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': [self.policy.log_std], 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCriticNonResNet(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            state_t = torch.FloatTensor(state).to(device)
            action, action_logprob = self.policy_old.act(state_t)
        return action.cpu().numpy().flatten(), action_logprob.cpu().numpy()

    def learn(self, memory):
        # Standard PPO reward discounting 
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal: discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing rewards for stability [cite: 14]
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.squeeze(torch.stack(memory.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(memory.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs, dim=0)).detach().to(device)

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            ratios = torch.exp(logprobs - old_logprobs)
            advantages = rewards - state_values.detach().squeeze()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values.squeeze(), rewards) - 0.01*dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())