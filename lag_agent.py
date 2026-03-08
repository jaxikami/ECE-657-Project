import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np

# Hardware setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCriticNonResNet(nn.Module):
    """
    Standard PPO Actor-Critic: Learns actions directly.
    Relies on environment penalties to learn constraints.
    Updated to handle 4D state space (Biomass, Nitrate, Product, Time).
    """
    def __init__(self, state_dim, action_dim):
        super(ActorCriticNonResNet, self).__init__()
        self.LOG_STD_MIN = -2.0
        self.LOG_STD_MAX = 0.5

        # Actor: Maps 4D state to action mean
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

        # Critic: Estimating Value
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def act(self, state):
        """Generates an action and its log probability."""
        action_mean = self.actor(state)
        std = torch.exp(torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX))
        dist = Normal(action_mean, std)
        
        # Sample in raw space then squash to [-1, 1]
        raw_action = dist.sample()
        action = torch.tanh(raw_action)
        
        # Tanh Squashing Correction for LogProb
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
        
        return action.detach(), log_prob.detach(), raw_action.detach()

    def evaluate(self, state, raw_action):
        """Evaluates actions during the learning phase."""
        action_mean = self.actor(state)
        std = torch.exp(torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX))
        dist = Normal(action_mean, std)
        
        # Calculate log_probs using the raw sampled action
        action_logprobs = dist.log_prob(raw_action).sum(dim=-1)
        
        # Re-apply Tanh correction for the current policy evaluation
        action_squashed = torch.tanh(raw_action)
        action_logprobs -= torch.log(1 - action_squashed.pow(2) + 1e-6).sum(dim=-1)
        
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
        """Selects an action using the current 'old' policy."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).to(device)
            action, action_logprob, raw_action = self.policy_old.act(state_t)
        # We return raw_action to store in memory for correct evaluation
        return action.cpu().numpy().flatten(), action_logprob.cpu().numpy(), raw_action.cpu().numpy().flatten()

    def learn(self, memory):
        """Standard PPO update logic."""
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal: discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalize rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.squeeze(torch.stack(memory.states, dim=0)).detach().to(device)
        # CRITICAL: We now evaluate using the raw actions stored in memory
        old_raw_actions = torch.squeeze(torch.stack(memory.raw_actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs, dim=0)).detach().to(device)

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_raw_actions)
            
            ratios = torch.exp(logprobs - old_logprobs)
            advantages = rewards - state_values.detach().squeeze()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # Combined loss: Policy + Value + Entropy
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values.squeeze(), rewards) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            # Gradient clipping for stability
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())