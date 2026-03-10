
import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np

# Hardware setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCriticNonResNet(nn.Module):
    """
    Standard PPO Actor-Critic neural network architecture.
    
    This agent learns mapping from states to actions directly. It is considered
    'NonResNet' compared to the constrained 'ResNetAgent' because it lacks the
    built-in action projection (skip-connection safety filter) manifold.
    
    It outputs a probability distribution over the action space, allowing the 
    agent to explore the environment safely.
    """
    def __init__(self, state_dim=4, action_dim=2):
        super(ActorCriticNonResNet, self).__init__()
        self.LOG_STD_MIN = -1.0
        self.LOG_STD_MAX = 0.5

        # Actor Network: Maps the 4D state space directly to an Action Mean (intent).
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
        # Trainable independent standard deviation parameter for action exploration.
        # Initialized to -0.5 (in log space) to provide moderate initial exploration.
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

        # Critic: Estimating Value
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def act(self, state):
        """
        Generates an action during the environment rollout phase.
        
        Returns:
            z: The squashed action mapped to [-1, 1] for physics denormalization.
            log_prob: The log probability of taking this action (used in PPO updates).
            z_raw: The unbounded Gaussian sample (needed to calculate exact log-probs during training).
        """
        mean = self.actor(state)
        std = torch.exp(torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX))
        dist = Normal(mean, std)
        
        # 1. Sample from unbounded Gaussian first
        z_raw = dist.sample()
        # 2. Squash to the physical bounds of [-1, 1] requirement using Tanh
        z = torch.tanh(z_raw)
        
        # 3. Tanh Squashing Correction for LogProb
        # Since we squashed the distribution, we must strictly adjust the log probability 
        # using the change-of-variables formula for the Tanh inverse transform mapping.
        log_prob = dist.log_prob(z_raw).sum(dim=-1)
        log_prob -= torch.log(1 - z.pow(2) + 1e-6).sum(dim=-1)
        
        return z.detach(), log_prob.detach(), z_raw.detach()

    def evaluate(self, state, z_raw):
        """
        Evaluates the probabilities and values of previous actions during the PPO update loop.
        Critically, this function requires the UNBOUNDED `z_raw` to recalculate 
        the exact Gaussian probabilities.
        """
        mean = self.actor(state)
        std = torch.exp(torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX))
        dist = Normal(mean, std)
        
        # Calculate log_probs using the raw unbounded sampled action stored in memory
        log_probs = dist.log_prob(z_raw).sum(dim=-1)
        
        # Re-apply the Tanh mathematical correction for the current policy evaluation
        z_tanhed = torch.tanh(z_raw)
        log_probs -= torch.log(1 - z_tanhed.pow(2) + 1e-6).sum(dim=-1)
        
        dist_entropy = dist.entropy().sum(dim=-1)
        state_values = self.critic(state)
        return log_probs, state_values, dist_entropy

class NonResNet_Agent:
    """
    Wrapper class managing the PPO learning algorithms, memory, and model updates.
    """
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, entropy_coeff):
        self.gamma = gamma          # Discount factor for future rewards
        self.eps_clip = eps_clip    # PPO surrogate objective clipping parameter
        self.K_epochs = K_epochs    # Number of policy update iterations per rollout batch
        self.entropy_coeff = entropy_coeff
        
        # 1. Actor-Critic (Now 4D to include Time)
        self.policy = ActorCriticNonResNet(state_dim=4, action_dim=2).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': [self.policy.log_std], 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCriticNonResNet(state_dim=4, action_dim=2).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def select_action(self, state_norm):
        """
        Selects an action using the fixed 'old' policy for collecting environment trajectories.
        """
        with torch.no_grad():
            state_t = torch.FloatTensor(state_norm).to(device).unsqueeze(0)
            
            # 1. Generate Intent 'z'
            z, log_prob, z_raw = self.policy_old.act(state_t)
            
        # The z_raw is returned specifically so it can be stored in the roll-out memory.
        # This is strictly required to recalculate log probabilities correctly during the PPO update.
        return z.cpu().numpy().flatten(), log_prob.cpu().numpy(), z_raw.cpu().numpy().flatten()

    def learn(self, memory):
        """
        Executes the Proximal Policy Optimization (PPO) training update.
        Calculates discounted advantages and updates the actor and critic networks.
        """
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal: discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Standardize baseline rewards to improve learning stability
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.squeeze(torch.stack(memory.states, dim=0)).detach().to(device)
        
        # CRITICAL REFINEMENT: The PPO objective is evaluated using the unbounded raw actions 
        # initially stored in memory, rather than the squashed [-1, 1] actions.
        old_z_raw = torch.squeeze(torch.stack(memory.raw_actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs, dim=0)).detach().to(device)

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_z_raw)
            
            # 1. PPO Loss
            ratios = torch.exp(logprobs - old_logprobs)
            advantages = rewards - state_values.detach().squeeze()

            surr1 = ratios * advantages
            
            # PPO Clipped Surrogate Objective: Restricts policy from changing too wildly in one epoch
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            ppo_loss = -torch.min(surr1, surr2).mean()

            # 2. Total Loss: PPO Policy Minimization + Critic Value (MSE) Minimization - Exploration (Entropy) Maximization
            loss = ppo_loss + \
                   0.5 * self.MseLoss(state_values.squeeze(), rewards) - \
                   self.entropy_coeff * dist_entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())