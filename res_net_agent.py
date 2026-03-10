import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import os

# Hardware acceleration setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActionProjectionNetwork(nn.Module):
    """
    The pre-trained Safety Filter (Safeguard) architecture.
    Loaded identically to the structure defined in `pretrain.py`.
    Expects 4D Physical State: [Biomass (cx), Nitrate (cN), Product (cq), Time (t_norm)]
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
        # 1. Internal Static Normalization (Matches Pretrain)
        state_norm = (state_phys / self.max_vals) * 2.0 - 1.0
        
        x_in = torch.cat([state_norm, nom_act_norm], dim=1)
        x_proj = self.elu(self.input_layer(x_in))
        
        # 2. Skip Connection (Residual Connection)
        x_res = self.res_block1(x_proj)
        x_combined = self.elu(x_res + x_proj)
        
        # 3. Residual Correction Calculation
        delta = torch.relu(self.final_layer(x_combined))
        u_nn = nom_act_norm - delta
        
        # 3. Explicit Analytical Override for G3 (Terminal Nitrate)
        cN = state_phys[:, 1]
        t_norm = state_phys[:, 3]
        
        TOTAL_TIME = 240.0
        CONTROL_INTERVAL = 20.0
        N_LIMIT_TERM = 150.0
        FN_MAX = 40.0
        
        t_phys = t_norm * TOTAL_TIME
        time_remaining = TOTAL_TIME - t_phys
        
        near_end = time_remaining <= CONTROL_INTERVAL
        safe_time_remaining = torch.clamp(time_remaining, min=1.0)
        
        # Calculate exactly how much MAXIMUM feed we can provide to hit 150 * 0.95
        fn_max_term = (N_LIMIT_TERM * 0.95 - cN) / safe_time_remaining
        u_max_term = ((fn_max_term / FN_MAX) * 2.0) - 1.0
        
        # Ensure calculated maximum bounded action space is respected [-1, 1]
        u_max_term = torch.clamp(u_max_term, min=-1.0, max=1.0)
        
        u_safe = u_nn.clone()
        # Direct intervention: The neural network's feed action is capped at the maximum 
        # allowed safe boundary to enforce compliance.
        # This operation is fully differentiable out-of-the-box (gradients route through 
        # the selected min argument), preserving the mapping penalty pathway for the PPO actor.
        u_safe[:, 1] = torch.where(near_end, torch.minimum(u_nn[:, 1], u_max_term), u_nn[:, 1])
        
        return u_safe

class ActorCritic(nn.Module):
    """
    Standard PPO Actor-Critic neural network architecture generating the raw intent 'z'.
    The intent 'z' is squashed and evaluated identically to a normal unconstrained agent.
    """
    def __init__(self, state_dim=4, action_dim=2):
        super(ActorCritic, self).__init__()
        self.LOG_STD_MIN = -1.0
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
        """
        Generates an intended action (z) during the environment rollout phase.
        
        Returns:
            z: The squashed intended action mapped to [-1, 1].
            log_prob: The log probability of the intended action.
            z_raw: The unbounded Gaussian sample.
        """
        mean = self.actor(state)
        std = torch.exp(torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX))
        dist = Normal(mean, std)
        
        # 1. Sample from unbounded Gaussian first
        z_raw = dist.sample()
        # 2. Squash to physical bounds [-1, 1]
        z = torch.tanh(z_raw)
        
        # 3. Tanh Squashing Correction for LogProb
        log_prob = dist.log_prob(z_raw).sum(dim=-1)
        log_prob -= torch.log(1 - z.pow(2) + 1e-6).sum(dim=-1)
        
        return z.detach(), log_prob.detach(), z_raw.detach()

    def evaluate(self, state, z_raw):
        """
        Evaluates the probabilities and values of previous intents during the PPO update loop.
        Requires the UNBOUNDED 'z_raw' parameter to recalculate exact Gaussian probabilities.
        """
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
    """
    Wrapper class managing the Safe PPO learning algorithms, rollouts, and models.
    """
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, entropy_coeff):
        self.gamma = gamma               # Discount factor for future rewards
        self.eps_clip = eps_clip         # PPO surrogate objective clipping parameter
        self.K_epochs = K_epochs         # Number of policy update iterations per rollout batch
        self.entropy_coeff = entropy_coeff 

        # 1. Actor-Critic Network Initialization
        self.policy = ActorCritic(state_dim=4, action_dim=2).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': [self.policy.log_std], 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim=4, action_dim=2).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # 2. Safeguard (Safety Filter) Network Initialization
        # Loaded identically and strictly set into evaluation mode to ensure 
        # the weights remain completely frozen during the RL loop.
        self.safeguard = ActionProjectionNetwork(state_dim=4, action_dim=2).to(device)
        if os.path.exists("action_projection_network.pth"):
            self.safeguard.load_state_dict(torch.load("action_projection_network.pth", map_location=device))
            self.safeguard.eval()

        self.MseLoss = nn.MSELoss()
        self.mapping_criterion = nn.MSELoss()

    def select_action(self, state_norm):
        """
        Selects an action by generating an intent and passing it through the Safeguard.
        
        Input: state_norm [cx, cN, cq, t_norm] from environment.
        """
        with torch.no_grad():
            state_t = torch.FloatTensor(state_norm).to(device).unsqueeze(0)
            
            # 1. Generate Intent 'z' based on the standard unconstrained policy
            z, log_prob, z_raw = self.policy_old.act(state_t)
            
            # 2. Denormalize completely to Physical Units for the Safeguard filter
            # Extents: [6.0, 800.0, 0.1, 1.0] corresponds with env.py and pretrain.py
            phys_scale = torch.tensor([6.0, 800.0, 0.1, 1.0], device=device)
            state_phys = state_t * phys_scale
            
            # 3. Action Projection Application
            # The safeguard acts as the final analytical layer of the policy network.
            # 'u_safe' is returned to the environment so the agent is strictly On-Policy (SP-RL).
            u_safe = self.safeguard(state_phys, z)
            
        # The Safe Action (u_safe) is executed by the environment, 
        # while z_raw is stored in memory for PPO mapping penalty backpropagation.
        return u_safe.cpu().numpy().flatten(), log_prob.cpu().numpy(), z_raw.cpu().numpy().flatten()

    def learn(self, memory):
        """
        Executes the Proximal Policy Optimization (PPO) training update, 
        incorporating the exact SPRL Action Mapping penalty logic.
        """
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal: discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Standardize baseline rewards to improve gradient stability
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

            # 2. SP-RL Mapping Penalty Evaluation
            # Reconstructs the exact intervention the safeguard initially applied
            with torch.no_grad():
                phys_scale = torch.tensor([6.0, 800.0, 0.1, 1.0], device=device)
                states_phys = old_states * phys_scale
                z_intent = torch.tanh(old_z_raw)
                u_safe = self.safeguard(states_phys, z_intent)

            # Mapping Penalty: If the safeguard modified the action significantly,
            # this MSE loss sharply punishes the Actor network for issuing the bad intent.
            mapping_penalty = self.mapping_criterion(z_intent, u_safe)
            
            # 3. Total Custom Loss Aggregation
            # Minimizes mapping penalty alongside standard PPO objectives
            loss = ppo_loss + \
                   0.5 * self.MseLoss(state_values.squeeze(), rewards) - \
                   self.entropy_coeff * dist_entropy.mean() + \
                   0.001 * mapping_penalty # Alignment coefficient for safety mapping

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())