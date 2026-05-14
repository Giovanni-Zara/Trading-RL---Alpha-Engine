"""
RL Agent for Alpha Expression Synthesis

Uses PPO (Proximal Policy Optimization) algorithm.
Two networks: policy (actor) + value (critic).

Same interface as agent.py (REINFORCE) so they can be swapped easily.

Enhancements over vanilla PPO:
- GAE (Generalized Advantage Estimation) for smoother advantage targets
- Entropy regularization to prevent premature convergence
- Multi-episode batching: collects several episodes before each update
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import List, Tuple, Optional

# =============================================================================
# SECTION 1: ACTOR-CRITIC NETWORK
# =============================================================================

class ActorCriticNetwork(nn.Module):
    """
    Combined actor-critic network with shared feature extraction.
    
    Architecture:
        State → [Shared Layers] → Actor head → Action Probabilities
                                → Critic head → State Value
    
    Why two heads?
    - Actor (policy): decides WHICH action to take
    - Critic (value): estimates HOW GOOD the current state is
    
    The critic helps reduce variance compared to REINFORCE,
    because we can compute advantages: A = actual_return - predicted_value.
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        """
        Parameters:
            state_size: Dimension of state vector (8 in our case)
            action_size: Number of possible actions (27 in our case)
            hidden_size: Neurons in hidden layers
        """
        super(ActorCriticNetwork, self).__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )
        
        # Deeper actor head
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size),
        )
        
        # Deeper critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )
    
    def forward(
        self, 
        state: torch.Tensor, 
        action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: state → (action_probs, state_value).
        
        Parameters:
            state: State vector [batch_size, state_size]
            action_mask: Binary mask [batch_size, action_size], 1=valid
        
        Returns:
            probs: Action probabilities [batch_size, action_size]
            value: State value estimate [batch_size, 1]
        """
        # Shared features
        features = self.shared(state)
        
        # Actor: get action logits and mask invalid actions
        logits = self.actor(features)
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, float('-inf'))
        probs = torch.softmax(logits, dim=-1)
        
        # Critic: get state value
        value = self.critic(features)
        
        return probs, value


# =============================================================================
# SECTION 2: PPO AGENT
# =============================================================================

class PPOAgentEnhanced:
    """
    PPO (Proximal Policy Optimization) agent.
    
    Key differences from REINFORCE:
    1. Uses a value network (critic) to compute advantages
    2. Collects a batch of experiences, then does MULTIPLE updates on them
    3. Clips the policy ratio to prevent too-large updates
    
    Enhancements:
    - GAE (Generalized Advantage Estimation) with tunable lambda for
      bias-variance trade-off in advantage computation.
    - Entropy regularization to encourage exploration and prevent
      premature convergence on a narrow set of formulas.
    - Multi-episode batching: collects `batch_episodes` episodes of
      experience before performing a policy update, giving the optimizer
      a richer gradient signal per step.
    
    Same interface as REINFORCEAgent: (see otehr file in old.. folder)
        select_action(state, mask) → action
        store_reward(reward)
        store_done(done) → marks episode boundaries in the buffer
        update() → loss  (only performs the update when enough episodes
                          have been collected)
        should_update() → bool
        save(path) / load(path)
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int = 64,
        learning_rate: float = 0.0003,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        eps_clip: float = 0.2,
        k_epochs: int = 4,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        batch_episodes: int = 1, #NOTE: CHECK IF BATCH EPISODE SHOULD BE > 1 ( MAYBE 8 ?)
        device: str = 'cpu'
    ):
        """
        Parameters:
            state_size: Dimension of state vector
            action_size: Number of possible actions
            hidden_size: Neurons in hidden layers
            learning_rate: How fast to learn (lower than REINFORCE, PPO is more stable)
            gamma: Discount factor for future rewards
            gae_lambda: GAE lambda — controls bias-variance trade-off.
                        0 = high bias / low variance (pure TD),
                        1 = low bias / high variance (Monte Carlo-like).
                        Typical value: 0.95
            eps_clip: PPO clipping parameter (how much policy can change per update)
            k_epochs: Number of update passes on each batch of data
            value_coef: Weight for value loss (balances actor vs critic learning)
            entropy_coef: Weight for entropy bonus (encourages exploration).
                          Higher values → more exploration, prevents premature
                          convergence on a narrow set of formulas.
            batch_episodes: Number of episodes to collect before each update.
                            Larger batches give more stable gradients at the cost
                            of slower wall-clock updates.
            device: 'cpu' or 'cuda'
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.batch_episodes = batch_episodes
        self.device = torch.device(device)
        
        # Actor-Critic network
        self.policy = ActorCriticNetwork(state_size, action_size, hidden_size)
        self.policy.to(self.device)
        
        # Optimizer (single optimizer for both actor and critic)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Episode memory: stores full transitions across multiple episodes
        # Unlike REINFORCE which only stores log_probs and rewards,
        # PPO needs full transitions to recompute probabilities later.
        # `dones` marks episode boundaries so GAE can reset properly.
        self.states = []
        self.actions = []
        self.action_masks = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
        # Track how many complete episodes are in the buffer
        self._episodes_collected = 0
    
    def select_action(self, state: np.ndarray, action_mask: np.ndarray) -> int:
        """
        Select an action given current state.
        
        Same interface as REINFORCEAgent.select_action.
        
        Parameters:
            state: Current state vector
            action_mask: Binary mask of valid actions
            
        Returns:
            Selected action index
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mask_t = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)
        
        # No gradient needed during action selection
        with torch.no_grad():
            probs, value = self.policy(state_t, mask_t)
        
        # Sample action
        dist = Categorical(probs)
        action = dist.sample()
        
        # Store transition data for later update
        self.states.append(state)
        self.actions.append(action.item())
        self.action_masks.append(action_mask)
        self.log_probs.append(dist.log_prob(action).item())
        self.values.append(value.item())
        
        return action.item()
    
    def store_reward(self, reward: float):
        """Store reward for current step. Same interface as REINFORCE."""
        self.rewards.append(reward)
    
    def store_done(self, done: bool):
        """
        Store the done flag for the current step.
        
        This marks episode boundaries inside the buffer, which is needed
        for correct GAE computation across multiple episodes: the advantage
        must not bleed across episode boundaries.
        
        Parameters:
            done: True if this step ends the episode, False otherwise
        """
        self.dones.append(done)
        if done:
            self._episodes_collected += 1
    
    def should_update(self) -> bool:
        """
        Check whether enough episodes have been collected to run an update.
        
        Returns:
            True if `batch_episodes` complete episodes are in the buffer.
        """
        return self._episodes_collected >= self.batch_episodes
    
    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        GAE smoothly interpolates between TD(0) (high bias, low variance)
        and Monte Carlo (low bias, high variance) via the lambda parameter.
        
        For each timestep t:
            delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
            A_t     = sum_{l=0}^{T-t} (gamma * lambda)^l * delta_l
        
        Parameters:
            rewards: [T] tensor of rewards
            values:  [T] tensor of value estimates V(s_t)
            dones:   [T] tensor of done flags (1.0 = episode ended)
        
        Returns:
            advantages: [T] tensor of GAE advantages
            returns:    [T] tensor of target returns (advantages + values)
        """
        T = len(rewards)
        advantages = torch.zeros(T, device=self.device)
        gae = 0.0
        
        for t in reversed(range(T)):
            # If this is the last timestep or episode ended here,
            # there is no next value to bootstrap from
            if t == T - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1].item()
            
            # Reset GAE accumulator at episode boundaries
            # (1 - done_t) ensures we don't bootstrap across episodes
            not_done = 1.0 - dones[t].item()
            
            # TD residual: delta = r + gamma * V(s') * (1 - done) - V(s)
            delta = rewards[t] + self.gamma * next_value * not_done - values[t]
            
            # GAE recursive formula: A_t = delta_t + gamma * lambda * (1-done) * A_{t+1}
            gae = delta + self.gamma * self.gae_lambda * not_done * gae
            advantages[t] = gae
        
        # Returns = advantages + values  (used as critic targets)
        returns = advantages + values
        
        return advantages, returns
    
    def update(self) -> float:
        """
        Update policy using PPO algorithm with GAE.
        
        Same interface as REINFORCEAgent.update.
        
        PPO steps:
        1. Compute GAE advantages and returns from collected episodes
        2. For K epochs:
           a. Recompute action probabilities with current network
           b. Compute ratio = new_prob / old_prob
           c. Clip the ratio to prevent too-large updates
           d. Compute entropy bonus to encourage exploration
           e. Update actor (policy) and critic (value) together
        
        Returns:
            Average loss over all epochs (for logging).
            Returns 0.0 if buffer is empty or not enough episodes collected.
        """
        if len(self.rewards) == 0:
            return 0.0
        
        # =====================================================================
        # STEP 1: Prepare data from collected episodes
        # =====================================================================
        
        # Convert everything to tensors
        states_t = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions_t = torch.LongTensor(self.actions).to(self.device)
        masks_t = torch.FloatTensor(np.array(self.action_masks)).to(self.device)
        old_log_probs_t = torch.FloatTensor(self.log_probs).to(self.device)
        rewards_t = torch.FloatTensor(self.rewards).to(self.device)
        values_t = torch.FloatTensor(self.values).to(self.device)
        dones_t = torch.FloatTensor(self.dones).to(self.device)
        
        # Compute GAE advantages and returns
        advantages, returns_t = self._compute_gae(rewards_t, values_t, dones_t)
        
        # Normalize advantages (stabilizes training)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # =====================================================================
        # STEP 2: Multiple update epochs on the same data
        # This is what makes PPO more sample-efficient than REINFORCE
        # =====================================================================
        
        total_loss = 0.0
        
        for _ in range(self.k_epochs):
            # Get current policy's probabilities and values
            probs, values = self.policy(states_t, masks_t)
            values = values.squeeze(-1)
            
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(actions_t)
            entropy = dist.entropy()
            
            # ----- Policy (Actor) Loss -----
            
            # Importance ratio: how much has the policy changed?
            # ratio > 1: action became MORE likely
            # ratio < 1: action became LESS likely
            ratio = torch.exp(new_log_probs - old_log_probs_t)
            
            # Two surrogate objectives
            surr1 = ratio * advantages                # Unclipped
            surr2 = torch.clamp(                      # Clipped
                ratio, 
                1.0 - self.eps_clip,                  # Lower bound (e.g. 0.8)
                1.0 + self.eps_clip                   # Upper bound (e.g. 1.2)
            ) * advantages
            
            # Take the MINIMUM (pessimistic bound)
            # This prevents the policy from changing too much
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # ----- Value (Critic) Loss -----
            # Simple MSE: how well does the critic predict returns?
            value_loss = nn.functional.mse_loss(values, returns_t)
            
            # ----- Entropy Bonus -----
            # Encourages exploration by penalizing overly confident policies.
            # Without this, the agent might converge prematurely on a narrow
            # set of formulas and never discover better alpha expressions.
            # Higher entropy_coef → stronger push toward uniform action
            # distribution → more diverse formula exploration.
            entropy_bonus = entropy.mean()
            
            # ----- Total Loss -----
            loss = (
                policy_loss                            # Actor learning
                + self.value_coef * value_loss         # Critic learning
                - self.entropy_coef * entropy_bonus    # Exploration bonus
            )
            
            # Update network
            self.optimizer.zero_grad()
            loss.backward()
            # Gradient clipping (prevents exploding gradients)
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        # Clear episode memory
        self.states = []
        self.actions = []
        self.action_masks = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self._episodes_collected = 0
        
        return total_loss / self.k_epochs
    
    def save(self, path: str):
        """Save the actor-critic network."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Agent saved to {path}")
    
    def load(self, path: str):
        """Load the actor-critic network."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Agent loaded from {path}")