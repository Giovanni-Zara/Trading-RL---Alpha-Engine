"""
RL Agent for Alpha Expression Synthesis

Uses REINFORCE (Policy Gradient) algorithm.
Simple neural network that outputs action probabilities.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import List, Tuple, Optional


class PolicyNetwork(nn.Module):
    """
    Simple neural network that outputs action probabilities.
    
    Architecture:
        State → Linear → ReLU → Linear → Softmax → Action Probabilities
    
    The network learns which actions lead to high rewards.
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        """
        Parameters:
            state_size: Dimension of state vector
            action_size: Number of possible actions
            hidden_size: Number of neurons in hidden layer
        """
        super(PolicyNetwork, self).__init__()
        
        # Simple 2-layer network
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),   # Input layer
            nn.ReLU(),                             # Activation
            nn.Linear(hidden_size, hidden_size),  # Hidden layer
            nn.ReLU(),                             # Activation
            nn.Linear(hidden_size, action_size),  # Output layer
        )
    
    def forward(self, state: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass: state → action probabilities.
        
        Parameters:
            state: State vector [batch_size, state_size]
            action_mask: Binary mask of valid actions [batch_size, action_size]
                        1 = valid, 0 = invalid
        
        Returns:
            Action probabilities [batch_size, action_size]
        """
        # Get raw scores (logits)
        logits = self.network(state)
        
        # Apply mask: set invalid actions to very negative (so softmax → 0)
        if action_mask is not None:
            # Where mask is 0, set logits to -infinity
            logits = logits.masked_fill(action_mask == 0, float('-inf'))
        
        # Convert to probabilities
        probs = torch.softmax(logits, dim=-1)
        
        return probs


class REINFORCEAgent:
    """
    REINFORCE (Policy Gradient) agent.
    
    How it works:
    1. Collect a full episode (state, action, reward sequence)
    2. Compute returns (cumulative discounted rewards)
    3. Update policy to increase probability of good actions
    
    The key insight: actions that led to high rewards should become
    more likely; actions that led to low rewards should become less likely.
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int = 64,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        device: str = 'cpu'
    ):
        """
        Parameters:
            state_size: Dimension of state vector
            action_size: Number of possible actions
            hidden_size: Neurons in hidden layer
            learning_rate: How fast to learn
            gamma: Discount factor for future rewards
            device: 'cpu' or 'cuda'
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.device = torch.device(device)
        
        # Create policy network
        self.policy = PolicyNetwork(state_size, action_size, hidden_size)
        self.policy.to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Episode memory (cleared after each update)
        self.log_probs = []   # Log probabilities of chosen actions
        self.rewards = []     # Rewards received
    
    def select_action(self, state: np.ndarray, action_mask: np.ndarray) -> int:
        """
        Select an action given current state.
        
        Uses the policy network to get probabilities, then samples.
        
        Parameters:
            state: Current state vector
            action_mask: Binary mask of valid actions
            
        Returns:
            Selected action index
        """
        # Convert to tensors
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mask_t = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)
        
        # Get action probabilities
        probs = self.policy(state_t, mask_t)
        
        # Sample action from distribution
        dist = Categorical(probs)
        action = dist.sample()
        
        # Store log probability for training
        self.log_probs.append(dist.log_prob(action))
        
        return action.item()
    
    def store_reward(self, reward: float):
        """Store reward for current step."""
        self.rewards.append(reward)
    
    def update(self) -> float:
        """
        Update policy after episode ends.
        
        REINFORCE algorithm:
        1. Compute returns (discounted cumulative rewards)
        2. Compute policy loss = -log_prob * return
        3. Backpropagate and update weights
        
        Returns:
            Loss value (for logging)
        """
        if len(self.rewards) == 0:
            return 0.0
        
        # ----- Step 1: Compute returns -----
        returns = []
        G = 0
        
        # Go backwards through rewards
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize returns (helps stability)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # ----- Step 2: Compute policy loss -----
        policy_loss = []
        for log_prob, G in zip(self.log_probs, returns):
            # Negative because we want to MAXIMIZE reward
            policy_loss.append(-log_prob * G)
        
        # Sum up losses
        loss = torch.stack(policy_loss).sum()
        
        # ----- Step 3: Update network -----
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Clear episode memory
        self.log_probs = []
        self.rewards = []
        
        return loss.item()
    
    def save(self, path: str):
        """Save the policy network."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Agent saved to {path}")
    
    def load(self, path: str):
        """Load the policy network."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Agent loaded from {path}")

