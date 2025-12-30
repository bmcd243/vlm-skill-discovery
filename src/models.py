import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- 1. Discriminator (For Mutual Information) ---
class Discriminator(nn.Module):
    def __init__(self, embedding_dim, num_skills):
        super().__init__()
        # Input: Text Embedding (from VLM)
        # Output: Logits for which skill generated this behavior
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_skills)
        )
    
    def forward(self, embeddings):
        return self.net(embeddings)

# --- 2. SAC Actor ---
class Actor(nn.Module):
    def __init__(self, state_shape, skill_dim, action_dim):
        super().__init__()
        flat_dim = np.prod(state_shape)
        input_dim = flat_dim + skill_dim  # Conditioned on Skill
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, state, skill):
        B = state.size(0)
        x = state.reshape(B, -1)
        x = torch.cat([x, skill], dim=1)
        x = self.net(x)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

    def sample(self, state, skill):
        mean, log_std = self(state, skill)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        return action

# --- 3. SAC Critic ---
class Critic(nn.Module):
    def __init__(self, state_shape, skill_dim, action_dim):
        super().__init__()
        flat_dim = np.prod(state_shape)
        input_dim = flat_dim + skill_dim + action_dim
        
        self.q1 = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, skill, action):
        B = state.size(0)
        x = state.reshape(B, -1)
        xu = torch.cat([x, skill, action], dim=1)
        return self.q1(xu), self.q2(xu)
