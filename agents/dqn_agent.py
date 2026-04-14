import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.dqn_network import DQNNetwork


class ReplayBuffer:
    def __init__(self, capacity: int = 20000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Double Dueling DQN with Huber loss.

    Key improvements over the vanilla DQN baseline:
      - Double DQN  : online net selects action, target net evaluates it
                      → removes overestimation bias (van Hasselt et al., 2016)
      - Dueling arch: separate value / advantage streams
                      → better policy evaluation when actions have similar Q-values
      - Huber loss  : linear for large errors, less sensitive to outliers than MSE
                      → prevents exploding loss when Q-values are large / negative
      - Fixed ε-decay: 0.99 per episode → reaches ε_min ≈ 0.05 by episode 300
                      (old decay 0.998 → ε = 0.82 after 100 eps, agent stays near-random)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.99,   # reaches min by ~ep 300
        epsilon_min: float = 0.05,
        buffer_capacity: int = 20000,
        target_update_freq: int = 50,
        hidden_dim: int = 128,
        dueling: bool = True,
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update_freq = target_update_freq
        self.train_steps = 0

        self.model = DQNNetwork(state_dim, action_dim, hidden_dim=hidden_dim, dueling=dueling)
        self.target_model = DQNNetwork(state_dim, action_dim, hidden_dim=hidden_dim, dueling=dueling)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()       # Huber loss

        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    # ------------------------------------------------------------------
    def choose_action(self, state) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_t)
        return torch.argmax(q_values, dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, float(done))

    # ------------------------------------------------------------------
    def train_step_batch(self, batch_size: int = 64) -> float:
        if len(self.replay_buffer) < batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        states_t      = torch.FloatTensor(states)
        actions_t     = torch.LongTensor(actions).unsqueeze(1)
        rewards_t     = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_t = torch.FloatTensor(next_states)
        dones_t       = torch.FloatTensor(dones).unsqueeze(1)

        # Current Q(s, a)
        current_q = self.model(states_t).gather(1, actions_t)

        with torch.no_grad():
            # Double DQN: online net picks action, target net scores it
            next_actions = self.model(next_states_t).argmax(dim=1, keepdim=True)
            next_q = self.target_model(next_states_t).gather(1, next_actions)
            target_q = rewards_t + (1.0 - dones_t) * self.gamma * next_q

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return loss.item()

    # ------------------------------------------------------------------
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def save(self, path: str):
        torch.save(
            {
                "model":        self.model.state_dict(),
                "target_model": self.target_model.state_dict(),
                "optimizer":    self.optimizer.state_dict(),
                "epsilon":      self.epsilon,
                "train_steps":  self.train_steps,
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        self.model.load_state_dict(ckpt["model"])
        self.target_model.load_state_dict(ckpt["target_model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt["epsilon"]
        self.train_steps = ckpt["train_steps"]
