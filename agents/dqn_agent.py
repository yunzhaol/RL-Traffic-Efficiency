import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.dqn_network import DQNNetwork


class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.0001, gamma=0.99, #lr was 0.001, changed to 0.0001
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.model = DQNNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    # def train_step(self, state, action, reward, next_state, done):
    #     state_tensor = torch.FloatTensor(state).unsqueeze(0)
    #     next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
    #     reward_tensor = torch.FloatTensor([reward])

    #     current_q = self.model(state_tensor)[0, action]

    #     with torch.no_grad():
    #         max_next_q = torch.max(self.model(next_state_tensor))
    #         target_q = reward_tensor if done else reward_tensor + self.gamma * max_next_q

    #     loss = self.loss_fn(current_q, target_q)

    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()

    #     return loss.item()
    def train_step(self, state, action, reward, next_state, done):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        reward_tensor = torch.FloatTensor([reward])

        current_q_values = self.model(state_tensor)
        current_q = current_q_values[0, action].unsqueeze(0)

        with torch.no_grad():
            next_q_values = self.model(next_state_tensor)
            max_next_q = torch.max(next_q_values, dim=1)[0]

            if done:
                target_q = reward_tensor
            else:
                target_q = reward_tensor + self.gamma * max_next_q

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min