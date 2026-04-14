import torch
import torch.nn as nn
from torch.distributions import Categorical


class PPOActorCritic(nn.Module):
    """
    Shared-backbone Actor-Critic for discrete-action PPO.

    Architecture:
        shared backbone (Tanh activations — standard for PPO per SB3/CleanRL)
            ├── policy head  → logits over action_dim
            └── value head   → scalar V(s)

    Orthogonal initialization with the gains recommended by Andrychowicz et al.
    (2021) "What Matters In On-Policy Reinforcement Learning".
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        tanh_gain = nn.init.calculate_gain("tanh")
        for m in self.backbone.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=tanh_gain)
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.constant_(self.policy_head.bias, 0.0)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.value_head.bias, 0.0)

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        return self.policy_head(features), self.value_head(features)

    def get_action_and_value(self, state: torch.Tensor, action: torch.Tensor = None):
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        _, value = self.forward(state)
        return value
