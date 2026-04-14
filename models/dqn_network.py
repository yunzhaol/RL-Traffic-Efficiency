import torch
import torch.nn as nn


class DQNNetwork(nn.Module):
    """
    Dueling DQN network (Wang et al., 2016).
    Splits into Value and Advantage streams so the network can learn
    state value independently from per-action advantages — leads to
    more stable Q-value estimates, especially when many actions have
    similar values (common in 2-phase traffic control).

    dueling=False falls back to a plain MLP for ablation.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128, dueling: bool = True):
        super().__init__()
        self.dueling = dueling

        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        if dueling:
            self.value_stream = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
            self.advantage_stream = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim),
            )
        else:
            self.output_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if self.dueling:
            value = self.value_stream(features)                        # (B, 1)
            advantage = self.advantage_stream(features)                # (B, A)
            # Q(s,a) = V(s) + A(s,a) - mean_a[A(s,a)]
            return value + advantage - advantage.mean(dim=1, keepdim=True)
        return self.output_layer(features)
