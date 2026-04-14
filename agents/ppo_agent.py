import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.ppo_network import PPOActorCritic


class PPOAgent:
    """
    Proximal Policy Optimization for discrete action spaces.

    Uses:
      - Generalized Advantage Estimation (GAE, Schulman et al., 2016)
      - Clipped surrogate objective
      - Shared backbone actor-critic
      - Advantage normalization per mini-batch update

    The loss reported per episode is the average over all mini-batch updates
    in that episode's update phase (policy + value + entropy combined).
    Unlike DQN loss, PPO loss does not monotonically decrease — focus on
    the reward trend for convergence assessment.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        mini_batch_size: int = 64,
        hidden_dim: int = 128,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.mini_batch_size = mini_batch_size

        self.network = PPOActorCritic(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

        # Rollout buffer — cleared after every update
        self._states:    list = []
        self._actions:   list = []
        self._log_probs: list = []
        self._rewards:   list = []
        self._values:    list = []
        self._dones:     list = []

    # ------------------------------------------------------------------
    def choose_action(self, state):
        """Returns (action, log_prob, value) — all Python scalars."""
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, _, value = self.network.get_action_and_value(state_t)
        return action.item(), log_prob.item(), value.item()

    def store_transition(self, state, action, log_prob, reward, value, done):
        self._states.append(state)
        self._actions.append(action)
        self._log_probs.append(log_prob)
        self._rewards.append(reward)
        self._values.append(value)
        self._dones.append(done)

    # ------------------------------------------------------------------
    def _compute_gae(self, last_value: float):
        """Compute returns and advantages via GAE."""
        rewards = np.array(self._rewards, dtype=np.float32)
        values  = np.array(self._values,  dtype=np.float32)
        dones   = np.array(self._dones,   dtype=np.float32)

        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(n)):
            next_val  = last_value if t == n - 1 else values[t + 1]
            next_done = 0.0       if t == n - 1 else dones[t + 1]
            delta = rewards[t] + self.gamma * next_val * (1.0 - next_done) - values[t]
            gae   = delta + self.gamma * self.gae_lambda * (1.0 - next_done) * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    # ------------------------------------------------------------------
    def update(self, last_value: float) -> float:
        """
        Run PPO update on the collected rollout, then clear the buffer.
        Returns: average combined loss over all mini-batch updates.
        """
        advantages, returns = self._compute_gae(last_value)

        states      = torch.FloatTensor(np.array(self._states))
        actions     = torch.LongTensor(np.array(self._actions))
        old_logprobs = torch.FloatTensor(np.array(self._log_probs))
        adv_t       = torch.FloatTensor(advantages)
        returns_t   = torch.FloatTensor(returns)

        n = len(self._states)
        total_loss = 0.0
        n_updates  = 0

        for _ in range(self.n_epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, self.mini_batch_size):
                mb_idx = idx[start : start + self.mini_batch_size]

                mb_states    = states[mb_idx]
                mb_actions   = actions[mb_idx]
                mb_old_lp    = old_logprobs[mb_idx]
                mb_adv       = adv_t[mb_idx]
                mb_returns   = returns_t[mb_idx]

                # Normalize advantages within mini-batch
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                _, new_lp, entropy, values = self.network.get_action_and_value(
                    mb_states, mb_actions
                )
                values = values.squeeze(-1)

                # Clipped surrogate policy loss
                ratio  = torch.exp(new_lp - mb_old_lp)
                surr1  = ratio * mb_adv
                surr2  = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values, mb_returns)

                # Entropy bonus (encourages exploration)
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                n_updates  += 1

        # Clear rollout buffer
        self._states.clear()
        self._actions.clear()
        self._log_probs.clear()
        self._rewards.clear()
        self._values.clear()
        self._dones.clear()

        return total_loss / n_updates if n_updates > 0 else 0.0

    def save(self, path: str):
        torch.save(self.network.state_dict(), path)

    def load(self, path: str):
        self.network.load_state_dict(torch.load(path, map_location="cpu"))
