import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x):
        return self.policy(x)

class ValueNet(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.value = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.value(x)

class PPOAgent:
    def __init__(self,
                 state_dim,
                 action_dim,
                 gamma=0.99,
                 lam=0.95,
                 clip_eps=0.2,
                 lr=3e-4,
                 train_iters=10,
                 minibatch_size=64,
                 device="cpu"): # <--- Parametru nou

        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.train_iters = train_iters
        self.minibatch_size = minibatch_size
        self.device = device # <--- Salvăm device-ul

        self.policy = PolicyNet(state_dim, action_dim).to(self.device)
        self.value_fn = ValueNet(state_dim).to(self.device)

        self.opt_policy = optim.Adam(self.policy.parameters(), lr=lr)
        self.opt_value = optim.Adam(self.value_fn.parameters(), lr=lr)

    def act(self, state):
        s = torch.tensor(state, dtype=torch.float32).to(self.device)

        logits = self.policy(s)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        action = dist.sample()
        logp = dist.log_prob(action)
        value = self.value_fn(s)

        return action.item(), logp.detach(), value.detach()

    def get_best_action(self, state):
        s = torch.tensor(state, dtype=torch.float32).to(self.device)
        logits = self.policy(s)
        return torch.argmax(logits).item()

    def compute_gae(self, rewards, values, next_value, dones):
        values = values + [next_value]
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, values[:-1])]

        return torch.tensor(advantages, dtype=torch.float32).to(self.device), \
               torch.tensor(returns, dtype=torch.float32).to(self.device)

    def train(self, obs, actions, logp_old, advantages, returns):
        dataset_size = len(obs)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.train_iters):
            idx = np.random.permutation(dataset_size)

            for start in range(0, dataset_size, self.minibatch_size):
                batch_idx = idx[start:start + self.minibatch_size]

                b_obs = obs[batch_idx]
                b_actions = actions[batch_idx]
                b_logp_old = logp_old[batch_idx]
                b_adv = advantages[batch_idx]
                b_returns = returns[batch_idx]

                logits = self.policy(b_obs)
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)

                new_logp = dist.log_prob(b_actions)
                ratio = torch.exp(new_logp - b_logp_old)
                clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)

                policy_loss = -torch.min(ratio * b_adv, clipped * b_adv).mean()

                value_pred = self.value_fn(b_obs).squeeze()
                value_loss = (b_returns - value_pred).pow(2).mean()

                self.opt_policy.zero_grad()
                policy_loss.backward()
                self.opt_policy.step()

                self.opt_value.zero_grad()
                value_loss.backward()
                self.opt_value.step()

    def save(self, filename):
        torch.save(self.policy.state_dict(), filename)

    def load(self, filename):
        self.policy.load_state_dict(torch.load(filename, map_location=self.device))