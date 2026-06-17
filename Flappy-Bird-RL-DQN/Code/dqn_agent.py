# dqn_agent.py
"""
Deep Q-Network (DQN) agent.

Architecture
────────────
  state (STATE_SIZE)
    → Linear(128) → ReLU
    → Linear(128) → ReLU
    → Linear(ACTION_SIZE)         ← Q-values for each action

Training tricks
───────────────
  • Experience replay (uniform sampling from a circular buffer)
  • Target network (hard update every TARGET_UPDATE gradient steps)
  • ε-greedy exploration with exponential decay
  • Huber (smooth-L1) loss for stability
  • Gradient clipping (norm ≤ 10)

Typical training flow
─────────────────────
    agent = DQNAgent()
    state = env.reset()
    for step in range(MAX_STEPS):
        action               = agent.select_action(state)
        next_s, r, done, _   = env.step(action)
        agent.store(state, action, r, next_s, done)
        agent.learn()
        state = env.reset() if done else next_s
"""

import math
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    STATE_SIZE, ACTION_SIZE,
    GAMMA, LR, BATCH_SIZE, REPLAY_CAPACITY,
    EPS_START, EPS_END, EPS_DECAY, TARGET_UPDATE,
)


class QNetwork(nn.Module):
    """
    Simple two-hidden-layer MLP that maps a state vector to Q-values.
    Separate policy and target instances are used by DQNAgent.

    FIX (action bias): the output layer bias for action=1 (flap) is
    initialised to +0.1 so the agent starts with a slight preference for
    flapping rather than the "do nothing" local minimum that causes the
    bird to immediately fall to the ground in early training.
    """

    def __init__(self, state_size: int = STATE_SIZE,
                 action_size: int = ACTION_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )
        # Bias flap action slightly upward to break the do-nothing attractor
        with torch.no_grad():
            self.net[-1].bias[1] += 0.1   # index 1 = flap

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """
    Fixed-capacity experience buffer (circular / deque-backed).

    Stores (state, action, reward, next_state, done) tuples and
    provides random mini-batch sampling.
    """

    def __init__(self, capacity: int = REPLAY_CAPACITY):
        self._buf = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        self._buf.append((state, int(action), float(reward), next_state, bool(done)))

    def sample(self, batch_size: int):
        """Return a random mini-batch as stacked tensors."""
        batch                          = random.sample(self._buf, batch_size)
        states, acts, rews, nexts, ds  = zip(*batch)

        return (
            torch.tensor(np.array(states),  dtype=torch.float32),
            torch.tensor(acts,              dtype=torch.long).unsqueeze(1),
            torch.tensor(rews,              dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(nexts),   dtype=torch.float32),
            torch.tensor(ds,                dtype=torch.float32).unsqueeze(1),
        )

    def __len__(self) -> int:
        return len(self._buf)


class DQNAgent:
    """
    Full DQN agent: policy network, target network, replay buffer,
    ε-greedy policy, gradient updates, and checkpoint I/O.
    """

    def __init__(self, device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.policy_net = QNetwork().to(self.device)
        self.target_net = QNetwork().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer  = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.buffer     = ReplayBuffer()

        self.steps_done = 0     # total gradient steps (drives ε decay)
        self._last_loss  = None

    def select_action(self, state: np.ndarray,
                      greedy: bool = False) -> int:
        """
        Choose an action via ε-greedy policy.

        Parameters
        ----------
        state   : np.ndarray – current observation
        greedy  : bool       – if True, always choose the best Q-value action
                               (used during evaluation / watching the agent)

        Returns
        -------
        int – 0 (do nothing) or 1 (flap)
        """
        eps = EPS_END + (EPS_START - EPS_END) * math.exp(
            -self.steps_done / EPS_DECAY
        )

        if not greedy and random.random() < eps:
            return random.randrange(ACTION_SIZE)

        with torch.no_grad():
            t = torch.tensor(state, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
            return int(self.policy_net(t).argmax(dim=1).item())

    def store(self, state, action, reward, next_state, done):
        """Add one transition to the replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)

    def learn(self) -> float | None:
        """
        Sample a mini-batch and perform one gradient step.

        Returns
        -------
        float | None – the Huber loss value, or None if buffer is still
                       filling up (fewer than BATCH_SIZE transitions).
        """
        if len(self.buffer) < BATCH_SIZE:
            return None

        states, actions, rewards, next_states, dones = (
            t.to(self.device) for t in self.buffer.sample(BATCH_SIZE)
        )

        # ── Q(s, a) from policy net 
        q_values = self.policy_net(states).gather(1, actions)

        # ── Double-DQN target
        with torch.no_grad():
            best_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q       = self.target_net(next_states).gather(1, best_actions)
            targets      = rewards + GAMMA * next_q * (1.0 - dones)

        # ── Huber loss & optimiser step #
        loss = nn.functional.smooth_l1_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        self.steps_done += 1
        self._last_loss  = loss.item()

        # ── Hard-update target network #
        if self.steps_done % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return self._last_loss

    @property
    def epsilon(self) -> float:
        """Current exploration rate (for logging)."""
        return EPS_END + (EPS_START - EPS_END) * math.exp(
            -self.steps_done / EPS_DECAY
        )

    def save(self, path: str):
        """Save networks and training state to a .pth file."""
        torch.save({
            "policy":     self.policy_net.state_dict(),
            "target":     self.target_net.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "steps_done": self.steps_done,
        }, path)
        print(f"[DQN] Checkpoint saved -> {path}")

    def load(self, path: str):
        """Load networks and training state from a .pth file."""
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy"])
        self.target_net.load_state_dict(ckpt["target"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.steps_done = ckpt.get("steps_done", 0)
        print(f"[DQN] Checkpoint loaded <- {path}  (steps={self.steps_done})")