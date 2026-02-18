from collections import deque
import random
import numpy as np


class OpenArmedBandit:
    def __init__(self, reward_prob: float):
        self.reward_prob = reward_prob

    def pull(self) -> int:
        return 1 if random.random() < self.reward_prob else 0


class EpsilonGreedyPolicy:
    def __init__(self, epsilon: float, decay: float = 1.0):
        self.epsilon = epsilon
        self.decay = decay

    def select_arm(self, n_arms: int, estimates: list) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, n_arms - 1)  # Explore
        else:
            return int(max(range(n_arms), key=lambda i: estimates[i]))  # Greedy

    def after_step(self):
        """Decay epsilon after each agent step (only when decay < 1)."""
        if self.decay < 1.0:
            self.epsilon *= self.decay


class ThompsonSamplingPolicy:
    def __init__(self, n_arms: int):
        self.successes = [0] * n_arms
        self.failures = [0] * n_arms

    def select_arm(self, n_arms: int, estimates: list) -> int:
        import numpy as np
        sampled = [np.random.beta(s + 1, f + 1)
                   for s, f in zip(self.successes, self.failures)]
        return int(max(range(n_arms), key=lambda i: sampled[i]))

    def update(self, arm: int, reward: int):
        if reward == 1:
            self.successes[arm] += 1
        else:
            self.failures[arm] += 1

    def after_step(self):
        pass  # No decay for Thompson Sampling


class Agent:
    def __init__(self, n_arms: int, epsilon: float = 0.1, decay: float = 1.0,
                 memory: int = 0, method: str = "Epsilon Greedy"):
        self.n_arms = n_arms
        self.memory = memory
        self.pulls = [0] * n_arms
        self.successes = [0] * n_arms
        # Per-arm deque; maxlen=None means unlimited (use all history)
        maxlen = memory if memory > 0 else None
        self.history = {i: deque(maxlen=maxlen) for i in range(n_arms)}

        if method == "Thompson Sampling":
            self.policy = ThompsonSamplingPolicy(n_arms)
        else:
            self.policy = EpsilonGreedyPolicy(epsilon, decay)

    def _estimates(self) -> list:
        """Return empirical mean per arm based on history (respects memory)."""
        estimates = []
        for i in range(self.n_arms):
            h = self.history[i]
            if len(h) == 0:
                estimates.append(0.0)
            else:
                estimates.append(sum(h) / len(h))
        return estimates

    def select_action(self) -> int:
        return self.policy.select_arm(self.n_arms, self._estimates())

    def update(self, arm: int, reward: int) -> None:
        self.pulls[arm] += 1
        self.successes[arm] += reward
        self.history[arm].append(reward)
        # Sync Thompson Sampling policy counts
        if isinstance(self.policy, ThompsonSamplingPolicy):
            self.policy.update(arm, reward)
        self.policy.after_step()

    def run(self, envs: list, n_loops: int) -> list:
        rewards = []
        for _ in range(n_loops):
            arm = self.select_action()
            reward = envs[arm].pull()
            self.update(arm, reward)
            rewards.append(reward)
        return rewards

    def get_stats(self) -> dict:
        return {
            'pulls': list(self.pulls),
            'successes': list(self.successes),
            'cumulative_rewards': sum(self.successes),
            'success_rates': [s / p if p > 0 else 0.0
                              for s, p in zip(self.successes, self.pulls)],
        }
