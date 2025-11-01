import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Independent

from optimizer import AdamWC


class PPOMemory:

    def __init__(
        self,
        batch_size: int,
    ):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i : i + self.batch_size] for i in batch_start]

        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.probs),
            np.array(self.vals),
            np.array(self.rewards),
            np.array(self.dones),
            batches,
        )

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []


class ActorNetwork(nn.Module):

    def __init__(
        self,
        n_actions: int,
        input_dims: int,
        alpha: float,
        optimizer_name: str = "Adam",
        fc1_dims: int = 256,
        fc2_dims: int = 256,
        chkpt_dir: str = "tmp/ppo",
    ):

        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, "actor_torch_ppo")

        self.body = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.tanh(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.tanh(),
        )
        self.mu_head = nn.Linear(fc2_dims, n_actions)
        self.logstd_head = nn.Linear(fc2_dims, n_actions)

        if optimizer_name == "Adam":
            self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        elif optimizer_name == "AdamWC":
            self.optimizer = AdamWC(self.parameters(), lr=alpha)
        else:
            raise ValueError(f"No optimizer named {optimizer_name}\navailable optimizers are Adam and AdamWC")

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):

        x = self.body(state)
        mu = self.mu_head(x)
        log_std = self.logstd_head(x).clamp(-5, 2)
        std = log_std.exp()
        dist = Independent(Normal(mu, std), 1)

        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class CriticNetwork(nn.Module):

    def __init__(
        self,
        input_dims: int,
        alpha: float,
        optimizer_name: str = "Adam",
        fc1_dims: int = 256,
        fc2_dims: int = 256,
        chkpt_dir: str = "tmp/ppo",
    ):

        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, "critic_torch_ppo")
        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.tanh(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.tanh(),
            nn.Linear(fc2_dims, 1),
        )

        if optimizer_name == "Adam":
            self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        elif optimizer_name == "AdamWC":
            self.optimizer = AdamWC(self.parameters(), lr=alpha)
        else:
            raise ValueError(f"No optimizer named {optimizer_name}\navailable optimizers are Adam and AdamWC")

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)

        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class Agent:

    def __init__(
        self,
        n_actions: int,
        input_dims: int,
        optimizer_name: str,
        gamma: float = 0.99,
        alpha: float = 0.0003,
        gae_lambda: float = 0.95,
        policy_clip: float = 0.2,
        critic_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        batch_size: int = 64,
        n_epochs: int = 10,
        norm_advantage: bool = True,
        clip_value_loss: bool = True,
    ):

        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.norm_advantage = norm_advantage
        self.clip_value_loss = clip_value_loss
        self.critic_coef = critic_coef
        self.max_grad_norm = max_grad_norm

        self.actor = ActorNetwork(
            n_actions,
            input_dims,
            optimizer_name,
            alpha
        )
        self.critic = CriticNetwork(
            input_dims,
            optimizer_name,
            alpha
        )
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print("... saving models ...")
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print("... loading models ...")
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        state = T.tensor(observation, dtype=T.float32, device=self.actor.device)

        dist = self.actor(state)
        value = self.critic(state)

        action = dist.sample()
        logp = dist.log_prob(action)

        action = action.squeeze(0).detach().cpu().numpy()
        logp = logp.squeeze(0).item()
        value = value.squeeze(-1).squeeze(0).item()

        return action, logp, value

    def learn(self):

        for _ in range(self.n_epochs):
            (
                state_arr,
                action_arr,
                old_prob_arr,
                vals_arr,
                reward_arr,
                dones_arr,
                batches,
            ) = self.memory.generate_batches()

            values = T.tensor(vals_arr, dtype=T.float32, device=self.actor.device)
            advantages = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (
                        reward_arr[k]
                        + self.gamma * values[k + 1] * (1 - int(dones_arr[k]))
                        - values[k]
                    )
                    discount *= self.gamma * self.gae_lambda
                advantages[t] = a_t
            advantages = T.tensor(advantages, dtype=T.float32, device=self.actor.device)
            if self.norm_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            values = values.detach().clone().requires_grad_(True).to(self.actor.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float32).to(
                    self.actor.device
                )
                old_logp = T.tensor(old_prob_arr[batch], dtype=T.float32).to(
                    self.actor.device
                )
                actions = T.tensor(action_arr[batch], dtype=T.float32).to(
                    self.actor.device
                )

                dist = self.actor(states)
                critic_value = self.critic(states).squeeze(-1)
                critic_value = T.squeeze(critic_value)

                new_logp = dist.log_prob(actions)
                ratio = T.exp(new_logp - old_logp)
                surr1 = ratio * advantages[batch]
                surr2 = (
                    T.clamp(ratio, 1.0 - self.policy_clip, 1.0 + self.policy_clip)
                    * advantages[batch]
                )
                actor_loss = -T.min(surr1, surr2).mean()

                # use the same clip with policy
                value_clip = self.policy_clip
                returns = advantages[batch] + values[batch]
                if clip_value_loss:
                    unclipped_critic_loss = (returns - critic_value).pow(2).mean()
                    clipped_critic_value = values[batch] + T.clamp(
                        critic_value - values[batch],
                        -value_clip,
                        value_clip,
                    )
                    clipped_critic_loss = (clipped_critic_value).pow(2).mean()
                    max_critic_loss = T.max(
                        unclipped_critic_loss,
                        clipped_critic_loss,
                    )
                    critic_loss = 0.5 * max_critic_loss
                else:
                    critic_loss = 0.5 * (returns - critic_value).pow(2).mean()

                total_loss = actor_loss + self.critic_coef * critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()
