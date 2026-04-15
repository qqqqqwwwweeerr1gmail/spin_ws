
import torch
import torch.nn as nn
from torch.distributions import Categorical
import gymnasium as gym
import numpy as np
import time

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ExperienceBuffer:
    """A buffer for storing and processing trajectory data for PPO."""
    def __init__(self, n_steps, n_envs, obs_dim, act_dim, gamma, gae_lambda):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Buffer storage
        self.obs = np.zeros((n_steps, n_envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((n_steps, n_envs, act_dim), dtype=np.float32)
        self.log_probs = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values = np.zeros((n_steps, n_envs), dtype=np.float32)

        self.step = 0

    def store(self, obs, action, log_prob, reward, done, value):
        """Store a single step of experience."""
        self.obs[self.step] = obs
        self.actions[self.step] = action
        self.log_probs[self.step] = log_prob
        self.rewards[self.step] = reward
        self.dones[self.step] = done
        self.values[self.step] = value
        self.step = (self.step + 1) % self.n_steps

    def compute_returns_and_advantages(self, last_value, done):
        """
        Post-processing: compute GAE and returns for the entire buffer.
        """
        advantages = np.zeros_like(self.rewards)
        last_gae_lam = 0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - done
                next_values = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_values = self.values[t + 1]

            delta = self.rewards[t] + self.gamma * next_values * next_non_terminal - self.values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam

        returns = advantages + self.values
        return returns, advantages

class Actor(nn.Module):
    """The policy network."""
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, act_dim)
        )

    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    """The value network."""
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

class PPOAgent:
    def __init__(self, env, **hyperparameters):
        self.env = env
        self.obs_dim = np.prod(env.observation_space.shape)
        self.act_dim = env.action_space.n # For discrete action space

        # --- Hyperparameters ---
        self.lr = hyperparameters.get('lr', 3e-4)
        self.gamma = hyperparameters.get('gamma', 0.99)
        self.n_steps = hyperparameters.get('n_steps', 2048)
        self.n_epochs = hyperparameters.get('n_epochs', 10)
        self.batch_size = hyperparameters.get('batch_size', 64)
        self.clip_epsilon = hyperparameters.get('clip_epsilon', 0.2)
        self.gae_lambda = hyperparameters.get('gae_lambda', 0.95)
        self.ent_coef = hyperparameters.get('ent_coef', 0.0) # Entropy bonus
        self.vf_coef = hyperparameters.get('vf_coef', 0.5) # Value function loss coefficient

        # --- Networks and Optimizer ---
        self.actor = Actor(self.obs_dim, self.act_dim).to(device)
        self.critic = Critic(self.obs_dim).to(device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.lr
        )

        # --- Experience Buffer ---
        self.buffer = ExperienceBuffer(self.n_steps, 1, self.obs_dim, 1, self.gamma, self.gae_lambda)

    def get_action_and_value(self, obs, action=None):
        """Get action from actor and value from critic."""
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
        if obs_tensor.dim() == 1: # Add batch dimension if missing
            obs_tensor = obs_tensor.unsqueeze(0)

        logits = self.actor(obs_tensor)
        value = self.critic(obs_tensor)

        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), value

    def update(self, last_obs, done):
        """Perform the PPO update."""
        with torch.no_grad():
            last_value = self.critic(torch.tensor(last_obs, dtype=torch.float32).to(device)).cpu().numpy().flatten()

        returns, advantages = self.buffer.compute_returns_and_advantages(last_value, done)

        # Flatten the batch for training
        b_obs = self.buffer.obs.reshape((-1, self.obs_dim))
        b_log_probs = self.buffer.log_probs.reshape(-1)
        b_actions = self.buffer.actions.reshape((-1, 1))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        # Normalize advantages (important for stability)
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # Convert to tensors
        b_obs = torch.tensor(b_obs).to(device)
        b_actions = torch.tensor(b_actions).to(device)
        b_log_probs = torch.tensor(b_log_probs).to(device)
        b_advantages = torch.tensor(b_advantages).to(device)
        b_returns = torch.tensor(b_returns).to(device)

        # --- Training Loop for Multiple Epochs ---
        inds = np.arange(self.n_steps)
        for epoch in range(self.n_epochs):
            np.random.shuffle(inds)
            for start in range(0, self.n_steps, self.batch_size):
                end = start + self.batch_size
                batch_inds = inds[start:end]

                # Get new log_probs, entropy, and values for the batch
                _, new_log_probs, entropy, new_values = self.get_action_and_value(
                    b_obs[batch_inds], b_actions.long()[batch_inds]
                )
                new_values = new_values.squeeze()

                # --- Policy (Actor) Loss ---
                # This is the core PPO clipping equation
                log_ratio = new_log_probs - b_log_probs[batch_inds]
                ratio = torch.exp(log_ratio)

                adv_batch = b_advantages[batch_inds]

                policy_loss_1 = adv_batch * ratio
                policy_loss_2 = adv_batch * torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # --- Value (Critic) Loss ---
                value_loss = 0.5 * ((new_values - b_returns[batch_inds]) ** 2).mean()

                # --- Entropy Loss ---
                entropy_loss = -entropy.mean()

                # --- Total Loss ---
                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                # --- Optimization Step ---
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer.step()


if __name__ == "__main__":
    # --- Environment Setup ---
    # Using gymnasium, the successor to gym
    env = gym.make("CartPole-v1")

    # --- Agent Initialization ---
    # Hyperparameters from the original PPO paper
    ppo_hyperparameters = {
        'lr': 3e-4,
        'gamma': 0.99,
        'n_steps': 512,
        'n_epochs': 10,
        'batch_size': 64,
        'clip_epsilon': 0.2,
        'gae_lambda': 0.95,
        'ent_coef': 0.0,
        'vf_coef': 0.5,
    }
    agent = PPOAgent(env, **ppo_hyperparameters)

    # --- Training ---
    total_timesteps = 50000
    obs, _ = env.reset()

    print(f"Starting training on {device}...")
    start_time = time.time()

    for global_step in range(total_timesteps):
        # --- Rollout/Collect Phase ---
        with torch.no_grad():
            action, log_prob, _, value = agent.get_action_and_value(obs)

        next_obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy().item())
        done = terminated or truncated

        # Store experience
        agent.buffer.store(obs, action.cpu().numpy(), log_prob.cpu().numpy(), reward, done,
                           value.cpu().numpy().flatten())

        # Update observation
        obs = next_obs
        if done:
            obs, _ = env.reset()

        # --- Update Phase ---
        if (global_step + 1) % agent.n_steps == 0:
            agent.update(obs, done)

            # --- Logging ---
            steps_per_second = int(agent.n_steps / (time.time() - start_time))
            print(f"Global Step: {global_step + 1}/{total_timesteps} | Steps/sec: {steps_per_second}")
            start_time = time.time()

    print("Training finished.")
    env.close()

    # To see the trained agent in action
    print("\n--- Running Trained Agent ---")
    test_env = gym.make("CartPole-v1", render_mode="human")
    obs, _ = test_env.reset()
    for _ in range(1000):
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs)
        obs, _, terminated, truncated, _ = test_env.step(action.cpu().numpy().item())
        if terminated or truncated:
            obs, _ = test_env.reset()
    test_env.close()


















