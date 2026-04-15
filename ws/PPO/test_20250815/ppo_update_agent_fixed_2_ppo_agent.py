import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import time
import torch.nn.functional as F

# -- MLPAgent --

class MLPAgent(nn.Module):
    # def __init__(self, obs_dim, act_dim):
    #     super().__init__()
    #     self.net = nn.Sequential(
    #         nn.Linear(obs_dim, 64),
    #         nn.ReLU(),
    #         nn.Linear(64, 64),
    #         nn.ReLU()
    #     )
    #     self.policy = nn.Linear(64, act_dim)
    #     self.value = nn.Linear(64, 1)

    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.policy = nn.Sequential(
            nn.Linear(32, 24),
            nn.ReLU(),
            nn.Linear(24, act_dim)
        )
        self.value = nn.Sequential(
            nn.Linear(32, 1)
        )

    def get_action(self, obs):
        x = self.net(obs)
        logits = self.policy(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)             # scalar per sample
        value = self.value(x).squeeze(-1)            # scalar value
        return action.item(), log_prob, value

    def evaluate_actions(self, obs, actions):
        x = self.net(obs)
        logits = self.policy(x)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(actions)            # shape [batch]
        entropy = dist.entropy()                     # shape [batch]
        value = self.value(x).squeeze(-1)            # shape [batch]
        return log_prob, entropy, value

    def get_value(self, obs):
        x = self.net(obs)
        value = self.value(x).squeeze(-1)
        return value

def save_agent(agent, filepath):
    torch.save(agent.state_dict(), filepath)

def load_agent(filepath, obs_dim, act_dim):
    agent = MLPAgent(obs_dim, act_dim)
    agent.load_state_dict(torch.load(filepath, map_location="cpu"))
    return agent

# -- PPO class (Version 1) --

class PPO:
    def __init__(self, mlp, optimizer, gamma=0.99, lam=0.95, clip_eps=0.2,
                 update_iters=10, batch_size=64, ent_coef=0.01):
        self.mlp = mlp
        self.optimizer = optimizer
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.update_iters = update_iters
        self.batch_size = batch_size
        self.ent_coef = ent_coef

        self.obs_buf = []
        self.actions_buf = []
        self.rewards_buf = []
        self.dones_buf = []
        self.values_buf = []     # will be length T+1 after appending bootstrap
        self.log_probs_buf = []

    def add(self, obs, action, reward, done, value, log_prob):
        self.obs_buf.append(obs)
        self.actions_buf.append(action)
        self.rewards_buf.append(reward)
        self.dones_buf.append(done)
        self.values_buf.append(value)
        self.log_probs_buf.append(log_prob)

    def clear(self):
        self.obs_buf = []
        self.actions_buf = []
        self.rewards_buf = []
        self.dones_buf = []
        self.values_buf = []
        self.log_probs_buf = []

    def update(self, ent_coef=None):
        if len(self.rewards_buf) == 0:
            print("No data in buffers, skipping PPO update.")
            return
        current_ent_coef = self.ent_coef if ent_coef is None else ent_coef

        # numpy arrays
        rewards = np.array(self.rewards_buf, dtype=np.float32)   # [T]
        dones   = np.array(self.dones_buf,   dtype=np.float32)   # [T]
        values  = np.array(self.values_buf,  dtype=np.float32)   # [T+1] (bootstrap appended)
        log_probs = np.array(self.log_probs_buf, dtype=np.float32)  # [T]
        obs_buf = np.array(self.obs_buf, dtype=np.float32)       # [T, obs_dim]
        actions_buf = np.array(self.actions_buf, dtype=np.int64) # [T]

        T = len(rewards)
        assert len(values) == T + 1, "values must be length T+1 (include bootstrap V_last)."

        # GAE with explicit V_{t+1}
        advantages = np.zeros(T, dtype=np.float32)
        returns = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * values[t + 1] * (1.0 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1.0 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = gae + values[t]

        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # tensors
        obs_tensor = torch.as_tensor(obs_buf, dtype=torch.float32)
        actions_tensor = torch.as_tensor(actions_buf, dtype=torch.int64)
        old_log_probs_tensor = torch.as_tensor(log_probs, dtype=torch.float32)
        returns_tensor = torch.as_tensor(returns, dtype=torch.float32)
        adv_tensor = torch.as_tensor(advantages, dtype=torch.float32)

        dataset = torch.utils.data.TensorDataset(obs_tensor, actions_tensor,
                                                 old_log_probs_tensor, returns_tensor, adv_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for _ in range(self.update_iters):
            for obs_b, act_b, old_logp_b, ret_b, adv_b in loader:
                logp, entropy, value = self.mlp.evaluate_actions(obs_b, act_b)

                ratio = torch.exp(logp - old_logp_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(value, ret_b)
                loss = policy_loss + 0.5 * value_loss - current_ent_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.mlp.parameters(), 0.5)
                self.optimizer.step()

        self.clear()

# -- Training using PPO class --

def ppo_train(env, agent, epochs=2, steps_per_epoch=2048,
              gamma=0.99, lam=0.95, clip_eps=0.2, update_iters=10,
              batch_size=64, lr=3e-4, ent_coef=0.01, log_path="train_log.json"):

    optimizer = optim.Adam(agent.parameters(), lr=lr)
    ppo = PPO(agent, optimizer, gamma=gamma, lam=lam, clip_eps=clip_eps,
              update_iters=update_iters, batch_size=batch_size, ent_coef=ent_coef)

    training_log = {"epochs": [], "policy_loss": [], "value_loss": [], "total_loss": [], "episodes": [], "step_sars": []}

    for epoch in range(epochs):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        obs = torch.as_tensor(obs, dtype=torch.float32)

        ep_ret = 0
        ep_rewards = []
        done = False

        for t in range(steps_per_epoch):
            with torch.no_grad():
                action, log_prob, value = agent.get_action(obs)
            step_out = env.step(action)
            if len(step_out) == 5:
                next_obs, reward, terminated, truncated, *_ = step_out
                done = terminated or truncated
            else:
                next_obs, reward, done, *_ = step_out

            training_log["step_sars"].append({
                "step": epoch * steps_per_epoch + t,
                "state": obs.numpy().tolist(),
                "action": int(action),
                "reward": float(reward),
                "done": bool(done)
            })

            ppo.add(obs.numpy(), action, reward, int(done), value.item(), log_prob.item())

            obs = torch.as_tensor(next_obs if not isinstance(next_obs, tuple) else next_obs[0], dtype=torch.float32)
            ep_ret += reward

            if done:
                obs = env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]
                obs = torch.as_tensor(obs, dtype=torch.float32)
                ep_rewards.append(ep_ret)
                training_log["episodes"].append({"epoch": epoch, "episode_reward": ep_ret})
                ep_ret = 0
                done = False

        # Append one bootstrap value so values has length T+1
        with torch.no_grad():
            # Use the last obs (current obs) and the last done flag in buffer
            last_done = bool(ppo.dones_buf[-1]) if len(ppo.dones_buf) > 0 else True
            if last_done:
                ppo.values_buf.append(0.0)
            else:
                v_last = agent.get_value(obs).item()
                ppo.values_buf.append(v_last)

        ppo.update(ent_coef=ent_coef)

        print(f"[epoch {epoch + 1}] mean reward: {np.mean(ep_rewards) if ep_rewards else 'n/a'}")

        with open(log_path, "w") as f:
            json.dump(training_log, f, indent=2)

    print("Training finished.")

# -- Main --

if __name__ == '__main__':
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    agent = MLPAgent(obs_dim, act_dim)
    print("Agent initialized.")

    save_agent(agent, "agent_init.pt")
    agent2 = load_agent("agent_init.pt", obs_dim, act_dim)

    ppo_train(env, agent2, epochs=200)

    save_agent(agent2, "agent_trained.pt")

    def evaluate_agent(agent, env, episodes=20, render=False):
        total_rewards = []
        for ep in range(episodes):
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            done = False
            ep_reward = 0
            while not done:
                obs_t = torch.as_tensor(obs, dtype=torch.float32)
                with torch.no_grad():
                    action, _, _ = agent.get_action(obs_t)
                step_out = env.step(action)
                if len(step_out) == 5:
                    obs, reward, terminated, truncated, *_ = step_out
                    done = terminated or truncated
                else:
                    obs, reward, done, *_ = step_out
                ep_reward += reward
                if render:
                    env.render()
            total_rewards.append(ep_reward)
        mean_rew = np.mean(total_rewards)
        print(f"Avg reward over {episodes} episodes: {mean_rew}")
        return mean_rew

    print("Evaluating trained agent:")
    evaluate_agent(agent2, env, episodes=20)
