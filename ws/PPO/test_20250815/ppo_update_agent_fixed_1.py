import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import time

# -- MLPAgent --

class MLPAgent(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.policy = nn.Linear(64, act_dim)
        self.value = nn.Linear(64, 1)

    def get_action(self, obs):
        x = self.net(obs)
        logits = self.policy(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.value(x).squeeze(-1)
        return action.item(), log_prob, value

    def evaluate_actions(self, obs, actions):
        x = self.net(obs)
        logits = self.policy(x)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        value = self.value(x).squeeze(-1)
        return log_prob, entropy, value

    def get_value(self, obs):
        x = self.net(obs)
        value = self.value(x).squeeze(-1)
        return value

def save_agent(agent, filepath):
    torch.save(agent.state_dict(), filepath)

def load_agent(filepath, obs_dim, act_dim):
    agent = MLPAgent(obs_dim, act_dim)
    agent.load_state_dict(torch.load(filepath))
    return agent


# -- PPO class --

import torch.nn.functional as F

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
        self.values_buf = []
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
        current_ent_coef = ent_coef if ent_coef is not None else self.ent_coef

        rewards = np.array(self.rewards_buf, dtype=np.float32)
        dones = np.array(self.dones_buf, dtype=np.float32)
        values = np.array(self.values_buf, dtype=np.float32)
        log_probs = np.array(self.log_probs_buf, dtype=np.float32)
        obs_buf = np.array(self.obs_buf, dtype=np.float32)
        actions_buf = np.array(self.actions_buf, dtype=np.int64)

        # Compute returns and advantages with standard GAE
        returns = np.zeros(len(rewards), dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            returns[t] = gae + values[t]

        advantages = returns - values[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

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
                self.optimizer.step()



        self.clear()
        # print("PPO update completed.")


# -- Replace ppo_train function to use PPO class --

def ppo_train(env, agent, epochs=2, steps_per_epoch=2048,
              gamma=0.99, lam=0.95, clip_eps=0.2, update_iters=10,
              batch_size=64, lr=3e-4, log_path="train_log.json"):

    optimizer = optim.Adam(agent.parameters(), lr=lr)
    ppo = PPO(agent, optimizer, gamma=gamma, lam=lam, clip_eps=clip_eps,
              update_iters=update_iters, batch_size=batch_size)

    training_log = {
        "epochs": [],
        "policy_loss": [],
        "value_loss": [],
        "total_loss": [],
        "episodes": [],
        "step_sars": []
    }

    for epoch in range(epochs):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        obs = torch.as_tensor(obs, dtype=torch.float32)

        ep_ret = 0
        ep_rewards = []

        for t in range(steps_per_epoch):
            with torch.no_grad():
                action, log_prob, value = agent.get_action(obs)
            next_obs, reward, done, *_ = env.step(action)

            training_log["step_sars"].append({
                "step": epoch * steps_per_epoch + t,
                "state": obs.numpy().tolist(),
                "action": int(action),
                "reward": float(reward),
                "done": bool(done)
            })

            # Store data in PPO buffer
            ppo.add(obs.numpy(), action, reward,  int(done), value.item(), log_prob.item())

            obs = torch.as_tensor(next_obs, dtype=torch.float32) if not isinstance(next_obs, tuple) else torch.as_tensor(next_obs[0], dtype=torch.float32)
            ep_ret += reward

            if done:
                obs = env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]
                obs = torch.as_tensor(obs, dtype=torch.float32)
                ep_rewards.append(ep_ret)

                training_log["episodes"].append({
                    "epoch": epoch,
                    "episode_reward": ep_ret
                })
                ep_ret = 0

        with torch.no_grad():
            if not done:
                value = agent.get_value(obs).item()  # obs is the true next state
                ppo.values_buf.append(value)
            else:
                ppo.values_buf.append(0.0)  # Terminal, no bootstrap
        print(ppo.values_buf)
        # with torch.no_grad():
        #     value = agent.get_value(obs).item()
        # ppo.values_buf.append(value)  # Append bootstrap value for last state

        ppo.update()

        print(f"[epoch {epoch + 1}] mean reward: {np.mean(ep_rewards) if ep_rewards else 'n/a'}")

        # Optional: save logs to file
        with open(log_path, "w") as f:
            json.dump(training_log, f, indent=2)

    print("Training finished.")


# -- Main script --

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
                obs, reward, done, *_ = env.step(action)
                ep_reward += reward
                if render:
                    env.render()
            total_rewards.append(ep_reward)
        mean_rew = np.mean(total_rewards)
        print(f"Avg reward over {episodes} episodes: {mean_rew}")
        return mean_rew

    print("Evaluating trained agent:")
    evaluate_agent(agent2, env, episodes=50)

    def play_agent(agent, env):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        done = False
        total_reward = 0
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                action, _, _ = agent.get_action(obs_t)
            obs, reward, done, *_ = env.step(action)
            total_reward += reward
            env.render()
            time.sleep(0.02)
        print("Episode reward:", total_reward)
        env.close()

    # To watch trained agent:
    # play_agent(agent2, env)
