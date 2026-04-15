import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import time
import torch.nn.functional as F

# -- MLPAgent (same as Version 1) --

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

# -- Functional PPO update (Version 2) --

def ppo_update(mlp, optimizer, obs_buf, actions_buf, rewards, dones, values, v_last, old_log_probs,
               gamma=0.99, lam=0.95, clip_eps=0.2, update_iters=10, batch_size=64, ent_coef=0.01):
    """
    obs_buf: list/array [T, obs_dim]
    actions_buf: list/array [T] int indices
    rewards: [T] float
    dones: [T] {0,1}
    values: [T] value(s_t)
    v_last: scalar bootstrap value for s_{T} (0 if last done else V(s_T))
    old_log_probs: [T] scalar logp(action_t)
    """
    # numpy
    rewards = np.array(rewards, dtype=np.float32)    # [T]
    dones   = np.array(dones,   dtype=np.float32)    # [T]
    values  = np.array(values,  dtype=np.float32)    # [T]
    obs_buf = np.array(obs_buf, dtype=np.float32)    # [T, obs_dim]
    actions_buf = np.array(actions_buf, dtype=np.int64)  # [T]
    old_log_probs = np.array(old_log_probs, dtype=np.float32)  # [T]

    T = len(rewards)
    # GAE with explicit next value (values[t+1]) using v_last
    advantages = np.zeros(T, dtype=np.float32)
    returns = np.zeros(T, dtype=np.float32)
    gae = 0.0
    next_value = v_last
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_value * (1.0 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1.0 - dones[t]) * gae
        advantages[t] = gae
        returns[t] = gae + values[t]
        next_value = values[t]

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # tensors
    obs_tensor = torch.as_tensor(obs_buf, dtype=torch.float32)
    actions_tensor = torch.as_tensor(actions_buf, dtype=torch.int64)
    old_log_probs_tensor = torch.as_tensor(old_log_probs, dtype=torch.float32)
    returns_tensor = torch.as_tensor(returns, dtype=torch.float32)
    adv_tensor = torch.as_tensor(advantages, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(obs_tensor, actions_tensor,
                                             old_log_probs_tensor, returns_tensor, adv_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for _ in range(update_iters):
        for obs_b, act_b, old_logp_b, ret_b, adv_b in loader:
            logp, entropy, value = mlp.evaluate_actions(obs_b, act_b)
            ratio = torch.exp(logp - old_logp_b)
            surr1 = ratio * adv_b
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_b
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(value, ret_b)
            loss = policy_loss + 0.5 * value_loss - ent_coef * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mlp.parameters(), 0.5)
            optimizer.step()

# -- Training loop using functional ppo_update --

def ppo_train_functional(env, agent, epochs=2, steps_per_epoch=2048,
                         gamma=0.99, lam=0.95, clip_eps=0.2, update_iters=10,
                         batch_size=64, lr=3e-4, ent_coef=0.01, log_path="train_log.json"):

    optimizer = optim.Adam(agent.parameters(), lr=lr)
    training_log = {"epochs": [], "policy_loss": [], "value_loss": [], "total_loss": [], "episodes": [], "step_sars": []}

    for epoch in range(epochs):
        # rollout buffers
        obs_buf, actions_buf, rewards_buf, dones_buf, values_buf, log_probs_buf = [], [], [], [], [], []

        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        obs_t = torch.as_tensor(obs, dtype=torch.float32)

        ep_ret = 0
        ep_rewards = []
        done = False

        for t in range(steps_per_epoch):
            with torch.no_grad():
                action, log_prob, value = agent.get_action(obs_t)

            step_out = env.step(action)
            if len(step_out) == 5:
                next_obs, reward, terminated, truncated, *_ = step_out
                done = terminated or truncated
            else:
                next_obs, reward, done, *_ = step_out

            # store
            obs_buf.append(obs_t.numpy())
            actions_buf.append(int(action))
            rewards_buf.append(float(reward))
            dones_buf.append(float(done))
            values_buf.append(float(value.item()))
            log_probs_buf.append(float(log_prob.item()))

            training_log["step_sars"].append({
                "step": epoch * steps_per_epoch + t,
                "state": obs_t.numpy().tolist(),
                "action": int(action),
                "reward": float(reward),
                "done": bool(done)
            })

            obs = next_obs if not isinstance(next_obs, tuple) else next_obs[0]
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            ep_ret += reward

            if done:
                obs = env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]
                obs_t = torch.as_tensor(obs, dtype=torch.float32)
                ep_rewards.append(ep_ret)
                training_log["episodes"].append({"epoch": epoch, "episode_reward": ep_ret})
                ep_ret = 0
                done = False

        # Bootstrap v_last from final observation and last done flag
        with torch.no_grad():
            last_done = bool(dones_buf[-1]) if len(dones_buf) > 0 else True
            if last_done:
                v_last = 0.0
            else:
                v_last = float(agent.get_value(obs_t).item())

        # Update
        ppo_update(agent, optimizer,
                   obs_buf=obs_buf,
                   actions_buf=actions_buf,
                   rewards=rewards_buf,
                   dones=dones_buf,
                   values=values_buf,
                   v_last=v_last,
                   old_log_probs=log_probs_buf,
                   gamma=gamma, lam=lam, clip_eps=clip_eps,
                   update_iters=update_iters, batch_size=batch_size, ent_coef=ent_coef)

        print(f"[epoch {epoch + 1}] mean reward: {np.mean(ep_rewards) if ep_rewards else 'n/a'}")

        with open(log_path, "w") as f:
            json.dump(training_log, f, indent=2)

    print("Training finished.")

# -- Main (Version 2) --

if __name__ == '__main__':
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    agent = MLPAgent(obs_dim, act_dim)
    print("Agent initialized (functional PPO).")

    # Train with functional version
    ppo_train_functional(env, agent, epochs=200)

    # Evaluate
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

    print("Evaluating trained agent (functional PPO):")
    evaluate_agent(agent, env, episodes=20)
