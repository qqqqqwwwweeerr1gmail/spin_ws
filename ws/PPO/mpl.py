import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class MLPAgent(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU()
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

    def get_action_batch(self, obs):
        x = self.net(obs)
        logits = self.policy(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.value(x).squeeze(-1)
        return action, log_prob, value

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


def compute_returns(rewards, dones, values, gamma, lam):
    returns = []
    gae = 0
    next_value = values[-1]
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * next_value * dones[step] - values[step]
        gae = delta + gamma * lam * dones[step] * gae
        returns.insert(0, gae + values[step])
        next_value = values[step]
    return returns

def ppo_train(env, agent, epochs=2, steps_per_epoch=2048, gamma=0.99, lam=0.95,
              clip_eps=0.2, update_iters=10, batch_size=64, lr=3e-4):
    optimizer = optim.Adam(agent.parameters(), lr=lr)
    obs_dim = env.observation_space.shape[0]
    for epoch in range(epochs):
        obs = env.reset()
        if isinstance(obs, tuple): obs = obs[0]
        obs = torch.as_tensor(obs, dtype=torch.float32)
        buffer = {'obs': [], 'actions': [], 'rewards': [], 'values': [], 'log_probs': [], 'dones': []}
        ep_rewards = []
        ep_ret = 0

        for t in range(steps_per_epoch):
            with torch.no_grad():
                action, log_prob, value = agent.get_action(obs)
            next_obs, reward, done, *_ = env.step(action)
            buffer['obs'].append(obs.numpy())
            buffer['actions'].append(action)
            buffer['rewards'].append(reward)
            buffer['values'].append(value.item())
            buffer['log_probs'].append(log_prob.item())
            buffer['dones'].append(1 - int(done))
            obs = torch.as_tensor(next_obs, dtype=torch.float32)
            ep_ret += reward
            if done:
                obs = env.reset()
                if isinstance(obs, tuple): obs = obs[0]
                obs = torch.as_tensor(obs, dtype=torch.float32)
                ep_rewards.append(ep_ret)
                ep_ret = 0

        with torch.no_grad():
            value = agent.get_value(obs).item()
        buffer['values'].append(value)

        rewards = np.array(buffer['rewards'], dtype=np.float32)
        dones = np.array(buffer['dones'], dtype=np.float32)
        values = np.array(buffer['values'], dtype=np.float32)
        log_probs = np.array(buffer['log_probs'], dtype=np.float32)
        obs_buf = np.array(buffer['obs'], dtype=np.float32)
        actions = np.array(buffer['actions'], dtype=np.int64)

        returns = compute_returns(rewards, dones, values, gamma, lam)
        advantages = np.array(returns) - values[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_tensor = torch.as_tensor(obs_buf, dtype=torch.float32)
        actions_tensor = torch.as_tensor(actions, dtype=torch.int64)
        log_probs_tensor = torch.as_tensor(log_probs, dtype=torch.float32)
        returns_tensor = torch.as_tensor(returns, dtype=torch.float32)
        adv_tensor = torch.as_tensor(advantages, dtype=torch.float32)

        dataset = torch.utils.data.TensorDataset(obs_tensor, actions_tensor, log_probs_tensor, returns_tensor, adv_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for _ in range(update_iters):
            for obs_b, act_b, old_logp_b, ret_b, adv_b in loader:
                logp, entropy, value = agent.evaluate_actions(obs_b, act_b)
                ratio = (logp - old_logp_b).exp()
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (ret_b - value).pow(2).mean()
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print(f"[epoch {epoch+1}] mean reward: {np.mean(ep_rewards) if ep_rewards else 'n/a'}")

    print("Training finished.")

if __name__ == '__main__':

    # For "CartPole-v1", obs_dim = 4, act_dim = 2
    env_name = "CartPole-v1"


    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # -- Step 1: init agent --
    agent = MLPAgent(obs_dim, act_dim)
    print("Agent initialized.")

    # -- Step 2: Save agent untrained --
    save_agent(agent, "agent_init.pt")

    # -- Step 3: Load agent (verify loading) --
    agent2 = load_agent("agent_init.pt", obs_dim, act_dim)

    # -- Step 4: Train agent --
    ppo_train(env, agent2, epochs=200)  # increase epochs for better performance!

    # -- Step 5: Save after training --
    save_agent(agent2, "agent_trained.pt")

    # -- Step 6: Evaluate performance --
    def evaluate_agent(agent, env, episodes=20, render=False):
        total_rewards = []
        for ep in range(episodes):
            obs = env.reset()
            if isinstance(obs, tuple): obs = obs[0]
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
    evaluate_agent(agent2, env, episodes=20)




    import time

    def play_agent(agent, env):
        obs = env.reset()
        if isinstance(obs, tuple): obs = obs[0]
        done = False
        total_reward = 0
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                action, _, _ = agent.get_action(obs_t)
            obs, reward, done, *_ = env.step(action)
            total_reward += reward
            env.render()
            time.sleep(0.02)  # so you can see animation
        print("Episode reward:", total_reward)
        env.close()

    # Usage:
    # play_agent(agent2, env)













