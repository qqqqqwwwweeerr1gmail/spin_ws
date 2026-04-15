


import gym
import torch
import torch.nn as nn
import time
import numpy as np

# Reuse your MLPAgent class (copy-paste from your training script)
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
        return action.item()

def load_agent(filepath, obs_dim, act_dim):
    agent = MLPAgent(obs_dim, act_dim)
    agent.load_state_dict(torch.load(filepath))
    agent.eval()
    return agent

def test_play(agent, env, episodes=5):
    for ep in range(episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        done = False
        total_reward = 0
        while not done:
            env.render()
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                action = agent.get_action(obs_t)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            time.sleep(0.02)  # slows down the rendering so you can watch
        print(f"Episode {ep+1} total reward: {total_reward}")
    env.close()

if __name__ == '__main__':
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # Load your trained agent
    agent = load_agent("agent_trained.pt", obs_dim, act_dim)

    # Run test episodes and render
    test_play(agent, env, episodes=5)





















