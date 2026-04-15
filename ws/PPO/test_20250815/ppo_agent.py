import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

class PPO:
    def __init__(self, mlp, optimizer, gamma=0.99, lam=0.95, clip_eps=0.2, update_iters=10, batch_size=64, ent_coef=0.01):
        """
        Initialize the PPO trainer with the agent, optimizer, and hyperparameters.

        Args:
            mlp: The MLPAgent instance (your policy and value network).
            optimizer: The optimizer (e.g., torch.optim.Adam).
            gamma: Discount factor for rewards (default: 0.99).
            lam: Lambda for GAE (Generalized Advantage Estimation, default: 0.95).
            clip_eps: Clipping parameter for PPO ratio (default: 0.2).
            update_iters: Number of update epochs per PPO update (default: 10).
            batch_size: Batch size for mini-batch updates (default: 64).
            ent_coef: Coefficient for entropy regularization (default: 0.01).
        """
        self.mlp = mlp  # The MLPAgent (policy and value network)
        self.optimizer = optimizer  # Optimizer for training
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.update_iters = update_iters
        self.batch_size = batch_size
        self.ent_coef = ent_coef

        # Initialize buffers as empty lists
        self.obs_buf = []       # List to store observations (numpy arrays or lists)
        self.actions_buf = []   # List to store action indices (e.g., [WS_idx, AD_idx] as lists or tuples)
        self.rewards_buf = []   # List to store rewards (scalars)
        self.dones_buf = []     # List to store done flags (booleans or integers)
        self.values_buf = []    # List to store value estimates (scalars)
        self.log_probs_buf = [] # List to store log probabilities (scalars)

    def add(self, obs, action, reward, done, value, log_prob):
        """
        Add a single step of data to the buffers.

        Args:
            obs: Observation (numpy array or list, shape depends on input_dim).
            action: Action as indices (list or numpy array of shape [2], e.g., [ws_idx, ad_idx]).
            reward: Reward scalar (float).
            done: Done flag (boolean or int, indicating episode end).
            value: Value estimate from the agent (scalar float).
            log_prob: Log probability of the action (scalar float).
        """
        self.obs_buf.append(obs)
        self.actions_buf.append(action)  # Store as list or array; ensure it's [2] indices
        self.rewards_buf.append(reward)
        self.dones_buf.append(done)
        self.values_buf.append(value)
        self.log_probs_buf.append(log_prob)

    def clear(self):
        """
        Clear all buffers to prepare for a new rollout.
        """
        self.obs_buf = []
        self.actions_buf = []
        self.rewards_buf = []
        self.dones_buf = []
        self.values_buf = []
        self.log_probs_buf = []

    def update(self, ent_coef=None):
        """
        Perform the PPO update using the stored buffers.
        This method implements the GAE computation, advantage normalization, and PPO loss.

        Args:
            ent_coef: Optional entropy coefficient to override the default ent_coef.

        Returns:
            None, but performs the optimization step.
        """
        if len(self.rewards_buf) == 0:
            print("No data in buffers, skipping PPO update.")
            return  # Skip if no data

        # Use provided ent_coef if given, else use self.ent_coef
        current_ent_coef = ent_coef if ent_coef is not None else self.ent_coef

        # Convert buffers to numpy arrays for processing
        rewards = np.array(self.rewards_buf, dtype=np.float32)
        dones = np.array(self.dones_buf, dtype=np.float32)
        values = np.array(self.values_buf, dtype=np.float32)
        log_probs = np.array(self.log_probs_buf, dtype=np.float32)  # Old log probs from rollout
        obs_buf = np.array(self.obs_buf, dtype=np.float32)  # Shape: (num_steps, input_dim)
        actions_buf = np.array(self.actions_buf, dtype=np.long)  # Shape: (num_steps, 2), action indices

        # Compute returns and advantages using GAE
        returns = []
        gae = 0
        next_value = 0.0
        for step in reversed(range(len(rewards))):
            if dones[step]:
                delta = rewards[step] - values[step]
                next_value = 0.0
            else:
                delta = rewards[step] + self.gamma * next_value - values[step]
            gae = delta + self.gamma * self.lam * (1 - dones[step]) * gae
            returns.insert(0, gae + values[step])
            next_value = values[step]
        returns = np.array(returns, dtype=np.float32)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        obs_tensor = torch.as_tensor(obs_buf, dtype=torch.float32)
        actions_tensor = torch.as_tensor(actions_buf, dtype=torch.long)  # Long for indices
        old_log_probs_tensor = torch.as_tensor(log_probs, dtype=torch.float32)
        returns_tensor = torch.as_tensor(returns, dtype=torch.float32)
        adv_tensor = torch.as_tensor(advantages, dtype=torch.float32)

        # Create DataLoader for mini-batches
        dataset = torch.utils.data.TensorDataset(obs_tensor, actions_tensor, old_log_probs_tensor, returns_tensor, adv_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # PPO update loop
        for _ in range(self.update_iters):
            for obs_b, act_b, old_logp_b, ret_b, adv_b in loader:
                # Evaluate actions with current policy
                logp, entropy, value, _ = self.mlp.evaluate_actions(obs_b, act_b)  # act_b is [batch, 2] indices

                # Compute PPO ratio and losses
                ratio = torch.exp(logp - old_logp_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(ret_b, value)
                loss = policy_loss + 0.5 * value_loss - current_ent_coef * entropy.mean()

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # Clear buffers after update
        self.clear()
        print("PPO update completed.")





















