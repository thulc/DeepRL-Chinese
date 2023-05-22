
import random
from collections import deque
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 定义神经网络模型
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 定义离线 DQN 算法类
class OfflineDQNAgent:
    def __init__(self, env, buffer_size, batch_size, gamma, lr, hidden_size, device):
        self.env = env
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.hidden_size = hidden_size
        self.device = device

        self.replay_buffer = deque(maxlen=buffer_size)
        self.q_network = QNetwork(env.observation_space.shape[0], env.action_space.n, hidden_size).to(device)
        self.target_network = QNetwork(env.observation_space.shape[0], env.action_space.n, hidden_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

    def act(self, state, epsilon):
        if random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32, device=self.device)
                q_values = self.q_network(state)
                action = q_values.argmax().item()
            return action

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        # print(dones)
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        # dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        target_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values[dones] = 0
        target_q_values = rewards + self.gamma * target_q_values

        loss = nn.functional.mse_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def run(self, num_episodes, epsilon_start, epsilon_end, epsilon_decay):
        epsilon = epsilon_start
        for i_episode in range(1, num_episodes + 1):
            state = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = self.act(state, epsilon)
                next_state, reward, done, _ = self.env.step(action)
                self.replay_buffer.append((state, action, reward, next_state, done))
                state = next_state
                episode_reward += reward

            self.train()
            self.update_target_network()

            print(f"Episode {i_episode}: reward = {episode_reward}")

            epsilon = max(epsilon_end, epsilon_decay * epsilon)

# 设置参数
env = gym.make("CartPole-v0")
buffer_size = 100_000
batch_size = 32
gamma = 0.99
lr = 1e-3
hidden_size = 64
device=torch.device("cpu")

# device在这个实现中，我们首先定义了一个 `QNetwork` 类来表示 DQN 的神经网络模型。
# 然后，我们定义了一个 `OfflineDQNAgent` 类来实现离线 DQN 算法。
# 在这个类中，我们使用了一个经验回放缓冲区来存储离线数据集，使用 PyTorch 实现了两个神经网络模型，
# 一个用于更新 Q 值，一个用于生成目标 Q 值。我们还实现了选择动作、训练网络和更新目标网络的方法。

# 在 `run` 方法中，我们使用 `epsilon-greedy` 策略选择动作，并在每个回合结束时更新经验回放缓冲区、
# 训练网络和更新目标网络。我们还使用了一个指数衰减的方法来逐渐降低 `epsilon` 的值，以便在训练的后期更多地依赖网络输出的 Q 值。

# 需要注意的是，在离线版本中，我们需要先收集一定量的经验数据，然后才能开始训练网络。
# 在训练网络时，我们从经验回放缓冲区中随机采样一批数据来训练网络，并使用目标网络来生成目标 Q 值。这样可以提高算法的稳定性和收敛性，并避免网络的过拟合。
env = gym.make("CartPole-v0")
agent = OfflineDQNAgent(env, buffer_size=100_000, batch_size=32, gamma=0.99, lr=1e-3, hidden_size=64, device=torch.device("cpu"))
agent.run(num_episodes=500, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.999)
