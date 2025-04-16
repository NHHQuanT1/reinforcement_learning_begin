import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

# Thiết lập tham số
NUM_DEVICES = 3  # Số thiết bị (K=3, scenario 1)
NUM_SUBCHANNELS = 4  # Số subchannel Sub-6GHz (N)
NUM_BEAMS = 4  # Số beam mmWave (M)
MAX_PACKETS = 6  # Số gói tin tối đa mỗi frame (L_k(t))
PLR_MAX = 0.1  # Giới hạn PLR tối đa
NUM_ACTIONS = 3  # 3 hành động: 0 (Sub-6GHz), 1 (mmWave), 2 (cả hai)
STATE_SIZE = NUM_DEVICES * 4  # State: [u_sub, u_mw, omega_sub, omega_mw] cho mỗi thiết bị
BATCH_SIZE = 16
GAMMA = 0.9  # Discount factor
EPS_START = 0.5  # Khởi đầu epsilon
EPS_END = 0.05  # Kết thúc epsilon
EPS_DECAY = 0.995  # Decay factor
TARGET_UPDATE = 10  # Cập nhật mạng target mỗi 10 bước
MEMORY_SIZE = 10000  # Kích thước bộ nhớ replay
NUM_EPISODES = 1  # Số episode huấn luyện

# Định nghĩa mạng DQN
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_size * NUM_DEVICES)  # Output cho từng thiết bị

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1, NUM_DEVICES, NUM_ACTIONS)  # Reshape thành [batch, devices, actions]

#Thiết lập môi trường
class IoTCommEnv:
    def __init__(self, K=3, N=4, M=4, MAX_PACKETS=6):
        self.K = K                      # Số thiết bị
        self.N = N                      # Số giao diện Sub-6GHz
        self.M = M                      # Số beam mmWave
        self.MAX_PACKETS = MAX_PACKETS # Gói tối đa mỗi thiết bị gửi mỗi frame

        self.P_tx_dbm = 5               # Công suất phát (dBm)
        self.N0_dbm = -169               # Nhiễu nền (dBm)

        self.reset()

    def reset(self):
        self.positions = np.random.uniform(low=2, high=20, size=(self.K, 2))
        self.ap_position = np.array([0.0, 0.0])
        self.distances = np.linalg.norm(self.positions - self.ap_position, axis=1)
        self.qos_sub = np.zeros(self.K)    # PLR trạng thái Sub-6GHz
        self.qos_mmw = np.zeros(self.K)    # PLR trạng thái mmWave
        self.state = self._observe_state()
        return self.state

    def _observe_state(self):
        # Trạng thái đơn giản: khoảng cách + QoS trạng thái
        return np.concatenate([
            self.distances.reshape(-1, 1),
            self.qos_sub.reshape(-1, 1),
            self.qos_mmw.reshape(-1, 1)
        ], axis=1)

    def _path_loss_sub6(self, d):
        return 38.5 + 30 * np.log10(d)

    def _path_loss_mmwave(self, d, is_los=True):
        if is_los:
            shadowing = np.random.normal(0, 5.8)
            return 61.4 + 20 * np.log10(d) + shadowing
        else:
            shadowing = np.random.normal(0, 8.7)
            return 72 + 29.2 * np.log10(d) + shadowing

    def _snr_to_psr(self, snr_db):
        snr_linear = 10 ** (snr_db / 10)
        return min(1.0, snr_linear / 15)

    def step(self, actions):
        assert len(actions) == self.K

        # Kiểm tra ràng buộc tài nguyên
        num_sub = sum(1 for a in actions if a in [0, 2])
        num_mmw = sum(1 for a in actions if a in [1, 2])
        if num_sub > self.N or num_mmw > self.M:
            # Vi phạm tài nguyên, trả về phạt nặng
            reward = -10.0
            return self.state, reward, True, {}

        total_psr = 0.0
        for k in range(self.K):
            d = self.distances[k]

            # Xác định trạng thái kênh mmWave (LoS hoặc NLoS)
            is_los = np.random.rand() < 0.7
            pl_sub = self._path_loss_sub6(d)
            pl_mmw = self._path_loss_mmwave(d, is_los)

            # Tính SNR
            snr_sub = self.P_tx_dbm - pl_sub - self.N0_dbm
            snr_mmw = self.P_tx_dbm - pl_mmw - self.N0_dbm

            # Tính PSR
            psr_sub = self._snr_to_psr(snr_sub)
            psr_mmw = self._snr_to_psr(snr_mmw)

            # Hành động
            a = actions[k]
            psr = 0
            if a == 0:
                psr = psr_sub
                self.qos_sub[k] = 1 - psr
            elif a == 1:
                psr = psr_mmw
                self.qos_mmw[k] = 1 - psr
            elif a == 2:
                psr = 1 - (1 - psr_sub) * (1 - psr_mmw)  # kết hợp cả hai
                self.qos_sub[k] = 1 - psr_sub
                self.qos_mmw[k] = 1 - psr_mmw

            total_psr += psr

        avg_psr = total_psr / self.K
        reward = avg_psr  # bạn có thể thêm phần phạt nếu PLR > ngưỡng

        self.state = self._observe_state()
        done = False  # bạn có thể đặt done=True nếu muốn dừng sau T bước
        return self.state, reward, done, {}


# Huấn luyện DQN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(STATE_SIZE, NUM_ACTIONS).to(device)
target_net = DQN(STATE_SIZE, NUM_ACTIONS).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
memory = deque(maxlen=MEMORY_SIZE)
eps = EPS_START

env = IoTCommEnv()
rewards = []

for episode in range(NUM_EPISODES):
    state = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    total_reward = 0

    for t in range(100):  # Giới hạn bước mỗi episode
        # Chọn hành động
        if random.random() < eps:
            actions = [random.randrange(NUM_ACTIONS) for _ in range(NUM_DEVICES)]
        else:
            with torch.no_grad():
                q_values = policy_net(state)
                actions = q_values.max(2)[1].squeeze().cpu().numpy()

        # Thực hiện hành động
        next_state, reward, done = env.step(actions)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
        total_reward += reward

        # Lưu vào bộ nhớ
        action_tensor = torch.LongTensor(actions).unsqueeze(0).to(device)
        reward_tensor = torch.FloatTensor([reward]).to(device)
        memory.append((state, action_tensor, reward_tensor, next_state, done))

        # Huấn luyện
        if len(memory) >= BATCH_SIZE:
            batch = random.sample(memory, BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            states = torch.cat(states)
            actions = torch.cat(actions)
            rewards = torch.cat(rewards)
            next_states = torch.cat(next_states)
            dones = torch.FloatTensor(dones).to(device)

            # Tính Q hiện tại và Q mục tiêu
            q_values = policy_net(states).gather(2, actions.unsqueeze(2)).squeeze(2)
            next_q_values = target_net(next_states).max(2)[0].detach()
            target_q = rewards + GAMMA * next_q_values * (1 - dones)

            # Tính mất mát
            loss = nn.MSELoss()(q_values, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        state = next_state
        if done:
            break

    rewards.append(total_reward)
    eps = max(EPS_END, EPS_DECAY * eps)

    # Cập nhật mạng target
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {episode}, Reward: {total_reward:.3f}, Epsilon: {eps:.3f}")

# Vẽ kết quả
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()