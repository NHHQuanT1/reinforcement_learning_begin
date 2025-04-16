import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt

#Environment Parameters 
POSITION_MIN = -1.2
POSITION_MAX = 0.6
VELOCITY_MIN = -0.07
VELOCITY_MAX = 0.07
GOAL_POSITION = 0.5
FORCE = 0.001
GRAVITY = 0.0025

#RL Hyperparameters 
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.95
EPISODES = 3000
MAX_STEPS = 500
BATCH_SIZE = 64
MEMORY_CAPACITY = 10000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 10 #steps để update 

#Neural Network Definition 
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

#Replay Memory 
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity) #tạo bộ nhớ deque(bộ nhớ 2 đầu) khi đầy thì loại bỏ phần tử cũ
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def __len__(self):
        return len(self.buffer)

#Environment Functions 
def reset():
    position = random.uniform(-0.6, -0.4)
    velocity = 0.0
    return np.array([position, velocity])

def step(state, action):
    position, velocity = state
    
    # Update velocity based on action
    if action == 0:  # Push left
        velocity -= FORCE
    elif action == 2:  # Push right
        velocity += FORCE
    
    # Apply gravity
    velocity -= GRAVITY * np.cos(3 * position)
    velocity = np.clip(velocity, VELOCITY_MIN, VELOCITY_MAX)
    
    # Update position
    position += velocity
    position = np.clip(position, POSITION_MIN, POSITION_MAX)
    
    # Check termination
    done = position >= GOAL_POSITION
    reward = -1  # Penalty for each time step
    
    return np.array([position, velocity]), reward, done

def normalize_state(state):
    position, velocity = state
    norm_pos = (position - POSITION_MIN) / (POSITION_MAX - POSITION_MIN)
    norm_vel = (velocity - VELOCITY_MIN) / (VELOCITY_MAX - VELOCITY_MIN)
    return np.array([norm_pos, norm_vel]) #chuẩn hoá dữ liệu trong phạm vi [0,1] của vị trí và vận tốc

#Training Function 
def train_dqn():
    # Initialize networks
    policy_net = DQN(2, 3) #mạng nơ ron chính
    target_net = DQN(2, 3) #bản sao của policy_net
    target_net.load_state_dict(policy_net.state_dict()) # sao chép trọng số từ policy
    target_net.eval() #không cập nhật trọng số khi huấn luyện
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE) #hàm tối ưu việc tính gradient
    memory = ReplayBuffer(MEMORY_CAPACITY)
    
    epsilon = EPSILON_START
    episode_rewards = []
    # steps_done = 0
    losses = []
    
    for episode in range(EPISODES):
        state = reset()
        total_reward = 0
        done = False
        
        for step_count in range(MAX_STEPS):
            # Select action
            norm_state = normalize_state(state)
            state_tensor = torch.FloatTensor(norm_state).unsqueeze(0) #chuẩn hoá sang tensor và thêm 1 chiều vào
            
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randint(0, 2)
            else:
                with torch.no_grad(): #dự đoán mà k huấn luyện
                    q_values = policy_net(state_tensor)
                    action = q_values.argmax().item() #lựa chọn hành động có chỉ số cao nhất ở bên trên chưa được qua policy_net
            
            # Execute action
            next_state, reward, done = step(state, action)
            total_reward += reward
            
            # Store transition
            norm_next_state = normalize_state(next_state)
            memory.push(norm_state, action, reward, norm_next_state, done) #thêm vào replay memory
            
            state = next_state
            # steps_done += 1
            
            # Train if enough samples in memory
            if len(memory) >= BATCH_SIZE:
                # Sample batch from memory
                states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
                
                # chuyển đổi các dữ liệu lấy ra từ 1 lô thành tensor (64 kinh nghiệm được lấy ra từ replay)
                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones)
                
                # Tính giá trị tất cả Q trong 1 batch hiện tại lấy ra bằng policy_net
                current_q = policy_net(states).gather(1, actions.unsqueeze(1)) #policy_net(states) xử lý 64 trạng thái cùng lúc
                
                # Tính toán giá trị Q lấy ra từ replay (chỉ số tương ứng cho các action)
                with torch.no_grad():
                    next_q = target_net(next_states).max(1)[0]
                    target_q = rewards + (1 - dones) * DISCOUNT_FACTOR * next_q #đây là giá trị Q mục tiêu (Q kinh nghiệm) mà Q policy cần học được của 1 batch (64 mẫu)
                
                # Compute loss
                loss = nn.MSELoss()(current_q.squeeze(), target_q) #tính toán sai lệch giữa giá trị dự đoán và giá trụ mục tiêu trên toàn 1 bộ batch
                losses.append(loss.item())  # Lưu giá trị loss (chuyển từ tensor sang số)
                # Optimize the model
                optimizer.zero_grad() #đưa các trọng số về 0
                loss.backward() #tính toán lại gradient
                optimizer.step()  #cập nhật các trọng số mới
            
            if done:
                break
        
        # Sau mỗi 10 step trong 1 episode thì cập nhật lại mạng mục tiêu các trọng số
        if episode % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # Decay epsilon, giảm epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        
        episode_rewards.append(total_reward)
        
        # Print progress
        if episode % 10 == 0:
            print(f"Episode {episode:4d} | Reward: {total_reward:6.1f} | "
                  f"Epsilon: {epsilon:.3f} | Steps: {step_count+1:3d}")
    
    # #Plot training results
    plt.figure(figsize=(12, 6))

    # Subplot 1: Reward
    plt.subplot(2, 1, 1)
    plt.plot(episode_rewards, label='Total Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('DQN Training Progress on MountainCar - Reward')
    window_size = 50
    moving_avg_reward = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
    plt.plot(np.arange(window_size-1, len(episode_rewards)), moving_avg_reward, 'r-', linewidth=2, label='Moving Avg (50)')
    plt.legend()
    plt.grid()

    # Subplot 2: Loss
    plt.subplot(2, 1, 2) #biểu đô thứ 2 với 2 hàng 1 cột
    plt.plot(losses, label='Loss', color='b')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    moving_avg_loss = np.convolve(losses, np.ones(window_size)/window_size, mode='valid') # tính trung bình động của losses với cửa sổ bước 50
    plt.plot(np.arange(window_size-1, len(losses)), moving_avg_loss, 'r-', linewidth=2, label='Moving Avg (50)')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()
    return policy_net


#Main Program 
if __name__ == "__main__":
    # Train the DQN model
    print("Starting DQN training...")
    trained_model = train_dqn()
    
    # Save the trained model
    torch.save(trained_model.state_dict(), 'mountaincar_dqn.pth')
    print("Model saved to mountaincar_dqn.pth")
    
    # Evaluate the trained model
    print("\nEvaluating trained model...")
    evaluate_model(trained_model, num_episodes=20)