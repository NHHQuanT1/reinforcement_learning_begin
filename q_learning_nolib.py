import random
import numpy as np

# Tham số môi trường
POSITION_MIN = -1.2
POSITION_MAX = 0.6
VELOCITY_MIN = -0.07
VELOCITY_MAX = 0.07
GOAL_POSITION = 0.5
FORCE = 0.001  # Lực từ hành động đẩy
GRAVITY = 0.0025  # Lực hấp dẫn từ độ dốc

# Hàm khởi tạo trạng thái ban đầu
def reset():
    position = random.uniform(-0.6, -0.4)  # Vị trí ban đầu ngẫu nhiên
    velocity = 0.0
    return np.array([position, velocity])

# Hàm step: Hành động trả về, trạng thái mới, phần thưởng
def step(state, action):
    position, velocity = state
    
    # Cập nhật vận tốc dựa trên hành động
    if action == 0:  # Đẩy trái
        velocity -= FORCE
    elif action == 2:  # Đẩy phải
        velocity += FORCE
    # Hành động 1 (không làm gì) không thay đổi vận tốc từ lực đẩy
    
    # Cập nhật vận tốc dựa trên lực hấp dẫn
    velocity -= GRAVITY * np.cos(3 * position)
    velocity = max(VELOCITY_MIN, min(VELOCITY_MAX, velocity))
    
    # Cập nhật vị trí
    position += velocity
    position = max(POSITION_MIN, min(POSITION_MAX, position))
    
    # Kiểm tra trạng thái kết thúc
    done = position >= GOAL_POSITION
    
    reward = -1
    
    # Trạng thái mới
    new_state = np.array([position, velocity])
    
    return new_state, reward, done

# Tham số Q-learning
c_learning_rate = 0.1
c_discount_value = 0.9
c_no_of_eps = 5000

v_epsilon = 0.9
c_start_ep_epsilon_decay = 1
c_end_ep_epsilon_decay = c_no_of_eps // 2 # khoảng mà tập epsilon giảm dần
v_epsilon_decay = v_epsilon / (c_end_ep_epsilon_decay - c_start_ep_epsilon_decay) #giá trị giảm dần của mỗi tập

# Kích thước Q-table
q_table_size = [20, 20]
q_table_segment_size = np.array([POSITION_MAX - POSITION_MIN, VELOCITY_MAX - VELOCITY_MIN]) / q_table_size # kích thước mỗi ô khi chia 20x20

# Hàm chuyển đổi trạng thái thực sang trạng thái rời rạc
def convert_state(real_state):
    q_state = (real_state - np.array([POSITION_MIN, VELOCITY_MIN])) // q_table_segment_size 
    q_state = np.clip(q_state, [0, 0], np.array(q_table_size) - 1)
    return tuple(q_state.astype(int))
q_table = np.random.uniform(low=-2, high=0, size=(q_table_size + [3]))

max_ep_reward = -999
max_ep_action_list = []
max_start_state = None
max_steps = 200

# Quá trình traning
for ep in range(c_no_of_eps):
    print("Eps = ", ep)
    done = False
    ep_reward = 0
    action_list = []
    real_state = reset()
    current_state = convert_state(real_state)
    step_count = 0

    while not done and step_count < max_steps:
        # Chọn hành động
        if np.random.random() > v_epsilon:
            action = np.argmax(q_table[current_state])
        else:
            action = np.random.randint(0, 3)

        action_list.append(action)

        next_real_state, reward, done = step(real_state, action)
        ep_reward += reward
        step_count += 1

        # if show_now:
        #     print(f"Ep: {ep}, Action: {action}, Position: {next_real_state[0]:.3f}, Velocity: {next_real_state[1]:.3f}, Reward: {reward}")

        if done:
            if next_real_state[0] >= GOAL_POSITION:
                print(f"Đã đến cờ tại ep = {ep}, reward = {ep_reward}")
                if ep_reward > max_ep_reward:
                    max_ep_reward = ep_reward
                    max_ep_action_list = action_list
                    max_start_state = current_state
        else:
            next_state = convert_state(next_real_state)
            current_q_value = q_table[current_state + (action,)]
            new_q_value = (1 - c_learning_rate) * current_q_value + c_learning_rate * (reward + c_discount_value * np.max(q_table[next_state]))
            q_table[current_state + (action,)] = new_q_value

            real_state = next_real_state
            current_state = next_state

        # Giảm epsilon
        if c_end_ep_epsilon_decay >= ep > c_start_ep_epsilon_decay:
            v_epsilon = max(0, v_epsilon - v_epsilon_decay)


# In kết quả
print("Max reward = ", max_ep_reward)
print("Max action list = ", max_ep_action_list)