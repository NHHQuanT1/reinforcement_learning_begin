import torch
import torch.nn as nn
import torch.optim as optim

# Định nghĩa mô hình
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 16)  # Hidden layer 1
        self.fc2 = nn.Linear(16, 8)  # Hidden layer 2
        self.fc3 = nn.Linear(8, 1)   # Output layer
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation cho đầu ra

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Khởi tạo model
model = SimpleNN()
print(model)

# Định nghĩa loss function và optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy Loss (dùng cho phân loại nhị phân)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Tạo dữ liệu giả để kiểm tra mô hình
sample_input = torch.tensor([[0.5, 0.8]], dtype=torch.float32)
output = model(sample_input)
print("Output:", output.item())
