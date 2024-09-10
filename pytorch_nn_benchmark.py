import torch

# Enable benchmarking
torch.backends.cudnn.benchmark = True

import torch
import torch.nn as nn
import torch.optim as optim
import time

# Enable benchmark mode
torch.backends.cudnn.benchmark = True

# Check if CUDA is available and set device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(1000, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Generate random data
input_data = torch.randn(10000, 1000).to(device)  # Batch size of 10k, 1000 features
target_data = torch.randn(10000, 10).to(device)   # 10 output classes

# Initialize the model, loss function, and optimizer
model = SimpleNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Train for a few iterations and measure time
start_time = time.time()

for epoch in range(10):  # Run for 10 epochs
    optimizer.zero_grad()
    output = model(input_data)
    loss = criterion(output, target_data)
    loss.backward()
    optimizer.step()

end_time = time.time()

print(f"Training completed in {end_time - start_time:.3f} seconds.")
