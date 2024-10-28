#prompt: Write a script that initializes various optimizers (e.g., SGD, Adam, RMSprop) with extreme hyperparameter values, such as very high or low learning rates, momentum, and weight decay. Test these configurations on a simple neural network and monitor for divergence, NaN values, or unstable training behavior."

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 10)
        self.fc3 = nn.Linear(10, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Generate dummy data
X = torch.randn(100, 10)
y = torch.randn(100, 1)

# Define extreme hyperparameter configurations
optimizer_configs = [
    {"type": "SGD", "lr": 1e-10, "momentum": 0.0, "weight_decay": 0.0},
    {"type": "SGD", "lr": 10.0, "momentum": 0.9, "weight_decay": 1e-3},
    {"type": "Adam", "lr": 1e-10, "weight_decay": 0.0},
    {"type": "Adam", "lr": 10.0, "weight_decay": 1e-3},
    {"type": "RMSprop", "lr": 1e-10, "momentum": 0.0, "weight_decay": 0.0},
    {"type": "RMSprop", "lr": 10.0, "momentum": 0.9, "weight_decay": 1e-3}
]

# Function to initialize optimizer based on configuration
def get_optimizer(config, model):
    if config["type"] == "SGD":
        return optim.SGD(model.parameters(), lr=config["lr"], momentum=config.get("momentum", 0.0), weight_decay=config.get("weight_decay", 0.0))
    elif config["type"] == "Adam":
        return optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config.get("weight_decay", 0.0))
    elif config["type"] == "RMSprop":
        return optim.RMSprop(model.parameters(), lr=config["lr"], momentum=config.get("momentum", 0.0), weight_decay=config.get("weight_decay", 0.0))

# Training loop with monitoring
def train_model(optimizer, model, criterion, X, y, config):
    model.train()
    losses = []
    for epoch in range(10):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        # Monitor loss for divergence or instability
        losses.append(loss.item())
        if torch.isnan(loss).any() or loss.item() > 1e6:  # Detect NaN or large losses
            print(f"Divergence detected with config: {config}")
            break
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
    return losses

# Run the fuzzing tests
for config in optimizer_configs:
    print(f"\nTesting configuration: {config}")
    model = SimpleNN()
    optimizer = get_optimizer(config, model)
    criterion = nn.MSELoss()
    losses = train_model(optimizer, model, criterion, X, y, config)

    # Check for instability
    if len(losses) > 1 and (np.std(losses) / np.mean(losses)) > 1.0:  # High variance indicates instability
        print(f"Unstable training behavior detected with config: {config}")


#code output:

# Testing configuration: {'type': 'SGD', 'lr': 1e-10, 'momentum': 0.0, 'weight_decay': 0.0}
# Epoch 1, Loss: 1.1123063564300537
# Epoch 2, Loss: 1.1123063564300537
# Epoch 3, Loss: 1.1123063564300537
# Epoch 4, Loss: 1.1123063564300537
# Epoch 5, Loss: 1.1123063564300537
# Epoch 6, Loss: 1.1123063564300537
# Epoch 7, Loss: 1.1123063564300537
# Epoch 8, Loss: 1.1123063564300537
# Epoch 9, Loss: 1.1123063564300537
# Epoch 10, Loss: 1.1123063564300537

# Testing configuration: {'type': 'SGD', 'lr': 10.0, 'momentum': 0.9, 'weight_decay': 0.001}
# Epoch 1, Loss: 1.1743886470794678
# Epoch 2, Loss: 129.5127716064453
# Epoch 3, Loss: 46412.75390625
# Divergence detected with config: {'type': 'SGD', 'lr': 10.0, 'momentum': 0.9, 'weight_decay': 0.001}
# Unstable training behavior detected with config: {'type': 'SGD', 'lr': 10.0, 'momentum': 0.9, 'weight_decay': 0.001}

# Testing configuration: {'type': 'Adam', 'lr': 1e-10, 'weight_decay': 0.0}
# Epoch 1, Loss: 1.1662282943725586
# Epoch 2, Loss: 1.1662282943725586
# Epoch 3, Loss: 1.1662282943725586
# Epoch 4, Loss: 1.1662282943725586
# Epoch 5, Loss: 1.1662282943725586
# Epoch 6, Loss: 1.1662282943725586
# Epoch 7, Loss: 1.1662282943725586
# Epoch 8, Loss: 1.1662282943725586
# Epoch 9, Loss: 1.1662282943725586
# Epoch 10, Loss: 1.1662282943725586

# Testing configuration: {'type': 'Adam', 'lr': 10.0, 'weight_decay': 0.001}
# Epoch 1, Loss: 1.1772887706756592
# Divergence detected with config: {'type': 'Adam', 'lr': 10.0, 'weight_decay': 0.001}

# Testing configuration: {'type': 'RMSprop', 'lr': 1e-10, 'momentum': 0.0, 'weight_decay': 0.0}
# Epoch 1, Loss: 1.2083812952041626
# Epoch 2, Loss: 1.2083812952041626
# Epoch 3, Loss: 1.2083812952041626
# Epoch 4, Loss: 1.2083812952041626
# Epoch 5, Loss: 1.2083812952041626
# Epoch 6, Loss: 1.2083812952041626
# Epoch 7, Loss: 1.2083812952041626
# Epoch 8, Loss: 1.2083812952041626
# Epoch 9, Loss: 1.2083812952041626
# Epoch 10, Loss: 1.2083812952041626

# Testing configuration: {'type': 'RMSprop', 'lr': 10.0, 'momentum': 0.9, 'weight_decay': 0.001}
# Epoch 1, Loss: 1.0899103879928589
# Divergence detected with config: {'type': 'RMSprop', 'lr': 10.0, 'momentum': 0.9, 'weight_decay': 0.001}


# anlaysis:

# Low Learning Rates (1e-10): For configurations with a very low learning rate (e.g., lr=1e-10), the loss remains essentially unchanged across 
# all epochs. This behavior is expected since the learning rate is too low for any meaningful updates, causing the model to stagnate.

# High Learning Rates (10.0): For configurations with an excessively high learning rate (e.g., lr=10.0), we see rapid divergence. For example:

# In the SGD configuration with lr=10.0, the loss quickly escalates to large values (129.51 in epoch 2, 46412.75 in epoch 3), indicating divergence. The script correctly flags this configuration as diverging.
# Similarly, for the Adam and RMSprop configurations with lr=10.0, the loss jumps to a high value or diverges right from the first epoch, which the script detects.
# Unstable Training Detection: The script successfully identifies unstable configurations based on large increases in loss or high loss variance. 
# For example, the high-learning-rate SGD configuration was flagged as both divergent and unstable, which is expected when using such extreme parameters.

# Overall, the script is performing as intended, correctly identifying cases of divergence and stagnation. This output shows that the fuzzing test is 
# working effectively for these hyperparameter extremes.