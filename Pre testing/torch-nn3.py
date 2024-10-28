# prompt use for this "fix the errors in this code:" and pasting previous code from torch_nn2.py

import torch
import torch.nn as nn

# 1. Unexpected Input Shapes
model = nn.Linear(10, 5)
input_tensor = torch.randn(0, 10)  # Empty tensor with zero rows
try:
    output = model(input_tensor)
    print("1. Unexpected Input Shapes Output:", output)
except Exception as e:
    print("1. Unexpected Input Shapes Error:", e)

# 2. Incompatible Input Types
model = nn.ReLU()
input_data = "not_a_tensor"  # Corrected typo
try:
    output = model(input_data)
    print("2. Incompatible Input Types Output:", output)
except Exception as e:
    print("2. Incompatible Input Types Error:", e)

# 3. Extreme Values
model = nn.Linear(10, 5)
input_tensor = torch.tensor([[float('inf'), float('-inf'), float('nan')] * 3 + [float('inf'), float('-inf')]])
try:
    output = model(input_tensor)
    print("3. Extreme Values Output:", output)
except Exception as e:
    print("3. Extreme Values Error:", e)

# 4. Invalid Layer Configurations
try:
    model = nn.Conv2d(in_channels=-3, out_channels=64, kernel_size=3)
    print("4. Layer created successfully:", model)
except Exception as e:
    print("4. Invalid Layer Configurations Error:", e)

# 5. Non-standard Initialization
model = nn.Linear(10, 5)
with torch.no_grad():
    model.weight.fill_(float('nan'))  # Fill weights with NaN
input_tensor = torch.randn(1, 10)
try:
    output = model(input_tensor)
    print("5. Non-standard Initialization Output:", output)
except Exception as e:
    print("5. Non-standard Initialization Error:", e)

# 6. Incompatible Loss Function Inputs
loss_fn = nn.CrossEntropyLoss()
input_tensor = torch.randn(3, 5)
target_tensor = torch.tensor([6, 7, 8])  # Invalid target indices for CrossEntropyLoss
try:
    loss = loss_fn(input_tensor, target_tensor)
    print("6. Incompatible Loss Function Inputs Output:", loss)
except Exception as e:
    print("6. Incompatible Loss Function Inputs Error:", e)


#output of this final code, the erors within this are all expected as a result of fuzzing:
# 1. Unexpected Input Shapes Output: tensor([], size=(0, 5), grad_fn=<AddmmBackward0>)
# 2. Incompatible Input Types Error: relu(): argument 'input' (position 1) must be Tensor, not str
# 3. Extreme Values Error: mat1 and mat2 shapes cannot be multiplied (1x11 and 10x5)
# 4. Invalid Layer Configurations Error: Trying to create tensor with negative dimension -3: [64, -3, 3, 3]
# 5. Non-standard Initialization Output: tensor([[nan, nan, nan, nan, nan]], grad_fn=<AddmmBackward0>)
# 6. Incompatible Loss Function Inputs Error: Target 6 is out of bounds.