#prompt : Write a Python script that creates a tensor with random values and checks its properties like shape, dtype, and device


import torch

# Create a tensor with random values
# Specify the desired shape, e.g., (3, 4)
shape = (3, 4)
random_tensor = torch.rand(shape)

# Check tensor properties
tensor_shape = random_tensor.shape
tensor_dtype = random_tensor.dtype
tensor_device = random_tensor.device

# Print the properties
print(f"Random Tensor:\n{random_tensor}\n")
print(f"Shape: {tensor_shape}")
print(f"Dtype: {tensor_dtype}")
print(f"Device: {tensor_device}")


# Output:
# Random Tensor:
# tensor([[0.6256, 0.8337, 0.0070, 0.2175],
#         [0.8981, 0.4311, 0.7246, 0.7989],
#         [0.9298, 0.4317, 0.6306, 0.9105]])

# Shape: torch.Size([3, 4])
# Dtype: torch.float32
# Device: cpu


#