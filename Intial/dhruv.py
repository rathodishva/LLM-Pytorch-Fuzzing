import random
import numpy as np
import torch

random.seed(3498393318)
np.random.seed(3498393318)
torch.manual_seed(3498393318)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(3498393318)

import torch

# Create a sparse COO tensor with subtle bug
indices = torch.tensor([[0, 1], [0, 1]])
values = torch.tensor([3, 4], dtype=torch.float32)
shape = (2, 2)

# Intended behavior: Create a sparse tensor and convert it to a dense tensor
sparse_tensor = torch.sparse_coo_tensor(indices, values, shape, requires_grad=True)
dense_tensor = sparse_tensor.to_dense()

# Subtle bug: Gradients will not be computed for the dense tensor
# dense_tensor requires grad should be set to True but is missing
# Expected output: grad should be computed for dense_tensor
loss = dense_tensor.sum()
loss.backward()

print("Sparse Tensor:")
print(sparse_tensor)
print("\nDense Tensor:")
print(dense_tensor)
print("\nGradient of Sparse Tensor:")
print(sparse_tensor.grad)
print("\nGradient of Dense Tensor (should be non-None):")
print(dense_tensor.grad)  # Will print None due to the bug