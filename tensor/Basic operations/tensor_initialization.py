#prompt: Generate a Python function that initializes a tensor using different methods (e.g., zeros, ones, random, arange) and prints the resulting tensors.


import torch

def initialize_tensors():
    # Initialize tensors using different methods
    tensor_zeros = torch.zeros((2, 3))        # 2x3 tensor of zeros
    tensor_ones = torch.ones((2, 3))          # 2x3 tensor of ones
    tensor_random = torch.rand((2, 3))        # 2x3 tensor of random values
    tensor_arange = torch.arange(0, 6).reshape((2, 3))  # 2x3 tensor with values from 0 to 5

    # Print the resulting tensors
    print("Tensor of Zeros:\n", tensor_zeros)
    print("Tensor of Ones:\n", tensor_ones)
    print("Tensor of Random Values:\n", tensor_random)
    print("Tensor of Arange:\n", tensor_arange)

# Call the function
initialize_tensors()


# Out put:
# Tensor of Zeros:
#  tensor([[0., 0., 0.],
#         [0., 0., 0.]])
# Tensor of Ones:
#  tensor([[1., 1., 1.],
#         [1., 1., 1.]])
# Tensor of Random Values:
#  tensor([[0.1035, 0.9327, 0.6817],
#         [0.1815, 0.7810, 0.1096]])
# Tensor of Arange:
#  tensor([[0, 1, 2],
#         [3, 4, 5]])