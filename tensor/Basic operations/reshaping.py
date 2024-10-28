# prompt : Write code to create a tensor and reshape it in various ways. Include error handling for incompatible shapes.

import torch

def create_and_reshape_tensor():
    # Create a tensor with 12 elements
    original_tensor = torch.arange(12)  # Creates a 1D tensor with values from 0 to 11
    print("Original Tensor:\n", original_tensor)

    # List of new shapes to try
    shapes_to_try = [
        (3, 4),  # Compatible shape
        (4, 3),  # Compatible shape
        (6, 2),  # Compatible shape
        (2, 2, 3),  # Compatible shape
        (5, 2)   # Incompatible shape
    ]

    for shape in shapes_to_try:
        try:
            reshaped_tensor = original_tensor.reshape(shape)
            print(f"Reshaped Tensor to {shape}:\n", reshaped_tensor)
        except RuntimeError as e:
            print(f"Error reshaping to {shape}: {e}")

# Call the function
create_and_reshape_tensor()


# Output:

# Original Tensor:
#  tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
# Reshaped Tensor to (3, 4):
#  tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])
# Reshaped Tensor to (4, 3):
#  tensor([[ 0,  1,  2],
#         [ 3,  4,  5],
#         [ 6,  7,  8],
#         [ 9, 10, 11]])
# Reshaped Tensor to (6, 2):
#  tensor([[ 0,  1],
#         [ 2,  3],
#         [ 4,  5],
#         [ 6,  7],
#         [ 8,  9],
#         [10, 11]])
# Reshaped Tensor to (2, 2, 3):
#  tensor([[[ 0,  1,  2],
#          [ 3,  4,  5]],

#         [[ 6,  7,  8],
#          [ 9, 10, 11]]])
# Error reshaping to (5, 2): shape '[5, 2]' is invalid for input of size 12