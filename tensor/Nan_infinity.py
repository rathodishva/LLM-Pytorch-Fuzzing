#prompt : Create a test that initializes tensors with NaN and infinite values, then performs arithmetic operations to check for correct handling.


import torch

def test_tensor_nan_infinity_operations():
    # Initialize tensors with NaN and infinite values
    nan_tensor = torch.tensor([[float('nan'), 1], [2, 3]])
    inf_tensor = torch.tensor([[float('inf'), 1], [2, 3]])
    neg_inf_tensor = torch.tensor([[float('-inf'), 1], [2, 3]])

    print("Tensor with NaN values:\n", nan_tensor)
    print("Tensor with Infinite values:\n", inf_tensor)
    print("Tensor with Negative Infinite values:\n", neg_inf_tensor)

    # Define arithmetic operations
    operations = {
        'Addition': {
            'nan': nan_tensor + 5,
            'inf': inf_tensor + 5,
            'neg_inf': neg_inf_tensor + 5
        },
        'Subtraction': {
            'nan': nan_tensor - 1,
            'inf': inf_tensor - 1,
            'neg_inf': neg_inf_tensor - 1
        },
        'Multiplication': {
            'nan': nan_tensor * 2,
            'inf': inf_tensor * 2,
            'neg_inf': neg_inf_tensor * 2
        },
        'Division': {
            'nan': nan_tensor / 2,
            'inf': inf_tensor / 2,
            'neg_inf': neg_inf_tensor / 2
        }
    }

    # Perform operations and print results
    for operation, results in operations.items():
        print(f"\n{operation} Results:")
        for tensor_type, result in results.items():
            print(f"  {tensor_type.capitalize()} tensor: {result}")

# Call the test function
test_tensor_nan_infinity_operations()




# Actual output:
# Tensor with NaN values:
#  tensor([[nan, 1.],
#         [2., 3.]])
# Tensor with Infinite values:
#  tensor([[inf, 1.],
#         [2., 3.]])
# Tensor with Negative Infinite values:
#  tensor([[-inf, 1.],
#         [2., 3.]])

# Addition Results:
#   Nan tensor: tensor([[nan, 6.],
#         [7., 8.]])
#   Inf tensor: tensor([[inf, 6.],
#         [7., 8.]])
#   Neg_inf tensor: tensor([[-inf, 6.],
#         [7., 8.]])

# Subtraction Results:
#   Nan tensor: tensor([[nan, 0.],
#         [1., 2.]])
#   Inf tensor: tensor([[inf, 0.],
#         [1., 2.]])
#   Neg_inf tensor: tensor([[-inf, 0.],
#         [1., 2.]])

# Multiplication Results:
#   Nan tensor: tensor([[nan, 2.],
#         [4., 6.]])
#   Inf tensor: tensor([[inf, 2.],
#         [4., 6.]])
#   Neg_inf tensor: tensor([[-inf, 2.],
#         [4., 6.]])

# Division Results:
#   Nan tensor: tensor([[   nan, 0.5000],
#         [1.0000, 1.5000]])
#   Inf tensor: tensor([[   inf, 0.5000],
#         [1.0000, 1.5000]])
#   Neg_inf tensor: tensor([[  -inf, 0.5000],
#         [1.0000, 1.5000]])




# anlaysis:
# All Outputs Are as Expected: The script correctly handles NaN and infinite values during arithmetic operations, providing the expected results in each case.
# The behavior of tensors with NaN and infinite values is consistent with the standard mathematical treatment of these special cases.

