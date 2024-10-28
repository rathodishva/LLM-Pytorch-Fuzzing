# prompt :here is the actual output, is it as expected: "index_slicing.py output"

# Key Changes:
# Lambda Functions: Each operation is wrapped in a lambda function, allowing you to call the
# operation inside the try block. This prevents the IndexError from occurring before the exception handling.

import torch

def test_tensor_indexing_and_slicing():
    # Create a sample tensor
    tensor = torch.arange(12).reshape(3, 4)  # Creates a 3x4 tensor
    print("Original Tensor:\n", tensor)

    # Define various indexing and slicing scenarios
    indexing_scenarios = [
        ("Index single element", lambda: tensor[1, 2]),             # Valid indexing
        ("Slice row", lambda: tensor[1, :]),                        # Valid slicing
        ("Slice column", lambda: tensor[:, 2]),                     # Valid slicing
        ("Slice sub-tensor", lambda: tensor[0:2, 1:3]),             # Valid slicing
        ("Out of bounds (row)", lambda: tensor[3, 0]),             # Out-of-bounds access
        ("Out of bounds (column)", lambda: tensor[0, 4]),          # Out-of-bounds access
        ("Negative indexing", lambda: tensor[-1, -2]),              # Valid negative indexing
        ("Negative slice", lambda: tensor[-2:, :]),                 # Valid negative slicing
        ("Out of bounds with negative index", lambda: tensor[-4, 0]) # Out-of-bounds access
    ]

    for description, operation in indexing_scenarios:
        try:
            result = operation()  # Call the lambda function to perform the operation
            print(f"{description}: {result}")
        except IndexError as e:
            print(f"{description}: Error - {e}")
        except Exception as e:
            print(f"{description}: Unexpected error - {e}")

# Call the test function
test_tensor_indexing_and_slicing()



# expected output:
# should handle all indexing and slicing operations properly, including those that are 
# out of bounds, without causing an unhandled exception. You should see error messages
# for out-of-bounds accesses printed as expected.




#actual output:

# Original Tensor:
#  tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])
# Index single element: 6
# Slice row: tensor([4, 5, 6, 7])
# Slice column: tensor([ 2,  6, 10])
# Slice sub-tensor: tensor([[1, 2],
#         [5, 6]])
# Out of bounds (row): Error - index 3 is out of bounds for dimension 0 with size 3
# Out of bounds (column): Error - index 4 is out of bounds for dimension 1 with size 4
# Negative indexing: 10
# Negative slice: tensor([[ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])
# Out of bounds with negative index: Error - index -4 is out of bounds for dimension 0 with size 3



#anlaysis:
# All Outputs Are as Expected: The script correctly handles both valid and invalid indexing operations,
# providing the expected results and error messages.
# The implementation properly captures exceptions for out-of-bounds accesses, which was the goal.