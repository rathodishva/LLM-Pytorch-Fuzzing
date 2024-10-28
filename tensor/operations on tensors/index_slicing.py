#prompt: Write a code snippet that tests various indexing and slicing techniques on tensors, including out-of-bounds accesses

import torch

def test_tensor_indexing_and_slicing():
    # Create a sample tensor
    tensor = torch.arange(12).reshape(3, 4)  # Creates a 3x4 tensor
    print("Original Tensor:\n", tensor)

    # Define various indexing and slicing scenarios
    indexing_scenarios = [
        ("Index single element", tensor[1, 2]),             # Valid indexing
        ("Slice row", tensor[1, :]),                        # Valid slicing
        ("Slice column", tensor[:, 2]),                     # Valid slicing
        ("Slice sub-tensor", tensor[0:2, 1:3]),             # Valid slicing
        ("Out of bounds (row)", tensor[3, 0]),             # Out-of-bounds access
        ("Out of bounds (column)", tensor[0, 4]),          # Out-of-bounds access
        ("Negative indexing", tensor[-1, -2]),              # Valid negative indexing
        ("Negative slice", tensor[-2:, :]),                 # Valid negative slicing
        ("Out of bounds with negative index", tensor[-4, 0]) # Out-of-bounds access
    ]

    for description, operation in indexing_scenarios:
        try:
            result = operation
            print(f"{description}: {result}")
        except IndexError as e:
            print(f"{description}: Error - {e}")
        except Exception as e:
            print(f"{description}: Unexpected error - {e}")

# Call the test function
test_tensor_indexing_and_slicing()


# expected output:
# The script prints the results of valid operations or error messages for any out-of-bounds accesses or unexpected errors.


# Actual output:
# Original Tensor:
#  tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])
# Traceback (most recent call last):
#   File "/Users/ishvarathod/Documents/Uni/Year 2/Sem 2/Topics/LLM-Pytorch-Fuzzing/tensor/operations on tensors/index_slicing.py", line 33, in <module>
#     test_tensor_indexing_and_slicing()
#   File "/Users/ishvarathod/Documents/Uni/Year 2/Sem 2/Topics/LLM-Pytorch-Fuzzing/tensor/operations on tensors/index_slicing.py", line 16, in test_tensor_indexing_and_slicing
#     ("Out of bounds (row)", tensor[3, 0]),             # Out-of-bounds access
#                             ~~~~~~^^^^^^
# IndexError: index 3 is out of bounds for dimension 0 with size 3



#analysis:
# 
#  traceback issues found

# The expectation was that your code would handle this IndexError using the try and except blocks. However, it seems 
# that the error is occurring before the try block can catch it, which is why you see a traceback instead of a handled error message.