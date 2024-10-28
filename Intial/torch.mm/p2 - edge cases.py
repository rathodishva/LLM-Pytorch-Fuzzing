import torch

def fuzz_test_torch_mm():
    cases = [
        # Case 1: Mismatched dimensions
        ("Mismatched Dimensions", torch.randn(2, 3), torch.randn(2, 3)),  # Should raise a RuntimeError

        # Case 2: Empty tensor
        ("Empty Tensor", torch.empty(0, 0), torch.randn(2, 2)),  # Empty tensor

        # Case 3: Extremely large matrices
        ("Large Matrices", torch.randn(1000, 1000), torch.randn(1000, 1000)),  # Large tensors for stress testing

        # Case 4: Very small values
        ("Very Small Values", torch.full((2, 2), 1e-10), torch.full((2, 2), 1e-10)),  # Small values

        # Case 5: Very large values
        ("Very Large Values", torch.full((2, 2), 1e10), torch.full((2, 2), 1e10)),  # Large values to check overflow

        # Case 6: Non-tensor input (list)
        ("Non-Tensor Input", [[1, 2], [3, 4]], torch.randn(2, 2)),  # Should raise a TypeError

        # Case 7: Non-float tensors (integers)
        ("Integer Tensors", torch.randint(0, 10, (2, 2)), torch.randint(0, 10, (2, 2))),  # Integer tensor multiplication

        # Case 8: Different data types
        ("Mixed Data Types", torch.randn(2, 2, dtype=torch.float16), torch.randn(2, 2, dtype=torch.float64)),  # Mixed float types

        # Case 9: Invalid tensor (NaN values)
        ("NaN Values", torch.tensor([[float('nan'), 1], [2, 3]]), torch.randn(2, 2)),  # NaN values
    ]

    for name, a, b in cases:
        print(f"Test: {name}")
        try:
            result = torch.mm(a, b)
            print(f"Success: Result is \n{result}\n")
        except Exception as e:
            print(f"Failed: {e}\n")

# Run the fuzz tests
fuzz_test_torch_mm()



## output:
# Test: Mismatched Dimensions
# Failed: mat1 and mat2 shapes cannot be multiplied (2x3 and 2x3)

# Test: Empty Tensor
# Failed: mat1 and mat2 shapes cannot be multiplied (0x0 and 2x2)

# Test: Large Matrices
# Success: Result is 
# tensor([[  26.0308,   57.8144,  -14.7519,  ...,    7.7655,    8.2851,
#            54.1343],
#         [ -23.7883,  -39.3154,   14.7971,  ...,   38.6902,   -2.7644,
#            34.6556],
#         [  34.3498,   51.0889, -102.9428,  ...,   24.1490,    0.6695,
#           -20.1557],
#         ...,
#         [  14.5352,   19.8186,  -18.6186,  ...,  -31.3839,   17.6577,
#           -14.6270],
#         [ -76.9051,  -14.2718,   11.9350,  ...,   16.9280,   11.8137,
#            -1.3305],
#         [ -18.2828,   11.6074,   -0.8752,  ...,   -4.4534,    4.7975,
#           -30.8917]])

# Test: Very Small Values
# Success: Result is 
# tensor([[2.0000e-20, 2.0000e-20],
#         [2.0000e-20, 2.0000e-20]])

# Test: Very Large Values
# Success: Result is 
# tensor([[2.0000e+20, 2.0000e+20],
#         [2.0000e+20, 2.0000e+20]])

# Test: Non-Tensor Input
# Failed: mm(): argument 'input' (position 1) must be Tensor, not list

# Test: Integer Tensors
# Success: Result is 
# tensor([[36, 58],
#         [38, 54]])

# Test: Mixed Data Types
# Failed: expected m1 and m2 to have the same dtype, but got: c10::Half != double

# Test: NaN Values
# Success: Result is 
# tensor([[    nan,     nan],
#         [-2.0635, -2.5542]])










# Analysis of resuts-

    # Detecting Edge Cases and Exceptions:

    # Cases like "Mismatched Dimensions," "Empty Tensor," and "Non-Tensor Input" successfully trigger exceptions, showing that GPT can generate meaningful
    # fuzzing prompts that reveal issues related to invalid input types and sizes. This aligns with fuzzing's goal to stress the system and uncover 
    # robustness or error handling issues.


    # Handling Extreme Values:

    # For "Very Small Values" and "Very Large Values," the multiplication doesn't trigger exceptions but outputs reasonable results (tiny or huge numbers).
    # GPT effectively generated scenarios to check for potential underflow or overflow issues. While no overflow occurred here, such cases are essential 
    # for testing numeric stability in deep learning libraries.


    # Mixed Data Types:

    # The "Mixed Data Types" test case failed due to incompatible data types (float16 vs. float64), which is a useful finding for your fuzzer. This 
    # demonstrates how GPT can help identify type mismatches that could lead to runtime issues.


    # Stress Testing with Large Matrices:

    # The successful handling of "Large Matrices" shows that GPT-generated fuzzing cases can be used for performance or stress tests without
    # crashing the system, providing insights into how PyTorch handles large tensor operations.


    # Non-numeric Behavior:

    # In the "NaN Values" case, the result propagates NaN values, indicating correct handling of invalid numbers in tensor computations,
    # which is valuable for testing robustness in numerical computations.