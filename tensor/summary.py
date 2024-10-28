import torch
import time
import numpy as np

def log_output(message):
    """Log messages to the console."""
    print(message)

def debug_tensor_operation(operation, *args):
    """Attempts a tensor operation and captures detailed error information if it fails."""
    start_time = time.time()  # Start timing the operation
    try:
        result = operation(*args)
        elapsed_time = time.time() - start_time  # Calculate elapsed time
        log_output(f"Operation successful: {operation.__name__}, Result:\n{result}")
        log_output(f"Performance: {elapsed_time:.6f} seconds")
        return result
    except Exception as e:
        elapsed_time = time.time() - start_time  # Calculate elapsed time even if it fails
        log_output(f"Operation failed: {operation.__name__}")
        log_output(f"Arguments: {args}")
        log_output(f"Error Type: {type(e).__name__}")
        log_output(f"Error Message: {e}")
        log_output(f"Error Arguments: {e.args}")
        log_output(f"Performance: {elapsed_time:.6f} seconds")
        return None

def test_indexing_and_slicing():
    """Test various indexing and slicing techniques on tensors."""
    tensor = torch.arange(12).reshape(3, 4)
    log_output("Original Tensor:\n" + str(tensor))

    # Valid indexing
    log_output(f"Index single element: {tensor[1, 2].item()}")
    log_output(f"Slice row: {tensor[1]}")
    log_output(f"Slice column: {tensor[:, 2]}")
    log_output(f"Slice sub-tensor: {tensor[1:, 1:3]}")

    # Out of bounds access
    debug_tensor_operation(lambda: tensor[3, 0])  # Out of bounds
    debug_tensor_operation(lambda: tensor[0, 4])  # Out of bounds
    log_output(f"Negative indexing: {tensor[-1, -1].item()}")
    log_output(f"Negative slice: {tensor[-2:]}")  # Last two rows
    debug_tensor_operation(lambda: tensor[-4])  # Out of bounds with negative index

def test_nan_infinite_operations():
    """Test tensor operations with NaN and infinite values."""
    nan_tensor = torch.tensor([[float('nan'), 1.], [2., 3.]])
    inf_tensor = torch.tensor([[float('inf'), 1.], [2., 3.]])
    neg_inf_tensor = torch.tensor([[float('-inf'), 1.], [2., 3.]])

    log_output("Tensor with NaN values:\n" + str(nan_tensor))
    log_output("Tensor with Infinite values:\n" + str(inf_tensor))
    log_output("Tensor with Negative Infinite values:\n" + str(neg_inf_tensor))

    log_output("Addition Results:")
    debug_tensor_operation(torch.add, nan_tensor, 5)
    debug_tensor_operation(torch.add, inf_tensor, 5)
    debug_tensor_operation(torch.add, neg_inf_tensor, 5)

    log_output("Subtraction Results:")
    debug_tensor_operation(torch.subtract, nan_tensor, 5)
    debug_tensor_operation(torch.subtract, inf_tensor, 5)
    debug_tensor_operation(torch.subtract, neg_inf_tensor, 5)

    log_output("Multiplication Results:")
    debug_tensor_operation(torch.multiply, nan_tensor, 2)
    debug_tensor_operation(torch.multiply, inf_tensor, 2)
    debug_tensor_operation(torch.multiply, neg_inf_tensor, 2)

    log_output("Division Results:")
    debug_tensor_operation(torch.divide, nan_tensor, 2)
    debug_tensor_operation(torch.divide, inf_tensor, 2)
    debug_tensor_operation(torch.divide, neg_inf_tensor, 2)

def test_matrix_operations():
    """Test matrix operations like dot product and transpose."""
    tensor_pairs = [
        (torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6], [7, 8]])),
        (torch.tensor([[1, 2, 3], [4, 5, 6]]), torch.tensor([[7, 8], [9, 10], [11, 12]])),
        (torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6]])),
        (torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5], [6]])),
        (torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 0], [0, 1]])),
    ]

    for i, (A, B) in enumerate(tensor_pairs):
        log_output(f"Tensor Pair {i + 1}:")
        log_output("A:\n" + str(A))
        log_output("B:\n" + str(B))
        
        debug_tensor_operation(torch.matmul, A, B)  # Dot product
        debug_tensor_operation(lambda x: x.T, A)     # Transpose A
        debug_tensor_operation(lambda x: x.T, B)     # Transpose B
        log_output("----------------------------------------")

def main():
    """Run all tests."""
    log_output("Starting Tensor API Fuzz Testing...\n")
    
    log_output("=== Indexing and Slicing Tests ===")
    test_indexing_and_slicing()
    
    log_output("\n=== NaN and Infinite Values Tests ===")
    test_nan_infinite_operations()
    
    log_output("\n=== Matrix Operations Tests ===")
    test_matrix_operations()

    log_output("\nAll tests completed.")

# Run the fuzz testing framework
if __name__ == "__main__":
    main()


#output:
# Starting Tensor API Fuzz Testing...

# === Indexing and Slicing Tests ===
# Original Tensor:
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])
# Index single element: 6
# Slice row: tensor([4, 5, 6, 7])
# Slice column: tensor([ 2,  6, 10])
# Slice sub-tensor: tensor([[ 5,  6],
#         [ 9, 10]])
# Operation failed: <lambda>
# Arguments: ()
# Error Type: IndexError
# Error Message: index 3 is out of bounds for dimension 0 with size 3
# Error Arguments: ('index 3 is out of bounds for dimension 0 with size 3',)
# Performance: 0.000079 seconds
# Operation failed: <lambda>
# Arguments: ()
# Error Type: IndexError
# Error Message: index 4 is out of bounds for dimension 1 with size 4
# Error Arguments: ('index 4 is out of bounds for dimension 1 with size 4',)
# Performance: 0.000025 seconds
# Negative indexing: 11
# Negative slice: tensor([[ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])
# Operation failed: <lambda>
# Arguments: ()
# Error Type: IndexError
# Error Message: index -4 is out of bounds for dimension 0 with size 3
# Error Arguments: ('index -4 is out of bounds for dimension 0 with size 3',)
# Performance: 0.000019 seconds

# === NaN and Infinite Values Tests ===
# Tensor with NaN values:
# tensor([[nan, 1.],
#         [2., 3.]])
# Tensor with Infinite values:
# tensor([[inf, 1.],
#         [2., 3.]])
# Tensor with Negative Infinite values:
# tensor([[-inf, 1.],
#         [2., 3.]])
# Addition Results:
# Operation successful: add, Result:
# tensor([[nan, 6.],
#         [7., 8.]])
# Performance: 0.000436 seconds
# Operation successful: add, Result:
# tensor([[inf, 6.],
#         [7., 8.]])
# Performance: 0.000006 seconds
# Operation successful: add, Result:
# tensor([[-inf, 6.],
#         [7., 8.]])
# Performance: 0.000005 seconds
# Subtraction Results:
# Operation successful: subtract, Result:
# tensor([[nan, -4.],
#         [-3., -2.]])
# Performance: 0.000308 seconds
# Operation successful: subtract, Result:
# tensor([[inf, -4.],
#         [-3., -2.]])
# Performance: 0.000006 seconds
# Operation successful: subtract, Result:
# tensor([[-inf, -4.],
#         [-3., -2.]])
# Performance: 0.000006 seconds
# Multiplication Results:
# Operation successful: multiply, Result:
# tensor([[nan, 2.],
#         [4., 6.]])
# Performance: 0.000313 seconds
# Operation successful: multiply, Result:
# tensor([[inf, 2.],
#         [4., 6.]])
# Performance: 0.000005 seconds
# Operation successful: multiply, Result:
# tensor([[-inf, 2.],
#         [4., 6.]])
# Performance: 0.000005 seconds
# Division Results:
# Operation successful: divide, Result:
# tensor([[   nan, 0.5000],
#         [1.0000, 1.5000]])
# Performance: 0.000293 seconds
# Operation successful: divide, Result:
# tensor([[   inf, 0.5000],
#         [1.0000, 1.5000]])
# Performance: 0.000004 seconds
# Operation successful: divide, Result:
# tensor([[  -inf, 0.5000],
#         [1.0000, 1.5000]])
# Performance: 0.000007 seconds

# === Matrix Operations Tests ===
# Tensor Pair 1:
# A:
# tensor([[1, 2],
#         [3, 4]])
# B:
# tensor([[5, 6],
#         [7, 8]])
# Operation successful: matmul, Result:
# tensor([[19, 22],
#         [43, 50]])
# Performance: 0.000026 seconds
# Operation successful: <lambda>, Result:
# tensor([[1, 3],
#         [2, 4]])
# Performance: 0.000014 seconds
# Operation successful: <lambda>, Result:
# tensor([[5, 7],
#         [6, 8]])
# Performance: 0.000002 seconds
# ----------------------------------------
# Tensor Pair 2:
# A:
# tensor([[1, 2, 3],
#         [4, 5, 6]])
# B:
# tensor([[ 7,  8],
#         [ 9, 10],
#         [11, 12]])
# Operation successful: matmul, Result:
# tensor([[ 58,  64],
#         [139, 154]])
# Performance: 0.000002 seconds
# Operation successful: <lambda>, Result:
# tensor([[1, 4],
#         [2, 5],
#         [3, 6]])
# Performance: 0.000002 seconds
# Operation successful: <lambda>, Result:
# tensor([[ 7,  9, 11],
#         [ 8, 10, 12]])
# Performance: 0.000001 seconds
# ----------------------------------------
# Tensor Pair 3:
# A:
# tensor([[1, 2],
#         [3, 4]])
# B:
# tensor([[5, 6]])
# Operation failed: matmul
# Arguments: (tensor([[1, 2],
#         [3, 4]]), tensor([[5, 6]]))
# Error Type: RuntimeError
# Error Message: mat1 and mat2 shapes cannot be multiplied (2x2 and 1x2)
# Error Arguments: ('mat1 and mat2 shapes cannot be multiplied (2x2 and 1x2)',)
# Performance: 0.000080 seconds
# Operation successful: <lambda>, Result:
# tensor([[1, 3],
#         [2, 4]])
# Performance: 0.000002 seconds
# Operation successful: <lambda>, Result:
# tensor([[5],
#         [6]])
# Performance: 0.000001 seconds
# ----------------------------------------
# Tensor Pair 4:
# A:
# tensor([[1, 2],
#         [3, 4]])
# B:
# tensor([[5],
#         [6]])
# Operation successful: matmul, Result:
# tensor([[17],
#         [39]])
# Performance: 0.000003 seconds
# Operation successful: <lambda>, Result:
# tensor([[1, 3],
#         [2, 4]])
# Performance: 0.000002 seconds
# Operation successful: <lambda>, Result:
# tensor([[5, 6]])
# Performance: 0.000002 seconds
# ----------------------------------------
# Tensor Pair 5:
# A:
# tensor([[1, 2],
#         [3, 4]])
# B:
# tensor([[1, 0],
#         [0, 1]])
# Operation successful: matmul, Result:
# tensor([[1, 2],
#         [3, 4]])
# Performance: 0.000002 seconds
# Operation successful: <lambda>, Result:
# tensor([[1, 3],
#         [2, 4]])
# Performance: 0.000001 seconds
# Operation successful: <lambda>, Result:
# tensor([[1, 0],
#         [0, 1]])
# Performance: 0.000001 seconds
# ----------------------------------------

# All tests completed.








#output analysis:
# Indexing and Slicing Tests:

# The original tensor is printed correctly.
# Valid indexing and slicing operations are logged as expected.
# Errors due to out-of-bounds indexing are correctly caught and reported with relevant details, including the error type, message, and performance metrics.
# Negative indexing and slicing also behave as expected.


# NaN and Infinite Values Tests:

# Tensors containing NaN and infinite values are created and displayed correctly.
# Operations on these tensors log expected results and also properly handle operations that should result in NaN or infinite outputs.
# The error handling captures details on each operation, and the performance metrics indicate how long each operation took.


# Matrix Operations Tests:

# Each tensor pair is processed, and successful matrix operations (like dot products and transpositions) are logged with results and performance metrics.
# The framework properly captures and logs an error when trying to multiply tensors of incompatible shapes, showing error details as expected.
# Performance metrics for each operation are consistently displayed, indicating the efficiency of each operation.


# Overall Structure:

# The framework is well-organized, and the output is clearly structured into sections for different types of tests. This makes it easy to read and understand the results of each operation.
# The use of lambda functions to handle operations like transpositions demonstrates flexibility in the framework.
