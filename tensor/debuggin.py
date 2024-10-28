# prompt: Write a function that captures and prints detailed debugging information when operations on tensors fail

import torch

def debug_tensor_operation(operation, *args):
    """Attempts a tensor operation and captures detailed error information if it fails."""
    try:
        result = operation(*args)
        print(f"Operation successful: {operation.__name__}, Result:\n{result}")
        return result
    except Exception as e:
        # Capture detailed debug information
        print(f"Operation failed: {operation.__name__}")
        print(f"Arguments: {args}")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {e}")
        if hasattr(e, 'args'):
            print("Error Arguments:", e.args)
        # Optionally, print stack trace if needed (uncomment below line)
        # import traceback; traceback.print_exc()
        return None

# Example usage of the debug_tensor_operation function
def test_debug_tensor_operations():
    # Create example tensors
    A = torch.tensor([[1, 2], [3, 4]])
    B = torch.tensor([[5, 6], [7, 8]])
    C = torch.tensor([[1], [2], [3]])  # Incompatible for dot product with A

    # Test valid operation
    debug_tensor_operation(torch.matmul, A, B)

    # Test invalid operation (dot product with incompatible shapes)
    debug_tensor_operation(torch.matmul, A, C)

    # Test valid operation (transpose)
    debug_tensor_operation(lambda x: x.T, A)

# Call the test function
test_debug_tensor_operations()





# acutal output:
# Operation successful: matmul, Result:
# tensor([[19, 22],
#         [43, 50]])
# Operation failed: matmul
# Arguments: (tensor([[1, 2],
#         [3, 4]]), tensor([[1],
#         [2],
#         [3]]))
# Error Type: RuntimeError
# Error Message: mat1 and mat2 shapes cannot be multiplied (2x2 and 3x1)
# Error Arguments: ('mat1 and mat2 shapes cannot be multiplied (2x2 and 3x1)',)
# Operation successful: <lambda>, Result:
# tensor([[1, 3],
#         [2, 4]])




#analysis:
# The output captures the results of both successful and failed tensor operations, providing detailed information about the error when applicable.