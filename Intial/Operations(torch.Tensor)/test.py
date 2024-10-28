import torch
import random
import logging
import traceback

# Set up logging to capture errors and crashes
logging.basicConfig(filename='torchnn_fuzzing_logs.txt', level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

def log_exception(e):
    logging.error(f"Exception: {e}")
    logging.error(traceback.format_exc())

# Random tensor generator
def generate_random_tensor():
    dim = random.randint(1, 4)  # Randomly choose dimensions between 1 and 4
    shape = tuple(random.randint(1, 10) for _ in range(dim))  # Random shape (1-10 elements per dimension)
    dtype = random.choice([torch.float32, torch.int32, torch.bool])  # Random data type
    tensor = torch.rand(shape).type(dtype)  # Create tensor with random shape and data type
    if dtype == torch.int32:
        tensor = tensor * 100  # Scale int tensors to reasonable range
    return tensor

# Fuzzing function to test PyTorch operations
def fuzz_tensor_operations():
    operations = [
        'add',          # Element-wise addition
        'sub',          # Element-wise subtraction
        'mul',          # Element-wise multiplication
        'div',          # Element-wise division
        'mm',           # Matrix multiplication
        'matmul',       # Matrix multiplication (generalized)
        'sum',          # Summing elements in the tensor
        'mean',         # Calculating mean of tensor elements
        'exp',          # Exponentiation
        'log',          # Logarithm
    ]

    for i in range(1000):  # Fuzzing with 1000 random inputs
        try:
            # Generate two random tensors for binary operations
            tensor1 = generate_random_tensor()
            tensor2 = generate_random_tensor()

            # Randomly choose an operation
            operation = random.choice(operations)
            print(f"Test {i+1}: Operation {operation} on tensors of shape {tensor1.shape} and {tensor2.shape}")

            # Perform the operation
            if operation in ['add', 'sub', 'mul', 'div']:  # Binary element-wise operations
                if tensor1.shape == tensor2.shape:  # Ensure shapes match for binary ops
                    result = getattr(torch, operation)(tensor1, tensor2)
                else:
                    raise ValueError("Shape mismatch for element-wise operations for", operation, "operation")
            elif operation in ['mm', 'matmul']:  # Matrix multiplication
                if len(tensor1.shape) == 2 and len(tensor2.shape) == 2:  # Only do matrix multiplication on 2D tensors
                    result = getattr(torch, operation)(tensor1, tensor2)
                else:
                    raise ValueError("Matrix multiplication requires 2D tensors")
            else:  # Unary operations (e.g., sum, mean, exp, log)
                result = getattr(torch, operation)(tensor1)

            print(f"Result: {result}")
        
        except Exception as e:
            # Log and print exceptions (bugs or errors)
            print(f"Exception encountered during operation {operation}: {e}")
            log_exception(e)

if __name__ == "__main__":
    print("Starting fuzz testing for PyTorch tensor operations...")
    fuzz_tensor_operations()
    print("Fuzz testing complete. Check fuzzing_logs.txt for any errors.")
