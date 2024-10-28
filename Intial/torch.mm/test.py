import torch
import random
import tracemalloc

# Initialize memory tracking
tracemalloc.start()

# Function to generate random tensors of appropriate dimensions for torch.mm
def generate_random_mm_input():
    rows_a = random.randint(1, 100)  # Random number of rows for the first matrix
    cols_a = random.randint(1, 100)  # Number of columns for the first matrix (and rows for the second matrix)
    cols_b = random.randint(1, 100)  # Random number of columns for the second matrix

    # Generate random matrices A (size: rows_a x cols_a) and B (size: cols_a x cols_b)
    A = torch.randn(rows_a, cols_a)
    B = torch.randn(cols_a, cols_b)

    return A, B

# Function to perform fuzz testing on torch.mm
def fuzz_torch_mm():
    A, B = generate_random_mm_input()

    try:
        result = torch.mm(A, B)
        # Check if the result contains NaN or Inf values
        if torch.isnan(result).any() or torch.isinf(result).any():
            print("WARNING: torch.mm produced NaN or Inf values.")
    except Exception as e:
        # Handle any unexpected errors
        print(f"ERROR in torch.mm: {str(e)}")

# Function to monitor performance and memory
def monitor_performance():
    current, peak = tracemalloc.get_traced_memory()
    return current, peak

# Main fuzz testing loop
def fuzz_test(num_tests=100):
    for _ in range(num_tests):
        fuzz_torch_mm()

        # Monitor memory usage
        current, peak = monitor_performance()
        if peak > 10**6:  # arbitrary threshold for peak memory usage in bytes
            print(f"WARNING: High memory usage - Peak: {peak / 10**6:.2f}MB")

# Run the fuzz test
if __name__ == "__main__":
    fuzz_test()
    tracemalloc.stop()
