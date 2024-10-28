#prompt used

import torch
import torch.nn as nn
import random
import traceback

# Function to generate random input tensors with shapes that are likely to cause errors
def generate_error_prone_input(shape):
    return torch.randn(shape)

# Function to fuzz nn.Linear layer with intentional errors (mismatched dimensions)
def test_nn_linear_error():
    try:
        # Random input and output sizes, but the input tensor will have a wrong shape
        input_size = random.randint(1, 10)
        output_size = random.randint(1, 100)
        
        layer = nn.Linear(input_size, output_size)
        wrong_input_size = input_size + random.randint(1, 10)  # Deliberately wrong size
        input_tensor = generate_error_prone_input((random.randint(1, 10), wrong_input_size))  # Random batch size
        
        result = layer(input_tensor)  # Expected to fail
        print(f"nn.Linear: Unexpectedly Passed | Input Shape: {input_tensor.shape} | Output Shape: {result.shape}")
    except Exception as e:
        print(f"nn.Linear: Expected Failure | Error: {e}")
        traceback.print_exc()

# Function to fuzz nn.Conv2d layer with intentional errors (mismatched channels)
def test_nn_conv2d_error():
    try:
        # Random dimensions, but the input will have a wrong number of channels
        in_channels = random.randint(1, 10)
        out_channels = random.randint(1, 10)
        kernel_size = random.randint(1, 5)
        
        layer = nn.Conv2d(in_channels, out_channels, kernel_size)
        wrong_in_channels = in_channels + random.randint(1, 5)  # Deliberately wrong number of channels
        input_tensor = generate_error_prone_input((random.randint(1, 10), wrong_in_channels, random.randint(5, 20), random.randint(5, 20)))  # Random batch, height, width
        
        result = layer(input_tensor)  # Expected to fail
        print(f"nn.Conv2d: Unexpectedly Passed | Input Shape: {input_tensor.shape} | Output Shape: {result.shape}")
    except Exception as e:
        print(f"nn.Conv2d: Expected Failure | Error: {e}")
        traceback.print_exc()

# Function to fuzz nn.ReLU with inappropriate input
def test_nn_relu_error():
    try:
        layer = nn.ReLU()
        # Passing a tensor with fewer dimensions than expected
        input_tensor = torch.tensor(random.randint(1, 100))  # Single scalar value, not a tensor
        
        result = layer(input_tensor)  # Expected to fail
        print(f"nn.ReLU: Unexpectedly Passed | Input: {input_tensor} | Output: {result}")
    except Exception as e:
        print(f"nn.ReLU: Expected Failure | Error: {e}")
        traceback.print_exc()

# Function to fuzz nn.BatchNorm2d with mismatched feature dimensions
def test_nn_batchnorm2d_error():
    try:
        num_features = random.randint(1, 10)
        layer = nn.BatchNorm2d(num_features)
        
        # Input tensor with incorrect number of features (wrong number of channels)
        wrong_num_features = num_features + random.randint(1, 5)
        input_tensor = generate_error_prone_input((random.randint(1, 10), wrong_num_features, random.randint(5, 20), random.randint(5, 20)))  # Wrong channels
        
        result = layer(input_tensor)  # Expected to fail
        print(f"nn.BatchNorm2d: Unexpectedly Passed | Input Shape: {input_tensor.shape} | Output Shape: {result.shape}")
    except Exception as e:
        print(f"nn.BatchNorm2d: Expected Failure | Error: {e}")
        traceback.print_exc()

# Main fuzzing loop for generating errors in various nn components
def fuzz_test_torch_nn_errors():
    test_cases = [test_nn_linear_error, test_nn_conv2d_error, test_nn_relu_error, test_nn_batchnorm2d_error]
    
    for test_case in test_cases:
        print(f"Running {test_case.__name__}...")
        test_case()
        print("\n")

if __name__ == "__main__":
    fuzz_test_torch_nn_errors()
