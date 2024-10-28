# prompt : "Create a fuzz testing script that performs arithmetic operations (addition, subtraction, multiplication, division) on tensors of different shapes and types."
import torch
import random

def generate_random_tensor(shape):
    """Generates a random tensor with the given shape."""
    dtype = random.choice([torch.float32, torch.float64, torch.int32, torch.int64])
    return torch.rand(shape, dtype=dtype) if dtype in [torch.float32, torch.float64] else torch.randint(0, 100, shape, dtype=dtype)

def perform_operations(tensor_a, tensor_b):
    """Performs arithmetic operations on two tensors."""
    operations = {
        'Addition': (tensor_a + tensor_b),
        'Subtraction': (tensor_a - tensor_b),
        'Multiplication': (tensor_a * tensor_b),
        'Division': (tensor_a / tensor_b) if tensor_b.nonzero().numel() > 0 else None  # Avoid division by zero
    }

    for op, result in operations.items():
        if result is not None:
            print(f"{op} Result:\n{result}\n")
        else:
            print(f"{op} Result: Operation not possible due to incompatible shapes or division by zero.\n")

def fuzz_test_arithmetic_operations(num_tests=5):
    """Fuzz tests arithmetic operations on random tensor shapes and types."""
    for i in range(num_tests):
        # Generate random shapes for two tensors
        shape_a = (random.randint(1, 5), random.randint(1, 5))
        shape_b = (random.randint(1, 5), random.randint(1, 5))

        # Create random tensors
        tensor_a = generate_random_tensor(shape_a)
        tensor_b = generate_random_tensor(shape_b)

        print(f"Test {i + 1}:")
        print(f"Tensor A (shape {shape_a}, dtype {tensor_a.dtype}):\n{tensor_a}\n")
        print(f"Tensor B (shape {shape_b}, dtype {tensor_b.dtype}):\n{tensor_b}\n")

        # Perform operations
        try:
            perform_operations(tensor_a, tensor_b)
        except Exception as e:
            print(f"Error during operations: {e}")

# Run the fuzz testing script
fuzz_test_arithmetic_operations(num_tests=5)


#output:
# Test 1:
# Tensor A (shape (5, 1), dtype torch.float64):
# tensor([[0.6786],
#         [0.8443],
#         [0.0059],
#         [0.7358],
#         [0.5052]], dtype=torch.float64)

# Tensor B (shape (5, 4), dtype torch.float32):
# tensor([[0.1523, 0.9596, 0.1545, 0.4811],
#         [0.7445, 0.4452, 0.3973, 0.0647],
#         [0.0350, 0.7738, 0.0380, 0.7345],
#         [0.5042, 0.2851, 0.9940, 0.7325],
#         [0.5525, 0.4817, 0.4211, 0.0850]])

# Addition Result:
# tensor([[0.8310, 1.6383, 0.8331, 1.1598],
#         [1.5888, 1.2895, 1.2416, 0.9090],
#         [0.0409, 0.7797, 0.0439, 0.7404],
#         [1.2400, 1.0209, 1.7298, 1.4683],
#         [1.0577, 0.9869, 0.9264, 0.5902]], dtype=torch.float64)

# Subtraction Result:
# tensor([[ 0.5263, -0.2810,  0.5242,  0.1975],
#         [ 0.0998,  0.3991,  0.4470,  0.7796],
#         [-0.0290, -0.7679, -0.0320, -0.7286],
#         [ 0.2316,  0.4507, -0.2582,  0.0033],
#         [-0.0472,  0.0236,  0.0841,  0.4203]], dtype=torch.float64)

# Multiplication Result:
# tensor([[1.0339e-01, 6.5123e-01, 1.0483e-01, 3.2652e-01],
#         [6.2857e-01, 3.7592e-01, 3.3542e-01, 5.4629e-02],
#         [2.0722e-04, 4.5863e-03, 2.2496e-04, 4.3536e-03],
#         [3.7102e-01, 2.0978e-01, 7.3140e-01, 5.3898e-01],
#         [2.7913e-01, 2.4336e-01, 2.1277e-01, 4.2926e-02]], dtype=torch.float64)

# Division Result:
# tensor([[4.4547e+00, 7.0720e-01, 4.3931e+00, 1.4105e+00],
#         [1.1341e+00, 1.8963e+00, 2.1253e+00, 1.3049e+01],
#         [1.6953e-01, 7.6599e-03, 1.5616e-01, 8.0694e-03],
#         [1.4592e+00, 2.5808e+00, 7.4022e-01, 1.0045e+00],
#         [9.1449e-01, 1.0489e+00, 1.1997e+00, 5.9467e+00]], dtype=torch.float64)

# Test 2:
# Tensor A (shape (3, 1), dtype torch.float64):
# tensor([[0.2670],
#         [0.7068],
#         [0.9831]], dtype=torch.float64)

# Tensor B (shape (4, 4), dtype torch.float64):
# tensor([[0.4838, 0.1458, 0.7444, 0.5017],
#         [0.3145, 0.7763, 0.8091, 0.3539],
#         [0.1562, 0.9453, 0.7117, 0.5805],
#         [0.3828, 0.9016, 0.5880, 0.2787]], dtype=torch.float64)

# Error during operations: The size of tensor a (3) must match the size of tensor b (4) at non-singleton dimension 0
# Test 3:
# Tensor A (shape (5, 2), dtype torch.float32):
# tensor([[0.8020, 0.1874],
#         [0.3754, 0.1976],
#         [0.4862, 0.5009],
#         [0.9193, 0.1982],
#         [0.4623, 0.6536]])

# Tensor B (shape (2, 1), dtype torch.int64):
# tensor([[36],
#         [35]])

# Error during operations: The size of tensor a (5) must match the size of tensor b (2) at non-singleton dimension 0
# Test 4:
# Tensor A (shape (4, 1), dtype torch.int32):
# tensor([[54],
#         [73],
#         [ 2],
#         [72]], dtype=torch.int32)

# Tensor B (shape (5, 1), dtype torch.int64):
# tensor([[71],
#         [71],
#         [16],
#         [66],
#         [70]])

# Error during operations: The size of tensor a (4) must match the size of tensor b (5) at non-singleton dimension 0
# Test 5:
# Tensor A (shape (1, 1), dtype torch.float64):
# tensor([[0.0119]], dtype=torch.float64)

# Tensor B (shape (4, 3), dtype torch.float64):
# tensor([[0.0924, 0.6722, 0.9729],
#         [0.7771, 0.6089, 0.0553],
#         [0.7039, 0.9698, 0.8711],
#         [0.2834, 0.5360, 0.0516]], dtype=torch.float64)

# Addition Result:
# tensor([[0.1043, 0.6841, 0.9847],
#         [0.7889, 0.6208, 0.0672],
#         [0.7157, 0.9817, 0.8830],
#         [0.2953, 0.5478, 0.0635]], dtype=torch.float64)

# Subtraction Result:
# tensor([[-0.0805, -0.6604, -0.9610],
#         [-0.7652, -0.5970, -0.0434],
#         [-0.6920, -0.9579, -0.8592],
#         [-0.2715, -0.5241, -0.0398]], dtype=torch.float64)

# Multiplication Result:
# tensor([[0.0011, 0.0080, 0.0116],
#         [0.0092, 0.0072, 0.0007],
#         [0.0084, 0.0115, 0.0103],
#         [0.0034, 0.0064, 0.0006]], dtype=torch.float64)

# Division Result:
# tensor([[0.1286, 0.0177, 0.0122],
#         [0.0153, 0.0195, 0.2148],
#         [0.0169, 0.0123, 0.0136],
#         [0.0419, 0.0222, 0.2301]], dtype=torch.float64)



#expections:
# output correctly handles tensor operations according to PyTorch's 
# broadcasting rules and accurately identifies incompatible shapes, 
# resulting in appropriate error messages. Each test follows the expected behavior