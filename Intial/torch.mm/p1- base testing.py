import torch

# VALID PROMPTS:

# Prompt 1: Simple valid matrix multiplication
A = torch.randn(2, 3)  # 2x3 matrix
B = torch.randn(3, 4)  # 3x4 matrix
try:
    result = torch.mm(A, B)
except RuntimeError as e:
    print("Prompt 1: ", e)

# Prompt 2: Larger matrices
A = torch.randn(1000, 500)
B = torch.randn(500, 1000)
try:
    result = torch.mm(A, B)
except RuntimeError as e:
   print("Prompt 2: ", e)

# Prompt 3: Square matrices
A = torch.randn(5, 5)
B = torch.randn(5, 5)
try:
    result = torch.mm(A, B)
except RuntimeError as e:
    print("Prompt 3: ", e)

#  Incompatible Matrix Sizes:

# Prompt 4: Mismatched dimensions for matrix multiplication
A = torch.randn(2, 4)
B = torch.randn(3, 5)
try:
    result = torch.mm(A, B)
except RuntimeError as e:
    print("Prompt 4: ", e)



# Edge cases:

# Prompt 5: Empty matrices
A = torch.randn(0, 3)  # Empty matrix with 0 rows
B = torch.randn(3, 4)
try:
    result = torch.mm(A, B)
except RuntimeError as e:
    print("Prompt 5: ", e)

# Prompt 6: Degenerate matrix with a dimension of size 1
A = torch.randn(1, 3)
B = torch.randn(3, 1)
try:
    result = torch.mm(A, B)
except RuntimeError as e:
    print("Prompt 6: ", e)


# Non-tensor inputs:
# Prompt 7: Non-tensor inputs (list instead of tensor)
A = [[1, 2, 3], [4, 5, 6]]
B = [[7, 8], [9, 10], [11, 12]]
try:
    result = torch.mm(A, B)
except TypeError as e:
    print("Prompt 7: ", e)



#High-Dimensional Tensors:

# Prompt 8: 3D tensor input
A = torch.randn(2, 3, 4)
B = torch.randn(3, 4, 5)
try:
    result = torch.mm(A, B)
except RuntimeError as e:
    print("Prompt 8: ", e)


# Non-Float Data Types - Test integer or boolean matrices, or matrices with mixed types :

# Prompt 9: Integer matrices
A = torch.randint(0, 10, (2, 3))  # Integer tensor
B = torch.randint(0, 10, (3, 2))
try:
    result = torch.mm(A.float(), B.float())  # Ensure data type compatibility
except RuntimeError as e:
    print("Prompt 9: ", e)



# Prompt 10: Boolean matrices
A = torch.randint(0, 2, (2, 3)).bool()
B = torch.randint(0, 2, (3, 2)).bool()
try:
    result = torch.mm(A, B)
except RuntimeError as e:
    print("Prompt 10: ", e)

# Gradients and Autograd
# Explore how torch.mm behaves with tensors requiring gradients (common in training).
# Prompt 11: Tensors requiring gradients
A = torch.randn(3, 3, requires_grad=True)
B = torch.randn(3, 3, requires_grad=True)
try:
    result = torch.mm(A, B)
    result.backward(torch.ones_like(result))  # Backpropagate gradients
except RuntimeError as e:
    print("Prompt 11: ", e)



#Broadcasting (Should Fail)
#Matrix multiplication doesn't support broadcasting, so test for this explicitly.
# Prompt 12: Broadcasting error test
A = torch.randn(2, 3)
B = torch.randn(1, 3, 4)
try:
    result = torch.mm(A, B)
except RuntimeError as e:
    print("Prompt 12: ", e)





#Code output:
    # Prompt 4:  mat1 and mat2 shapes cannot be multiplied (2x4 and 3x5)
    # Prompt 7:  mm(): argument 'input' (position 1) must be Tensor, not list
    # Prompt 8:  self must be a matrix
    # Prompt 10:  "addmm_impl_cpu_" not implemented for 'Bool'
    # Prompt 12:  mat2 must be a matrix





# Overall analysis of code results


# Valid Matrix Multiplications (Prompts 1-3):
    # These prompts test basic matrix multiplication using compatible matrices.
    # Result: Successful, no errors were thrown for valid shapes.


# Mismatched Dimensions (Prompt 4):
    # Matrix dimensions do not align for multiplication.
    # Result: The error message mat1 and mat2 shapes cannot be multiplied (2x4 and 3x5) was correctly raised, indicating torch.mm correctly checks matrix dimension compatibility.


# Empty Matrices (Prompt 5):
    # Tests how torch.mm handles an empty matrix.
    # Result: No specific error message, which may indicate silent handling of empty matrices in PyTorch.

# Degenerate Matrices (Prompt 6):
    # This tests matrices where one dimension is 1, a potential edge case.
    # Result: No errors occurred, suggesting torch.mm handles degenerate cases smoothly.


# Non-Tensor Inputs (Prompt 7):
    # Attempts to pass lists instead of tensors.
    # Result: The error message mm(): argument 'input' (position 1) must be Tensor, not list shows that PyTorch correctly handles input type validation.


# High-Dimensional Tensors (Prompt 8):
    # torch.mm only supports 2D matrices, and this prompt uses 3D tensors.
    # Result: The error self must be a matrix indicates torch.mm checks for valid input dimensions and rejects tensors with more than 2 dimensions.


# Non-Float Data Types (Prompts 9-10):
    # Tests with integer and boolean matrices, with float casting in one case.
    # Result: addmm_impl_cpu_ not implemented for 'Bool' shows that PyTorch lacks support for certain data types (e.g., boolean) in matrix multiplication, although integer tensors were handled when cast to float.


# Gradients and Autograd (Prompt 11):
    # Tests with tensors that require gradients for backpropagation.
    # Result: This was successful, and the code handled gradient backpropagation as expected.


# Broadcasting (Prompt 12):
    # PyTorch's torch.mm doesn't support broadcasting, and this prompt intentionally violates that rule.
    # Result: The error mat2 must be a matrix correctly identifies this violation.