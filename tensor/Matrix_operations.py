#prompt: Write a script that tests various matrix operations (e.g., dot product, transpose) with tensors of different shapes and logs the outcomes.

import torch

def test_matrix_operations():
    # Define a list of tensor pairs with different shapes
    tensor_pairs = [
        (torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6], [7, 8]])),  # Compatible for dot product
        (torch.tensor([[1, 2, 3], [4, 5, 6]]), torch.tensor([[7, 8], [9, 10], [11, 12]])),  # Compatible for dot product
        (torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6]])),  # Incompatible for dot product
        (torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5], [6]])),  # Compatible for dot product (result is column vector)
        (torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 0], [0, 1]])),  # Identity matrix
    ]

    # Log the outcomes
    for idx, (A, B) in enumerate(tensor_pairs):
        print(f"Tensor Pair {idx + 1}:")
        print("A:\n", A)
        print("B:\n", B)

        # Test the dot product
        try:
            dot_product = torch.matmul(A, B)
            print("Dot Product:\n", dot_product)
        except RuntimeError as e:
            print("Dot Product Error:", e)

        # Test the transpose
        try:
            A_transpose = A.T
            B_transpose = B.T
            print("A Transpose:\n", A_transpose)
            print("B Transpose:\n", B_transpose)
        except RuntimeError as e:
            print("Transpose Error:", e)

        print("\n" + "-" * 40 + "\n")

# Call the test function
test_matrix_operations()




#expected outputs:
# The script captures any runtime errors from invalid operations (e.g., incompatible shapes for dot products) and logs appropriate error messages.



#actual output:
# Tensor Pair 1:
# A:
#  tensor([[1, 2],
#         [3, 4]])
# B:
#  tensor([[5, 6],
#         [7, 8]])
# Dot Product:
#  tensor([[19, 22],
#         [43, 50]])
# A Transpose:
#  tensor([[1, 3],
#         [2, 4]])
# B Transpose:
#  tensor([[5, 7],
#         [6, 8]])

# ----------------------------------------

# Tensor Pair 2:
# A:
#  tensor([[1, 2, 3],
#         [4, 5, 6]])
# B:
#  tensor([[ 7,  8],
#         [ 9, 10],
#         [11, 12]])
# Dot Product:
#  tensor([[ 58,  64],
#         [139, 154]])
# A Transpose:
#  tensor([[1, 4],
#         [2, 5],
#         [3, 6]])
# B Transpose:
#  tensor([[ 7,  9, 11],
#         [ 8, 10, 12]])

# ----------------------------------------

# Tensor Pair 3:
# A:
#  tensor([[1, 2],
#         [3, 4]])
# B:
#  tensor([[5, 6]])
# Dot Product Error: mat1 and mat2 shapes cannot be multiplied (2x2 and 1x2)
# A Transpose:
#  tensor([[1, 3],
#         [2, 4]])
# B Transpose:
#  tensor([[5],
#         [6]])

# ----------------------------------------

# Tensor Pair 4:
# A:
#  tensor([[1, 2],
#         [3, 4]])
# B:
#  tensor([[5],
#         [6]])
# Dot Product:
#  tensor([[17],
#         [39]])
# A Transpose:
#  tensor([[1, 3],
#         [2, 4]])
# B Transpose:
#  tensor([[5, 6]])

# ----------------------------------------

# Tensor Pair 5:
# A:
#  tensor([[1, 2],
#         [3, 4]])
# B:
#  tensor([[1, 0],
#         [0, 1]])
# Dot Product:
#  tensor([[1, 2],
#         [3, 4]])
# A Transpose:
#  tensor([[1, 3],
#         [2, 4]])
# B Transpose:
#  tensor([[1, 0],
#         [0, 1]])

# ----------------------------------------





#analysis:

# All Outputs Are as Expected: The script correctly handles both valid and invalid tensor pairs,
#  providing expected results for the dot product and accurate transposes for all tensors.
# Error Handling: The script successfully captures and reports errors for incompatible 
# shapes during the dot product operation.