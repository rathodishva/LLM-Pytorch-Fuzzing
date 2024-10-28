# prompt:  "Write a Python script that creates a tensor with requires_grad set to True, performs a simple operation, and verifies that the gradient is correctly calculated. Include edge cases such as operations that could lead to NaN gradients."


import torch

# Function to test tensor operations and gradient calculations
def test_tensor_operations():
    # Creating a tensor with requires_grad set to True
    x = torch.tensor([2.0, 3.0], requires_grad=True)

    # Perform a simple operation: y = x^2
    y = x ** 2

    # Perform a backward pass to calculate gradients
    y.backward(torch.tensor([1.0, 1.0]))  # Gradient of y with respect to x

    # Print the gradients
    print("Gradient of y with respect to x:", x.grad)

    # Reset gradients for the next test
    x.grad.zero_()

    # Edge case 1: Division by zero
    try:
        y_div_zero = x / (x - 3)  # This will create a NaN gradient at x=3
        y_div_zero.backward(torch.tensor([1.0, 1.0]))
        print("Gradient with division by zero (should not reach here):", x.grad)
    except RuntimeError as e:
        print("RuntimeError encountered:", e)

    # Reset gradients for the next test
    x.grad.zero_()

    # Edge case 2: Log of a negative number
    try:
        y_log_negative = torch.log(x - 5)  # This will create NaN for x<5
        y_log_negative.backward(torch.tensor([1.0, 1.0]))
        print("Gradient with log of a negative number (should not reach here):", x.grad)
    except RuntimeError as e:
        print("RuntimeError encountered:", e)

    # Reset gradients for the next test
    x.grad.zero_()

    # Edge case 3: Square root of a negative number
    try:
        y_sqrt_negative = torch.sqrt(x - 5)  # This will create NaN for x<5
        y_sqrt_negative.backward(torch.tensor([1.0, 1.0]))
        print("Gradient with square root of a negative number (should not reach here):", x.grad)
    except RuntimeError as e:
        print("RuntimeError encountered:", e)

# Run the tests
test_tensor_operations()



#output:

# Gradient of y with respect to x: tensor([4., 6.])
# Gradient with division by zero (should not reach here): tensor([-3., nan])
# Gradient with log of a negative number (should not reach here): tensor([-0.3333, -0.5000])
# Gradient with square root of a negative number (should not reach here): tensor([nan, nan])





#analysis of results:

# 1. **Gradient of \(y\) with respect to \(x\)**: 
#    ```
#    Gradient of y with respect to x: tensor([4., 6.])
#    ```
#    - This is expected. The operation \(y = x^2\) results in \(\frac{dy}{dx} = 2x\). For \(x = 2.0\), the gradient
#  is \(2 \times 2 = 4\), and for \(x = 3.0\), the gradient is \(2 \times 3 = 6\).

# 2. **Gradient with division by zero**: 
#    ```
#    Gradient with division by zero (should not reach here): tensor([-3., nan])
#    ```
#    - This is partially expected. The gradient should be `nan` for the second element, as the operation involves
#  a division by zero when \(x\) approaches 3. However, the first element showing `-3` indicates that the backward 
# pass did not completely fail, but rather that it still computed a valid gradient before hitting the division by 
# zero. In practice, you might want to check the conditions leading to this case and possibly prevent calculating the 
# backward pass for values that will lead to `nan`.

# 3. **Gradient with log of a negative number**: 
#    ```
#    Gradient with log of a negative number (should not reach here): tensor([-0.3333, -0.5000])
#    ```
#    - This is unexpected. The logarithm of a negative number should result in `nan` for both elements, and the backward
# pass should not return valid gradients. The actual values imply that the operation somehow calculated gradients before 
# encountering an invalid log operation, which should not happen in a properly controlled scenario.

# 4. **Gradient with square root of a negative number**: 
#    ```
#    Gradient with square root of a negative number (should not reach here): tensor([nan, nan])
#    ```
#    - This output is expected. The square root of a negative number is not defined in the realm of real numbers, so the 
# gradients for both elements result in `nan`.

# ### Summary

# - The first gradient calculation is correct.
# - The division by zero should indeed result in `nan` without valid gradients for the first element, so the output here 
# indicates that you might want to handle that case more robustly.
# - The logarithm operation shows valid gradients unexpectedly; this needs to be addressed since it should not return valid
# gradients if it involves a negative number.
# - The square root of a negative number correctly returns `nan`.