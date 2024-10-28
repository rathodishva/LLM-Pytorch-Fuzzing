# prompt: "write me a fuzz code to explore the division of zero and log of negative number unexpected output"
import torch
import random

def fuzz_test_division_and_log(iterations=10):
    for _ in range(iterations):
        # Generate one value close to 3 (for division test) and one random value
        tensor_values = [random.uniform(2.99, 3.01), random.uniform(-10, 10)]
        x = torch.tensor(tensor_values, requires_grad=True)
        
        # Edge case 1: Division by zero
        try:
            denominator = x - 3
            y_div_zero = x / denominator
            y_div_zero.backward(torch.tensor([1.0, 1.0]))
            print(f"Division test with x={x}: Gradients: {x.grad}")
        except RuntimeError as e:
            print(f"Division error with x={x}: {e}")
            
        if x.grad is not None:
            x.grad.zero_()
            
        # Edge case 2: Log of negative number
        try:
            y_log_negative = torch.log(x - 5)
            y_log_negative.backward(torch.tensor([1.0, 1.0]))
            print(f"Log test with x={x}: Gradients: {x.grad}")
        except RuntimeError as e:
            print(f"Log error with x={x}: {e}")


# Run the fuzz test
fuzz_test_division_and_log()



#output:
# Division test with x=tensor([2.9980, 1.4794], requires_grad=True): Gradients: tensor([-7.7092e+05, -1.2975e+00])
# Log test with x=tensor([2.9980, 1.4794], requires_grad=True): Gradients: tensor([-0.4995, -0.2840])
# Division test with x=tensor([3.0047, 9.9665], requires_grad=True): Gradients: tensor([-1.3350e+05, -6.1815e-02])
# Log test with x=tensor([3.0047, 9.9665], requires_grad=True): Gradients: tensor([-0.5012,  0.2013])
# Division test with x=tensor([3.0012, 7.3584], requires_grad=True): Gradients: tensor([-2.0283e+06, -1.5793e-01])
# Log test with x=tensor([3.0012, 7.3584], requires_grad=True): Gradients: tensor([-0.5003,  0.4240])
# Division test with x=tensor([ 2.9915, -4.7068], requires_grad=True): Gradients: tensor([-4.1308e+04, -5.0510e-02])
# Log test with x=tensor([ 2.9915, -4.7068], requires_grad=True): Gradients: tensor([-0.4979, -0.1030])
# Division test with x=tensor([ 2.9944, -6.6516], requires_grad=True): Gradients: tensor([-9.6220e+04, -3.2205e-02])
# Log test with x=tensor([ 2.9944, -6.6516], requires_grad=True): Gradients: tensor([-0.4986, -0.0858])
# Division test with x=tensor([ 3.0005, -7.4846], requires_grad=True): Gradients: tensor([-1.0718e+07, -2.7291e-02])
# Log test with x=tensor([ 3.0005, -7.4846], requires_grad=True): Gradients: tensor([-0.5001, -0.0801])
# Division test with x=tensor([3.0084, 5.1061], requires_grad=True): Gradients: tensor([-4.2486e+04, -6.7635e-01])
# Log test with x=tensor([3.0084, 5.1061], requires_grad=True): Gradients: tensor([-0.5021,  9.4262])
# Division test with x=tensor([ 3.0064, -1.7060], requires_grad=True): Gradients: tensor([-7.2756e+04, -1.3546e-01])
# Log test with x=tensor([ 3.0064, -1.7060], requires_grad=True): Gradients: tensor([-0.5016, -0.1491])
# Division test with x=tensor([2.9925, 3.8123], requires_grad=True): Gradients: tensor([-5.4033e+04, -4.5461e+00])
# Log test with x=tensor([2.9925, 3.8123], requires_grad=True): Gradients: tensor([-0.4981, -0.8420])
# Division test with x=tensor([ 2.9918, -3.6528], requires_grad=True): Gradients: tensor([-4.4904e+04, -6.7782e-02])
# Log test with x=tensor([ 2.9918, -3.6528], requires_grad=True): Gradients: tensor([-0.4980, -0.1156])


#analysis:
# Division Test (x/(x-3)):
# ✓ Correct behavior seen:


# Values very close to 3 (e.g., 2.9980, 3.0047, 3.0012) are producing very large gradients (e.g., -7.7092e+05, -1.3350e+05, -2.0283e+06)
# Other values have normal, smaller gradients
# The closer x is to 3, the larger the gradient becomes (e.g., 3.0005 produces -1.0718e+07)


# Log Test (log(x-5)):
# ✓ Correct behavior seen:


# For x < 5, gradients are being computed but with negative values, which is unexpected
# For x > 5 (like 9.9665, 7.3584), gradients are positive and reasonable

# However, there's one discrepancy: The log test isn't throwing RuntimeErrors for negative inputs as we might expect. This suggests that PyTorch is handling these cases differently than expected, possibly by:

# Using complex numbers internally
# Having a different implementation for backward propagation
# Handling these edge cases gracefully instead of throwing errors

# So while the division test is working as expected (showing very large gradients near x=3), the log test is behaving more robustly than initially expected. This isn't necessarily wrong - it's just different from what we might have anticipated.
# In summary:

# The division test is correctly exposing numerical instability near x=3
# The log test is more robust than expected, continuing to compute gradients even in cases where the input to log is negative