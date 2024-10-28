#prompt : Generate a function that builds a complex computation graph involving multiple tensors and operations, and tests if the gradients are calculated correctly for each tensor in the graph. Log any discrepancies or errors

import torch

def complex_computation_graph():
    # Step 1: Create multiple tensors with requires_grad set to True
    a = torch.tensor([2.0, 3.0], requires_grad=True)
    b = torch.tensor([4.0, 5.0], requires_grad=True)
    c = torch.tensor([1.0, 1.0], requires_grad=True)
    
    # Step 2: Build the computation graph with various operations
    # Example operations
    # z1 = a * b + c
    z1 = a * b + c

    # z2 = sin(z1)
    z2 = torch.sin(z1)

    # z3 = log(z1) (ensure z1 > 0)
    if (z1 > 0).all():
        z3 = torch.log(z1)
    else:
        z3 = None

    # z4 = z2 * z3
    if z3 is not None:
        z4 = z2 * z3
    else:
        z4 = None

    # Step 3: Backward pass to calculate gradients
    if z4 is not None:
        z4.backward(torch.tensor([1.0, 1.0]))
        
        # Log the gradients
        print("Gradients:")
        print(f"a.grad: {a.grad}")
        print(f"b.grad: {b.grad}")
        print(f"c.grad: {c.grad}")
        
        # Step 4: Check for discrepancies or errors in gradients
        expected_grad_a = (b * torch.cos(z1))
        expected_grad_b = (a * torch.cos(z1))
        expected_grad_c = torch.ones_like(c)

        discrepancies = {
            "a": torch.allclose(a.grad, expected_grad_a, atol=1e-5),
            "b": torch.allclose(b.grad, expected_grad_b, atol=1e-5),
            "c": torch.allclose(c.grad, expected_grad_c, atol=1e-5)
        }

        for tensor_name, is_correct in discrepancies.items():
            if not is_correct:
                print(f"Discrepancy detected in gradients for tensor {tensor_name}!")
                print(f"Expected gradient for {tensor_name}: {expected_grad_a if tensor_name == 'a' else expected_grad_b if tensor_name == 'b' else expected_grad_c}")
                print(f"Actual gradient for {tensor_name}: {a.grad if tensor_name == 'a' else b.grad if tensor_name == 'b' else c.grad}")

    else:
        print("Skipping backward pass due to invalid log operation.")

# Run the function to test the computation graph
complex_computation_graph()


#output:
# Gradients:
# a.grad: tensor([ -7.8247, -13.3659])
# b.grad: tensor([-3.9123, -8.0196])
# c.grad: tensor([-1.9562, -2.6732])
# Discrepancy detected in gradients for tensor a!
# Expected gradient for a: tensor([-3.6445, -4.7883], grad_fn=<MulBackward0>)
# Actual gradient for a: tensor([ -7.8247, -13.3659])
# Discrepancy detected in gradients for tensor b!
# Expected gradient for b: tensor([-1.8223, -2.8730], grad_fn=<MulBackward0>)
# Actual gradient for b: tensor([-3.9123, -8.0196])
# Discrepancy detected in gradients for tensor c!
# Expected gradient for c: tensor([1., 1.])
# Actual gradient for c: tensor([-1.9562, -2.6732])


#expected output
#intrestingly chatgpt says the output does not match the expected behaviour where

# For tensor a: The actual gradient is significantly lower (more negative) than expected.
# For tensor b: The actual gradient is also lower than expected.
# For tensor c: The actual gradient is negative instead of positive.