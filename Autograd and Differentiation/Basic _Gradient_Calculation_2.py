# prompt: "write me a fuzz code to explore the division of zero and log of negative number unexpected output"
import torch
import random

def fuzz_test_division_and_log(iterations=10):
    for _ in range(iterations):
        # Generate random values for the tensor
        tensor_values = [random.uniform(-10, 10) for _ in range(2)]
        x = torch.tensor(tensor_values, requires_grad=True)

        # Edge case 1: Division by zero
        try:
            # Performing division that could lead to zero in the denominator
            denominator = x - 3
            y_div_zero = x / denominator
            y_div_zero.backward(torch.tensor([1.0, 1.0]))
            print(f"Division by zero test with x={x}: Gradients: {x.grad}")
        except RuntimeError as e:
            print(f"Division by zero encountered with x={x}: {e}")

        # Reset gradients for the next test
        x.grad.zero_()

        # Edge case 2: Log of a negative number
        try:
            # Attempting to compute log of x - 5, which could be negative
            y_log_negative = torch.log(x - 5)
            y_log_negative.backward(torch.tensor([1.0, 1.0]))
            print(f"Log of negative number test with x={x}: Gradients: {x.grad}")
        except RuntimeError as e:
            print(f"Log of negative number encountered with x={x}: {e}")

# Run the fuzz test
fuzz_test_division_and_log()



#output:
# Division by zero test with x=tensor([6.8819, 6.0454], requires_grad=True): Gradients: tensor([-0.1991, -0.3235])
# Log of negative number test with x=tensor([6.8819, 6.0454], requires_grad=True): Gradients: tensor([0.5314, 0.9566])
# Division by zero test with x=tensor([-5.2654,  7.1551], requires_grad=True): Gradients: tensor([-0.0439, -0.1738])
# Log of negative number test with x=tensor([-5.2654,  7.1551], requires_grad=True): Gradients: tensor([-0.0974,  0.4640])
# Division by zero test with x=tensor([-5.1617,  4.0953], requires_grad=True): Gradients: tensor([-0.0450, -2.5006])
# Log of negative number test with x=tensor([-5.1617,  4.0953], requires_grad=True): Gradients: tensor([-0.0984, -1.1053])
# Division by zero test with x=tensor([ 1.6360, -5.7308], requires_grad=True): Gradients: tensor([-1.6124, -0.0394])
# Log of negative number test with x=tensor([ 1.6360, -5.7308], requires_grad=True): Gradients: tensor([-0.2973, -0.0932])
# Division by zero test with x=tensor([1.8089, 9.5569], requires_grad=True): Gradients: tensor([-2.1146, -0.0698])
# Log of negative number test with x=tensor([1.8089, 9.5569], requires_grad=True): Gradients: tensor([-0.3134,  0.2194])
# Division by zero test with x=tensor([ 7.2458, -0.9703], requires_grad=True): Gradients: tensor([-0.1664, -0.1903])
# Log of negative number test with x=tensor([ 7.2458, -0.9703], requires_grad=True): Gradients: tensor([ 0.4453, -0.1675])
# Division by zero test with x=tensor([1.7132, 0.9212], requires_grad=True): Gradients: tensor([-1.8117, -0.6942])
# Log of negative number test with x=tensor([1.7132, 0.9212], requires_grad=True): Gradients: tensor([-0.3042, -0.2452])
# Division by zero test with x=tensor([-1.1256,  8.8160], requires_grad=True): Gradients: tensor([-0.1763, -0.0887])
# Log of negative number test with x=tensor([-1.1256,  8.8160], requires_grad=True): Gradients: tensor([-0.1633,  0.2621])
# Division by zero test with x=tensor([-3.7549, -1.7611], requires_grad=True): Gradients: tensor([-0.0657, -0.1323])
# Log of negative number test with x=tensor([-3.7549, -1.7611], requires_grad=True): Gradients: tensor([-0.1142, -0.1479])
# Division by zero test with x=tensor([-2.6974, -3.0038], requires_grad=True): Gradients: tensor([-0.0924, -0.0832])
# Log of negative number test with x=tensor([-2.6974, -3.0038], requires_grad=True): Gradients: tensor([-0.1299, -0.1249])



#analysis:

