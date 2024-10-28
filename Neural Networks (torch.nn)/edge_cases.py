
# prompt:Generate a model with layers that use non-standard activations (e.g., extremely high negative values for ReLU input) and unusual weight
# initializations (e.g., very large or very small weights). Pass a variety of inputs through the model, focusing on potential overflow, NaN
# values, or divergence during training.
import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom activation function that simulates extreme negative values for ReLU
def extreme_relu(x):
    return F.relu(x)  # Standard ReLU can be used as a base, but with extreme inputs

# Define a custom neural network with unusual weight initializations
class UnusualModel(nn.Module):
    def __init__(self):
        super(UnusualModel, self).__init__()
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 5)
        self.fc3 = nn.Linear(5, 2)

        # Unusual weight initializations
        self.fc1.weight.data.fill_(1e6)  # Very large weights
        self.fc2.weight.data.fill_(1e-6)  # Very small weights
        self.fc3.weight.data.normal_(0, 1)  # Standard normal initialization for variety

    def forward(self, x):
        x = extreme_relu(self.fc1(x))  # Apply extreme ReLU
        x = extreme_relu(self.fc2(x))  # Apply extreme ReLU
        x = self.fc3(x)  # No activation for the output layer
        return x

def test_unusual_model():
    model = UnusualModel()
    model.train()  # Set the model to training mode

    # Create a variety of inputs
    inputs = [
        torch.tensor([[1.0, -1.0, 0.0]], dtype=torch.float32),
        torch.tensor([[1e10, -1e10, 1e5]], dtype=torch.float32),  # Extreme positive/negative values
        torch.tensor([[1e-10, 1e-10, 1e-10]], dtype=torch.float32),  # Extremely small values
        torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)  # Zero input
    ]

    for input_tensor in inputs:
        try:
            output = model(input_tensor)
            print(f"Input: {input_tensor.numpy()}, Output: {output.detach().numpy()}")
            if torch.isnan(output).any():
                print("Warning: NaN values detected in output!")
        except Exception as e:
            print(f"Error during forward pass with input {input_tensor.numpy()}: {str(e)}")

if __name__ == "__main__":
    test_unusual_model()
#output after run four times:
    # Neural Networks (torch.nn)/edge_cases.py"
    # Input: [[ 1. -1.  0.]], Output: [[-0.40168875 -0.17261839]]
    # Input: [[ 1.e+10 -1.e+10  1.e+05]], Output: [[-493255.8  742364.8]]
    # Input: [[1.e-10 1.e-10 1.e-10]], Output: [[-0.40168875 -0.17261839]]
    # Input: [[0. 0. 0.]], Output: [[-0.40168875 -0.17261839]]
    # (base) ishvarathod@Ishvas-MacBook-Pro LLM-Pytorch-Fuzzing % /usr/local/bin/python3 "/Users/ishvarathod/Documents/Uni/Year 2/Sem 2/Topics/LLM-Pytorch-Fuzzing/
    # Neural Networks (torch.nn)/edge_cases.py"
    # Input: [[ 1. -1.  0.]], Output: [[0.09080175 0.24614972]]
    # Input: [[ 1.e+10 -1.e+10  1.e+05]], Output: [[2390733.  2689050.8]]
    # Input: [[1.e-10 1.e-10 1.e-10]], Output: [[0.09080175 0.24614972]]
    # Input: [[0. 0. 0.]], Output: [[0.09080175 0.24614972]]
    # (base) ishvarathod@Ishvas-MacBook-Pro LLM-Pytorch-Fuzzing % /usr/local/bin/python3 "/Users/ishvarathod/Documents/Uni/Year 2/Sem 2/Topics/LLM-Pytorch-Fuzzing/
    # Neural Networks (torch.nn)/edge_cases.py"
    # Input: [[ 1. -1.  0.]], Output: [[ 0.36551976 -0.37039852]]
    # Input: [[ 1.e+10 -1.e+10  1.e+05]], Output: [[  141638.72 -1924270.4 ]]
    # Input: [[1.e-10 1.e-10 1.e-10]], Output: [[ 0.36551976 -0.37039852]]
    # Input: [[0. 0. 0.]], Output: [[ 0.36551976 -0.37039852]]
    # (base) ishvarathod@Ishvas-MacBook-Pro LLM-Pytorch-Fuzzing % /usr/local/bin/python3 "/Users/ishvarathod/Documents/Uni/Year 2/Sem 2/Topics/LLM-Pytorch-Fuzzing/
    # Neural Networks (torch.nn)/edge_cases.py"
    # Input: [[ 1. -1.  0.]], Output: [[-0.1816161  -0.40973657]]
    # Input: [[ 1.e+10 -1.e+10  1.e+05]], Output: [[-730849.5 -445547.3]]
    # Input: [[1.e-10 1.e-10 1.e-10]], Output: [[-0.1816161  -0.40973657]]
    # Input: [[0. 0. 0.]], Output: [[-0.1816161  -0.40973657]]



# analaysis:
    # This fuzzing approach has revealed:

    # Stability Issues:


    # Small/zero inputs produce consistent results within each run
    # Extreme inputs produce wildly varying results between runs
    # The model doesn't explode to NaN, which is actually interesting given the 1e6 weights


    # Limitations as a Fuzzer:


    # It's deterministic within each run
    # Only tests a fixed set of inputs
    # Doesn't explore weight space dynamically

# Regarding GPT as a fuzzer:
        # Strengths:

        # Good at generating structured test cases
        # Can create semantically meaningful edge cases
        # Understands the domain (neural networks) well enough to target likely failure points

        # Weaknesses:

        # Cannot dynamically adapt based on results
        # Generates relatively simple/predictable patterns
        # Doesn't leverage feedback from previous runs
        # May miss non-obvious edge cases that random fuzzing might find