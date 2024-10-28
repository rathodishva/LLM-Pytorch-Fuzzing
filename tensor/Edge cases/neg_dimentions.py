# prompt: "Generate a script that creates tensors with negative dimensions or dimensions with zero and checks for error message"
import torch

def create_tensors_with_invalid_dimensions():
    # List of invalid dimensions to try (negative and zero dimensions)
    invalid_dimensions = [
        (-1, 3),   # Negative dimension
        (3, 0),    # Zero dimension
        (0, 5),    # Zero dimension
        (-2, -3),  # Negative dimensions
        (0, 0),    # All dimensions are zero
    ]

    for dimensions in invalid_dimensions:
        try:
            # Attempt to create a tensor with invalid dimensions
            tensor = torch.empty(dimensions)
            print(f"Successfully created tensor with dimensions {dimensions}: {tensor}")
        except Exception as e:
            print(f"Error creating tensor with dimensions {dimensions}: {e}")

# Call the function
create_tensors_with_invalid_dimensions()


#expected output:
# should print error messages for each dimension that raises an exception



# actual output:
# Error creating tensor with dimensions (-1, 3): Trying to create tensor with negative dimension -1: [-1, 3]
# Successfully created tensor with dimensions (3, 0): tensor([], size=(3, 0))
# Successfully created tensor with dimensions (0, 5): tensor([], size=(0, 5))
# Error creating tensor with dimensions (-2, -3): Trying to create tensor with negative dimension -2: [-2, -3]
# Successfully created tensor with dimensions (0, 0): tensor([], size=(0, 0))



#KEYNOTE
# despite the actual output matching the expected output, when placed into gpt to confirm, 
# gpt shows the same expected output but states "The output you provided does not fully meet expectations"
