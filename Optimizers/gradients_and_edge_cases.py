#promts: Create a scenario where an optimizer must work with sparse gradients (e.g., with embeddings or models with sparse layers). Test various optimizers with this setup, observing if they handle sparse gradients correctly or if any crashes or unexpected updates occur."

import torch
import torch.nn as nn
import torch.optim as optim
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Define a simple model with an embedding layer
class SparseEmbeddingModel(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(SparseEmbeddingModel, self).__init__()
        # The embedding layer with sparse gradients
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, sparse=True)
        self.linear = nn.Linear(embedding_dim, 1)
    
    def forward(self, x):
        embedded = self.embedding(x)
        # Average the embeddings for each batch (simulating some downstream task)
        embedded = embedded.mean(dim=1)
        return self.linear(embedded)

# Initialize the model, dummy data, and loss function
num_embeddings = 1000
embedding_dim = 10
model = SparseEmbeddingModel(num_embeddings, embedding_dim)
criterion = nn.MSELoss()

# Dummy data: Batch of indices representing words or items
batch_size = 32
x = torch.randint(0, num_embeddings, (batch_size, 5))
y = torch.randn(batch_size, 1)

# Test various optimizers with sparse gradients
optimizers = {
    "SparseSGD": optim.SGD(model.parameters(), lr=0.1),
    "SparseAdam": optim.SparseAdam(model.embedding.parameters(), lr=0.1),
    "Adam": optim.Adam(model.parameters(), lr=0.1)  # Adam may not support sparse embeddings fully
}

# Training loop
def train_with_sparse_gradients(model, x, y, optimizer_name, optimizer):
    model.train()
    prev_params = [p.clone().detach() for p in model.parameters()]

    logging.info(f"\nTesting {optimizer_name} with sparse gradients:")
    
    for epoch in range(3):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()

        try:
            optimizer.step()
        except Exception as e:
            logging.error(f"Error with {optimizer_name} during optimizer step: {e}")
            return
        
        # Log epoch loss
        logging.info(f"Epoch {epoch + 1}, Loss: {loss.item()}")
        
        # Check for unexpected parameter updates or errors
        for idx, (p, prev_p) in enumerate(zip(model.parameters(), prev_params)):
            if torch.isnan(p).any():
                logging.error(f"NaN encountered in parameter group {idx} with {optimizer_name}")
            elif torch.equal(p, prev_p):
                logging.warning(f"No update in parameter group {idx} with {optimizer_name}")
            prev_params[idx] = p.clone().detach()

# Run tests for each optimizer
for optimizer_name, optimizer in optimizers.items():
    train_with_sparse_gradients(model, x, y, optimizer_name, optimizer)


#output:
# Testing SparseSGD with sparse gradients:
# Epoch 1, Loss: 1.5889273881912231
# Epoch 2, Loss: 1.490585446357727
# Epoch 3, Loss: 1.4130926132202148

# Testing SparseAdam with sparse gradients:
# Epoch 1, Loss: 1.3504838943481445
# No update in parameter group 1 with SparseAdam
# No update in parameter group 2 with SparseAdam
# Epoch 2, Loss: 1.1492887735366821
# No update in parameter group 1 with SparseAdam
# No update in parameter group 2 with SparseAdam
# Epoch 3, Loss: 0.9729595184326172
# No update in parameter group 1 with SparseAdam
# No update in parameter group 2 with SparseAdam

# Testing Adam with sparse gradients:
# Error with Adam during optimizer step: Adam does not support sparse gradients, please consider SparseAdam instead




#Expected Outcome
# SparseSGD and SparseAdam: These should handle sparse gradients without issues.
# Adam: If it fails to handle sparse gradients, an error will be logged, revealing its incompatibility with sparse embeddings.