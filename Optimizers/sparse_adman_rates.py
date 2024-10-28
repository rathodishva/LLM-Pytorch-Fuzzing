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
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, sparse=True)
        self.linear = nn.Linear(embedding_dim, 1)
    
    def forward(self, x):
        embedded = self.embedding(x)
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

# List of learning rates to test
learning_rates = [1e-4, 1e-3, 1e-2, 1e-1]  # Various learning rates for SparseAdam

# Training loop for SparseAdam with different learning rates
def train_with_sparse_adam(model, x, y, learning_rate):
    model.train()
    optimizer = optim.SparseAdam(model.embedding.parameters(), lr=learning_rate)
    prev_params = [p.clone().detach() for p in model.parameters()]

    logging.info(f"\nTesting SparseAdam with learning rate: {learning_rate}:")
    
    for epoch in range(3):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()

        try:
            optimizer.step()
        except Exception as e:
            logging.error(f"Error with SparseAdam during optimizer step: {e}")
            return
        
        # Log epoch loss
        logging.info(f"Epoch {epoch + 1}, Loss: {loss.item()}")
        
        # Check for unexpected parameter updates or errors
        for idx, (p, prev_p) in enumerate(zip(model.parameters(), prev_params)):
            if torch.isnan(p).any():
                logging.error(f"NaN encountered in parameter group {idx} with SparseAdam at learning rate {learning_rate}")
            elif torch.equal(p, prev_p):
                logging.warning(f"No update in parameter group {idx} with SparseAdam at learning rate {learning_rate}")
            prev_params[idx] = p.clone().detach()

# Run tests for each learning rate
for lr in learning_rates:
    train_with_sparse_adam(model, x, y, lr)


# expected outcome:
# Lower Learning Rates (e.g., 1e-4, 1e-3): These may lead to smaller updates, possibly resulting in slow convergence. If there are no updates, warnings will be logged.
# Moderate Learning Rates (e.g., 1e-2): These should provide a balance, allowing the optimizer to learn without causing instability.
# High Learning Rates (e.g., 1e-1): These might lead to larger updates, possibly causing instability or divergence, reflected in rapid fluctuations in loss.



#fuzzing analysis:
# In the fuzzing analysis of the SparseAdam optimizer with various learning rates, the primary observation was the consistent lack of parameter 
# updates across all configurations, which is unexpected behavior given that the loss values were decreasing over epochs. While a reduction in loss
# typically indicates that the model is learning, the absence of updates suggests potential issues in how the optimizer handles sparse gradients,
# particularly with the embedding layer. This could point to a flaw in the optimizer's implementation or its sensitivity to the input data's sparsity.
# Additionally, the lack of updates raises concerns about whether the gradients computed during backpropagation are effectively driving parameter changes, 
# which could hinder model convergence. No explicit bugs were encountered, but the unexpected behavior highlights the need for further investigation into
# the optimizer's performance with sparse data and whether the chosen learning rates were suitable for facilitating meaningful updates.






