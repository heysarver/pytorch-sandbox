import torch
import torch.nn as nn

# Input size
n_in = 10
# Hidden layer size
n_h = 5
# Output size
n_out = 1
# Batch size
batch_size = 10

# Create dummy input and target tensors
x = torch.randn(batch_size, n_in)
y = torch.tensor([[1.0], [0.0], [0.0], [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0]])

# Create model
model = nn.Sequential(nn.Linear(n_in, n_h),
  nn.ReLU(),
  nn.Linear(n_h, n_out),
  nn.Sigmoid())

# Construct the loss function
criterion = torch.nn.MSELoss()

# Construct the optimizer (Stochastic Gradient Descent in this case)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

# Gradient Descent
for epoch in range(5000):
  # Forward pass: Compute predicted y by passing x to the model
  y_pred = model(x)
  # Compute and print loss
  loss = criterion(y_pred, y)
  print('epoch: ', epoch,' loss: ', loss.item())
  # Zero gradients, perform a backward pass, and update the weights.
  optimizer.zero_grad()
  # perform a backward pass (backpropagation)
  loss.backward()
  # Update the parameters
  optimizer.step()
