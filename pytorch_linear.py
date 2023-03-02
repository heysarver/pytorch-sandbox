# https://www.geeksforgeeks.org/linear-transformation-to-incoming-data-in-pytorch/

# Python program to apply Linear transform
# to incoming data
# Step 1: Importing PyTorch
import torch
  
# Step 2: Define incoming data as torch 
# tensor (float32)
data = torch.randn(3, 5)
print("Data before Transformation:\n", data)
#print("dtype of Data:\n", data.dtype)
#print("Size of Data:\n", data.size())
  
# Step 3: Define the in_features, out_features
in_features = 5
out_features = 4
  
# Step 4: Define a linear transformation
linear = torch.nn.Linear(in_features, out_features)
  
# Step 5: Apply the Linear transformation to 
# the tensor
data_out = linear(data)
print("Data after Transformation:\n", data_out)
#print("Size of Data after Transformation:\n", data_out.size())
