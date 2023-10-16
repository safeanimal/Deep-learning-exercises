# import torch
#
# data = torch.randn((3, 4))
# print(data)
# softmax_data = torch.softmax(data, dim=1)
# print(softmax_data)

import torch
from torch.utils.data import TensorDataset, DataLoader

# Create tensors
images = torch.randint(0, 255, size=(64, 64))
labels = torch.tensor([0, 1, 0], dtype=torch.long)

# Create a TensorDataset
dataset = TensorDataset(images, labels)

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Iterate through the DataLoader
for batch_features, batch_labels in dataloader:
    print("Batch Features:", batch_features)
    print("Batch Labels:", batch_labels)
