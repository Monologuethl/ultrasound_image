import torch
import numpy as np

np_data = np.arange(6).reshape((2, 3))

torch_data = torch.from_numpy(np_data)

print(np_data)
print(torch_data)

data = [-1, -2, 1, 2]
tensor = torch.FloatTensor(data)
print(np.abs(data))
print(torch.abs(tensor))

data = [[1, 2], [3, 4]]

tensor = torch.FloatTensor(data)

print(np.matmul(data, data))
print(torch.mm(tensor, tensor))









