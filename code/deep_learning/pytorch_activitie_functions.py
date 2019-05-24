import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.autograd import Variable

x = torch.linspace(-5, 5, 200)
x = Variable(x)

x_np = x.data.numpy()

y_relu = F.relu(x).data.numpy()
y_sigmoid = torch.sigmoid(x).data.numpy()
y_tanh = torch.tanh(x).data.numpy()
y_softplus = F.softplus(x).data.numpy()


plt.figure(1, figsize=(8, 6))
plt.subplot(221)
plt.plot(x_np, y_relu, c='red', label='relu')
plt.ylim((-1, 5))
plt.legend(loc="best")

plt.show()


