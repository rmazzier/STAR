# test gpu with pytorch

import torch.nn.functional as F
import torch.nn as nn
import torch

print(torch.cuda.is_available())
print(torch.cuda.current_device())

print(torch.cuda.get_device_name(0))

# Do some test computations on gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

x = torch.tensor([1, 2, 3], device=device)
y = torch.tensor([4, 5, 6], device=device)

z = x + y
print(z)

# do a convolution on the gpu using the conv2d layer


# create a random tensor
x = torch.randn(1, 1, 28, 28, device=device)
print(x)

# create a convolutional layer
conv1 = nn.Conv2d(1, 32, 3).to(device)
print(conv1)

# apply the convolutional layer to the tensor
x = conv1(x)
print(x)
