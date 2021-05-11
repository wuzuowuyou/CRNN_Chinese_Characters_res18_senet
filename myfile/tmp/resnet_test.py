import torch
from resnet import *

# net = resnet18()

net = se_resnet18()

input = torch.randn(1,3,32,320)

out = net(input)

aa = 0

