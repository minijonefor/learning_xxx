import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import torchvision.transforms as T
import numpy as np
import timeit


# class ChunkSampler(sampler.Sampler):
#     """Samples elements sequentially from some offset.
#     Arguments:
#         num_samples: # of desired datapoints
#         start: offset where we should start selecting from
#     """
#     def __init__(self, num_samples, start=0):
#         self.num_samples = num_samples
#         self.start = start
#
#     def __iter__(self):
#         return iter(range(self.start, self.start + self.num_samples))
#
#     def __len__(self):
#         return self.num_samples
#
# NUM_TRAIN = 49000
# NUM_VAL = 1000
#
# cifar10_train = dset.CIFAR10('./cs231n/datasets', train=True, download=True,
#                            transform=T.ToTensor())
# loader_train = DataLoader(cifar10_train, batch_size=64, sampler=ChunkSampler(NUM_TRAIN, 0))
#
# cifar10_val = dset.CIFAR10('./cs231n/datasets', train=True, download=True,
#                            transform=T.ToTensor())
# loader_val = DataLoader(cifar10_val, batch_size=64, sampler=ChunkSampler(NUM_VAL, NUM_TRAIN))
#
# cifar10_test = dset.CIFAR10('./cs231n/datasets', train=False, download=True,
#                           transform=T.ToTensor())
# loader_test = DataLoader(cifar10_test, batch_size=64)
#
# dtype = torch.FloatTensor
# print_every = 100
# def rest(m):
#     if hasattr(m, 'parameters'):
#         m.reset_parameters()

# xx = np.loadtxt('diabetes_data.csv.gz', delimiter=' ', dtype=np.float32)
# yy = np.loadtxt('diabetes_target.csv.gz', delimiter=' ', dtype=np.float32)
# x_data = torch.from_numpy(xx)
# y_data = torch.from_numpy(yy)
# print(x_data.shape)
# print(y_data.shape)
#
#
# class Model(torch.nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.linear1 = torch.nn.Linear(10, 6)
#         self.linear2 = torch.nn.Linear(6, 4)
#         self.linear3 = torch.nn.Linear(4, 1)
#         self.sigmoid = torch.nn.Sigmoid()
#
#     def forward(self, x):
#         x = self.sigmoid(self.linear1(x))
#         x = self.sigmoid(self.linear2(x))
#         x = self.sigmoid(self.linear3(x))
#         return x
#
#
# model = Model
# criterion = torch.nn.BCELoss(size_average=True)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
# x = model.forward(x_data)
#
# for epoch in range(100):
#     y_pred = model(x_data)

def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)


def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4), nn.ReLU())


def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block {i}', block1())
    return net


class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super(self).__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias
        return F


net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4), nn.ReLU())
net.apply(init_normal)
X = torch.rand(size=(2, 4))
for layer in net:
    x = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', x.shape)
print(net[1].bias)









