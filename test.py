import torch
from torch import nn

layer1 = nn.Sequential(nn.Conv1d(2115,200,10,stride = 10),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=4, stride=2),
                                    nn.Conv1d(200,50,7),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=2, stride=2),
                                    nn.Conv1d(50,25,3),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=2, stride=2),
                                    nn.Flatten(),
                                    nn.Linear(25 * 9, 120),
                                    nn.ReLU(),
                                    nn.Linear(120,40),
                                    nn.ReLU(),
                                    nn.Linear(40,10))


X = torch.rand(size=(22,2115,1000), dtype=torch.float32)
for layer in layer1:
    X = layer(X)
    print(layer.__class__.__name__,'output shape: \t',X.shape)
