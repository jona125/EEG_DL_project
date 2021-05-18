import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicCNN(nn.Module):
    # Basic Convultion network with Three layers of convolution with average pooling with FC netwotk.
    # Size of each filter 
    # Layer 1: 10X1 stride = 10, padding = 0

    def __init__(self,num_train, model_type, pool_type):
        super(BasicCNN, self).__init__()
        # 22 input channels, 15 output channels, 10X1 convolution
        self.layer1 = nn.Sequential(nn.Conv1d(num_train,25,10,padding=5),
                                    model_type(),
                                    pool_type(kernel_size=3),
                                    nn.BatchNorm1d(25,eps=1e-5,momentum=0.9))

        self.layer2 = nn.Sequential(nn.Conv1d(25,50,10,padding=5),
                                    model_type(),
                                    pool_type(kernel_size=3),
                                    nn.BatchNorm1d(50,eps=1e-5,momentum=0.9))

        self.layer3 = nn.Sequential(nn.Conv1d(50,100,10,padding=5),
                                    model_type(),
                                    pool_type(kernel_size=3),
                                    nn.BatchNorm1d(100,eps=1e-5,momentum=0.9))
        
        self.layer4 = nn.Sequential(nn.Conv1d(100,200,10,padding=5),
                                    model_type(),
                                    pool_type(kernel_size=3),
                                    nn.BatchNorm1d(200,eps=1e-5,momentum=0.9))
        
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(200 * 12, 4)


    def forward(self, x):

        out = self.layer1(x)
        out = self.drop_out(out)
        out = self.layer2(out)
        out = self.drop_out(out)
        out = self.layer3(out)
        out = self.drop_out(out)
        out = self.layer4(out)
        out = self.drop_out(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        return out

        

