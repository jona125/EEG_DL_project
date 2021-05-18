import torch
import torch.nn as nn
import torch.nn.functional as F

class Fire(nn.Module):

    def __init__(self, inputs, squeeze_num, expand1X1_num, expand3X3_num, func_type):
        super(Fire,self).__init__()
        self.squeeze = nn.Sequential(nn.Conv1d(inputs, squeeze_num, 1),
                                     func_type())

        self.expand1X1 = nn.Sequential(nn.Conv1d(squeeze_num, expand1X1_num, 1),
                                       func_type())

        self.expand3X3 = nn.Sequential(nn.Conv1d(squeeze_num,expand3X3_num, 3, padding=1),
                                       func_type())

    def forward(self, x):
        out = self.squeeze(x)
        return torch.cat([self.expand1X1(out),self.expand3X3(out)],1)


class SqueezeNet(nn.Module):

    def __init__(self,num_train, func_type, pool_type):
        super(SqueezeNet, self).__init__()
        
        self.layer1 = nn.Sequential(nn.Conv1d(num_train,25,10,padding=5),
                                    func_type(),
                                    pool_type(kernel_size=3),
                                    nn.BatchNorm1d(25,eps=1e-5,momentum=0.9))
        
                                   
        self.layer2 = nn.Sequential(nn.Conv1d(25,50,10,padding=5),
                                    func_type(),
                                    pool_type(kernel_size=3),
                                    nn.BatchNorm1d(50,eps=1e-5,momentum=0.9))
                                    
        self.layer3 = nn.Sequential(Fire(50,4,25,25,func_type),
                                    func_type(),
                                    pool_type(kernel_size=3),
                                    nn.BatchNorm1d(50,eps=1e-5,momentum=0.9))
        
        
        self.layer4 = nn.Sequential(nn.Conv1d(50,100,10,padding=5),
                                    func_type(),
                                    pool_type(kernel_size=3),
                                    nn.BatchNorm1d(100,eps=1e-5,momentum=0.9))


        self.layer5 = nn.Sequential(nn.Conv1d(100,200,10,padding=5),
                                    func_type(),
                                    pool_type(kernel_size=3),
                                    nn.BatchNorm1d(200,eps=1e-5,momentum=0.9))

        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(200 * 4, 4)



    def forward(self, x):

        out = self.layer1(x)
        out = self.drop_out(out)
        out = self.layer2(out)
        out = self.drop_out(out)
        out = self.layer3(out)
        out = self.drop_out(out)
        out = self.layer4(out)
        out = self.drop_out(out)
        out = self.layer5(out)
        out = self.drop_out(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        return out

        

