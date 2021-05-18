import torch
import torch.optim as optim
import torch.nn as nn
import timeit
from BasicCNN import *
from SqueezeNet import *

class Solver(object):

    def __init__(self, data, **kwargs):
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']

        # Unpack keyword arguments
        self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.weight_decay = kwargs.pop('weight_decay', 0.0)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.num_epochs = kwargs.pop('num_epochs', 10)
        self.learning_rate = kwargs.pop('learning_rate', 1e-3)

        self.loss_list = []
        self.train_acc_list = []
        self.val_acc_list = []
        self.val_loss_list = []


    def train(self,model_type = BasicCNN, func_type = nn.ReLU, pool_type = nn.AvgPool1d):

        self.loss_list = []
        self.train_acc_list = []
        self.val_acc_list = []
        self.val_loss_list = []
        

        num_trails = self.X_train.shape[0]
        model = model_type(self.X_train.shape[1],func_type, pool_type)
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay = self.weight_decay)


        # Train the model
        iterations_per_epoch = max(num_trails // self.batch_size, 1)

        for e in range(self.num_epochs):
            starttime = timeit.default_timer()
            for i in range(iterations_per_epoch):
                perm = torch.randperm(self.X_train.shape[0])
                idx = perm[:self.batch_size]
                X_batch = self.X_train[idx]
                y_batch = self.y_train[idx]

                # Run the forward pass
                outputs = model(X_batch.float())
                loss = criterion(outputs, y_batch.long())
                
                # Backprop and perform Adam optimisation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Print out progress
                if (i + 1) % 10 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(e + 1, self.num_epochs, i + 1, iterations_per_epoch, loss.item()))
            time_epoch = timeit.default_timer() - starttime
            self.loss_list.append(loss.item())

                
            # validation accuracy
            model.eval()
            tr_total = self.y_train.size(0)
            with torch.no_grad():
                outs = model(self.X_train.float())
                _, predicted = torch.max(outs.data, -1)
                tr_correct = (predicted == self.y_train).sum().item()
                self.train_acc_list.append(tr_correct / tr_total)
                
            te_total = self.y_val.size(0)
            with torch.no_grad():
                outs = model(self.X_val.float())
                loss = criterion(outs, self.y_val.long())
                self.val_loss_list.append(loss.item())
                _, predicted = torch.max(outs.data, -1)
                te_correct = (predicted == self.y_val).sum().item()
                self.val_acc_list.append(te_correct / te_total)
                print('Epoch [{}/{}], Train Accuracy: {:.2f}%, Test Accuracy: {:.2f}%, Time: {}s'.format(e + 1, self.num_epochs,(tr_correct / tr_total)*100,(te_correct / te_total)*100, time_epoch))


            model.train()

