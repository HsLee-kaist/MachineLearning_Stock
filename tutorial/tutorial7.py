import torch
import numpy as np
import pandas as pd
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler

class StatsDataset(Dataset):

    # Initialize your data, download, etc.
    def __init__(self):
        xy = np.loadtxt('./data/keystats.csv',
                        delimiter=',', skiprows=1, usecols = range(3,48), dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, 4:])
        kaist = xy[:, [1]]-xy[:, [3]]
        self.y_data = np.heaviside(kaist, 1)
        """
        if kaist == 0:
            self.y_data = 0
        else:
            self.y_data = torch.from_numpy(np.divide(np.add(np.divide((kaist), np.absolute(kaist)), 1), 2))
"""
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

dataset = StatsDataset()

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
#subsetA, subsetB = random_split(dataset, [train_size, test_size])
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
#x_train, x_test, y_train, y_test =  train_test_split(dataset.x_data,dataset.y_data)
#print(train_dataset)
#print(train_dataset.y_data)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=64,
                                            num_workers=2)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=64,
                                          shuffle=False)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.I1 = nn.Linear(41, 520)
        self.I2 = nn.Linear(520, 320)
        self.I3 = nn.Linear(320, 240)
        self.I4 = nn.Linear(240, 120)
        self.I5 = nn.Linear(120, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        out1 = self.sigmoid(self.I1(x))
        out2 = self.sigmoid(self.I2(out1))
        out3 = self.sigmoid(self.I3(out2))
        out4 = self.sigmoid(self.I4(out3))
        y_pred = self.sigmoid(self.I5(out4))

        return y_pred


model = Net()

criterion = nn.MSELoss(reduction='sum')
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data), Variable(target)
        output = model(data)
        # sum up batch loss
        test_loss += criterion(output, target).data.item()
        # get the index of the max
        pred = output.data.max(1, keepdim=True)[1]

        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
         test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':
    for epoch in range(1, 10):
        train(epoch)
        test()