import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

#First
#design model using variables
xy = np.loadtxt('./data/diabetes.csv.gz', delimiter=',', dtype=np.float32)
x_data = Variable(torch.from_numpy(xy[:, 0:-1]))
y_data = Variable(torch.from_numpy(xy[:, [-1]]))

print(x_data.data.shape)
print(y_data.data.shape)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(8, 6)
        self.l2 = torch.nn.Linear(6, 4)
        self.l3 = torch.nn.Linear(4, 1)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        y_pred = self.sigmoid(self.l3(out2))
        return y_pred

#our model
model = Model()

#Second
#Construct loss and optimizer

criterion = torch.nn.BCELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#Third
#Training Cycle

for epoch in range(1000):
    y_pred = model(x_data)

    loss = criterion(y_pred, y_data)
    print(epoch, loss.data.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# After Training
hour_var = Variable(torch.Tensor([[-0.29412, 0.487437, 0.180328, -0.29293, 0, 0.00149, -0.53117, -0.03333]]))
print("predict 1 hour", 1.0, model.forward(hour_var).data.item()>0.5)
