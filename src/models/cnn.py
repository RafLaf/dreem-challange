import importlib 
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(7, 7, 6)
        self.pool = nn.AvgPool1d(3)
        self.conv2 = nn.Conv1d(7, 14, 12)
        self.fc1 = nn.Linear(9072, 5152)
        self.fc2 = nn.Linear(5152, 512)
        self.fc3 = nn.Linear(512, 48)
        self.fc4 = nn.Linear(48, 5)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = x.view(-1, 9072)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x), dim=0)
        return x


class CNN:
    def __init__(self):
        self.net = Net()
        self.net = self.net.float()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001)

    def fit(self, X, y, batch_size=4, epochs=10):
        y = y - 1.

        train_data = []
        for i in range(len(X)):
           train_data.append([X[i], y[i]])

        train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader):
                inputs, labels = data

                self.optimizer.zero_grad()

                outputs = self.net(inputs.float())
                print(outputs.shape)
                print(labels.shape)
                loss = self.criterion(outputs, labels.long())
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

    def predict(self, X):
        train_loader = torch.utils.data.DataLoader(dataset=X, batch_size=len(X), shuffle=False)
        for data in train_loader:
            return np.argmax(self.net(data.float()).detach().numpy(), axis=1)

def gen_model():
    return CNN()

