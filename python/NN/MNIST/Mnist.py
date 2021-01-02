import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import time
import matplotlib.pyplot as plt

# from apex import amp

data_location = '../../../data_NN'

torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

kwargs = {'num_workers': 1, 'pin_memory': True}
plot = []


class Netz(nn.Module):
    def __init__(self):
        super(Netz, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv_dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 60)
        self.fc2 = nn.Linear(60, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.conv_dropout(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        # x = self.conv_dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x)


train_data = torch.utils.data.DataLoader(
    datasets.MNIST(data_location + '/MNIST/data', train=True, download=True, transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])), batch_size=1024, shuffle=True, **kwargs)

test_data = torch.utils.data.DataLoader(
    datasets.MNIST(data_location + '/MNIST/data', train=False, transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])), batch_size=1024, shuffle=True, **kwargs)

model = Netz()
model = model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.8)
# model, optimizer = amp.initialize(model, optimizer)

losses = []


def train(epoch):
    model = torch.load('../mnist.pt')
    model.train()
    for batch_id, (data, target) in enumerate(train_data):
        data = data.cuda()
        target = target.cuda()
        data = Variable(data)
        target = Variable(target)
        optimizer.zero_grad()
        out = model(data)
        criterion = F.nll_loss
        loss = criterion(out, target)
        loss.backward()
        plot.append(loss)
        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        # scaled_loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_id * len(data),
                                                                       len(train_data.dataset),
                                                                       100. * batch_id / len(train_data),
                                                                       loss.item()))
        torch.save(model, '../mnist.pt')


def test():
    model = torch.load('../mnist.pt')
    model.eval()
    loss = 0
    correct = 0
    print("Testing:")
    for data, target in train_data:
        data = Variable(data.cuda())
        target = Variable(target.cuda())
        out = model(data)
        loss += F.nll_loss(out, target, reduction='sum').item()
        prediction = out.argmax(dim=1, keepdim=True)
        correct += prediction.eq(target.view_as(prediction)).cpu().sum().item()

    testloss = loss / len(test_data.dataset)
    print("Durchschnitt:", testloss)
    print('Genauigkeit:', 100. * correct / (6 * len(test_data.dataset)), "%")


a = []
for epoch in range(1, 2000):
    if __name__ == '__main__':
        start = time.time()
        train(epoch)
        end = time.time()
        print("Zeit für Berechnung der Epoche:", end - start, "s")
        j = 0
        for i in range(len(plot)):
            j += plot[i].item()
        print("Durchschnitts Loss der jeweiligen Epoche:", j / len(plot))
        a.append(j / len(plot))
        test()
plt.plot(plot)
# plt.show()
