import torch
import torchvision
from torchvision import transforms
from PIL import Image
import os
from os import listdir
from matplotlib import pyplot
import random
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

size = 256
batch_size = 64

# TARGET : [isCat, isDog]

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transforms = transforms.Compose([
    transforms.Resize(size),
    transforms.CenterCrop(size),
    transforms.ToTensor(),
    normalize])

train_data_list = []
train_data = []
target_list = []
files = listdir('train/')
for i in range(len(listdir('train/'))):
    f = random.choice(files)
    files.remove(f)
    img = Image.open('train/' + f)
    img_tensor = transforms(img)
    # pyplot.imshow(img_tensor.reshape(size, size,3).numpy())
    # pyplot.show()
    train_data_list.append(img_tensor)
    isCat = 1 if 'cat' in f else 0
    isDog = 1 if 'dog' in f else 0
    target = [isCat, isDog]
    target_list.append(target)
    if len(train_data_list) >= batch_size:
        train_data.append((torch.stack(train_data_list), target_list))
        train_data_list = []
        target_list = []
        print('Loaded Batch {} of {}'.format(len(train_data), int(len(listdir('train/')) / batch_size)))
        print('Percentage Done : {:.3f} %'.format(100. * len(train_data) / int(len(listdir('train/')) / batch_size)))
    if len(train_data) == 200:
        break



class Netz(nn.Module):
    def __init__(self):
        super(Netz, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 12, kernel_size=5)
        self.conv3 = nn.Conv2d(12, 18, kernel_size=5)
        self.fc1 = nn.Linear(14112, 1000)
        self.fc2 = nn.Linear(1000, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        #print(x.size())
        #exit()
        x = x.view(-1, 14112)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x)


model = Netz()
model = model.cuda()

optimizer = optim.Adam(model.parameters(), lr=0.01)


def train(epoch):

    #model = torch.load('catdog.pt')
    model.train()
    batch_id = 0
    for data, target in train_data:
        data = data.cuda()
        target = torch.Tensor(target).cuda()
        data = Variable(data)
        target = Variable(target)
        optimizer.zero_grad()
        out = model(data)
        crierion = F.binary_cross_entropy
        loss = crierion(out, target)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_id * len(data),
                                                                       len(train_data),
                                                                       100. * batch_id / len(train_data),
                                                                       loss.item()))
        batch_id += 1
    torch.save(model, 'catdog.pt')


for epoch in range(0, 10):
    train(epoch)