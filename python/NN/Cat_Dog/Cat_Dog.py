import torch
import torchvision
from torchvision import transforms
from PIL import Image
import os
from os import listdir
from matplotlib import pyplot as plt
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
files = listdir('../../../../data_NN/Cat_Dog/train/')
for i in range(len(listdir('../../../../data_NN/Cat_Dog/train//'))):
    f = random.choice(files)
    files.remove(f)
    img = Image.open('../../../../data_NN/Cat_Dog/train/' + f)
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
        print('Loaded Batch {} of {}'.format(len(train_data), int(len(listdir('../../../../data_NN/Cat_Dog/train/')) / batch_size)))
        print('Percentage Done : {:.3f} %'.format(100. * len(train_data) / int(len(listdir('../../../../data_NN/Cat_Dog/train/')) / batch_size)))
    if len(train_data) == 15:
        break


class Netz(nn.Module):
    def __init__(self):
        super(Netz, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=3)
        self.conv3 = nn.Conv2d(12, 24, kernel_size=3)
        self.conv4 = nn.Conv2d(24, 48, kernel_size=3)
        self.conv5 = nn.Conv2d(48, 96, kernel_size=3)
        self.conv6 = nn.Conv2d(96, 192, kernel_size=3)
        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 2)

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
        x = self.conv4(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.dropout1(x)
        x = x.view(-1, 768)
        x = F.relu(self.dropout2(self.fc1(x)))
        x = self.fc2(x)
        return F.sigmoid(x)


model = Netz()
model = model.cuda()

if os.path.isfile('catdog.pt'):
    model = torch.load('catdog.pt')

optimizer = optim.RMSprop(model.parameters(), lr=1e-4)


# optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(epoch):
    # model = torch.load('catdog.pt')
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
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_id,
                                                                       len(train_data),
                                                                       100. * batch_id / len(train_data),
                                                                       loss.item()))
        batch_id += 1
    torch.save(model, 'catdog.pt')

def test():
    model.eval()
    files = listdir('../../../../data_NN/Cat_Dog/test/')
    f = random.choice(files)
    img = Image.open('../../../../data_NN/Cat_Dog/test/' + f)
    img_eval_tensor = transforms(img)
    img_eval_tensor.unsqueeze_(0)
    data = Variable(img_eval_tensor.cuda())
    out = model(data)
    print(str(f) + ": " + str(out.data.max(1, keepdim=True)[1]))
    plt.imshow(img)
    plt.show()

def test_own(file):
    model.eval()
    img = Image.open(file)
    img_eval_tensor = transforms(img)
    img_eval_tensor.unsqueeze_(0)
    data = Variable(img_eval_tensor.cuda())
    out = model(data)
    print(str(file) + ": " + str(out.data.max(1, keepdim=True)[1]))
    plt.imshow(img)
    plt.show()

#for epoch in range(0, 5):
   # train(epoch)
test()
test_own('test.jpeg')
