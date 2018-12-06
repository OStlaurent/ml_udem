# contains modified code from
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

##ResNet réduit


import torch
import torch.nn as nn
import torch.utils.data
import torchvision
from torchvision import transforms, utils
import numpy as np
import matplotlib.pyplot as plt
from label_transform import label_to_int

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=31):
        super(ResNet, self).__init__()
        planes = 64
        self.inplanes = planes
        self.conv1 = nn.Conv2d(1, planes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, planes, layers[0])
        planes = planes * 2
        self.layer2 = self._make_layer(block, planes, layers[1], stride=2)
        # planes = planes * 2
        # self.layer3 = self._make_layer(block, planes, layers[2], stride=2)
        # planes = planes*2
        # self.layer4 = self._make_layer(block, planes, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop_out = nn.Dropout2d()
        self.fc = nn.Linear(planes * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.drop_out(x)
        x = self.fc(x)

        return x


def resnet_m(**kwargs):
    """Constructs a ResNet model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def plot_result(epochs, loss_train, loss_valid):
    #correct the len if was interrupted during an epoch
    if len(epochs) == len(loss_train)+1:
        epochs = epochs[:-1]

    # loss curve
    plt.plot(epochs, loss_train, 'b', label='train')
    plt.plot(epochs, loss_valid, label='validation')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    # plt.figtext(0.99, 0.01, 'epoch={} batch_size={} learning_rate={} '.format(num_epochs, batch_size,learning_rate), horizontalalignment='right')
    plt.savefig("resnetm_loss.png")
    plt.show()

    # accuracy graph
    plt.plot(epochs, accuracy_train, 'b', label='train')
    plt.plot(epochs, accuracy_valid, label='validation')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    # plt.figtext(0.99, 0.01, 'epoch={} batch_size={} learning_rate={} '.format(num_epochs, batch_size,learning_rate), horizontalalignment='right')
    plt.savefig("resnetm_accuracy.png")
    plt.show()

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 100
num_classes = 31
batch_size = 16
learning_rate = 0.01

# Load numpy dataset
train_size = 8000
data_train = np.load('X_train_clean_bin_140_45.npy', encoding='latin1')
# data_train = np.load('X_train.npy', encoding='latin1')
X_train = data_train[:train_size]
X_valid = data_train[train_size:]

train_labels = np.loadtxt('input/train_labels.csv', delimiter=',', skiprows=1, dtype='str')
y = np.array([[label_to_int(e[1])] for e in train_labels])
y_train = y[:train_size]
y_valid = y[train_size:]
def resize(x):
    # resize an example (1000,) to (224,224)
    # which is the size needed for resnet
    #pad right and left
    pad_rl = np.zeros((100,62))
    x = np.hstack((pad_rl,x.reshape(100,100),pad_rl))
    # pad top and bottom
    pad_tb = np.zeros((62,224))
    x = np.vstack((pad_tb,x,pad_tb))
    return x

# Transform to torch tensors
tensor_X_train = torch.stack([torch.Tensor(resize(i)) for i in X_train])
tensor_y_train = torch.stack([torch.Tensor(i) for i in y_train])

tensor_X_valid = torch.stack([torch.Tensor(resize(i)) for i in X_valid])
tensor_y_valid = torch.stack([torch.Tensor(i) for i in y_valid])

# Transform to dataset
train_dataset = torch.utils.data.TensorDataset(tensor_X_train.unsqueeze(1),tensor_y_train)
valid_dataset = torch.utils.data.TensorDataset(tensor_X_valid.unsqueeze(1),tensor_y_valid)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# Not authorized to use pretrained/additionnal data
model = resnet_m()

model.to(device)

criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True, weight_decay=0)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0)
# multiply the learning rate by gamma after 10 epochs, 25 epochs, 35 epochs
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,40,50], gamma=0.5)

#variable for graph
epochs = []
loss_train = []
loss_valid = []
accuracy_train = []
accuracy_valid = []

# Train the model
total_step = len(train_loader)
total_step_valid = len(valid_loader)

# try makes it possible to send a keyboard interrupt to stop the learning and still have the graph and save the model
try:
    for epoch in range(num_epochs):
        epochs.append(epoch)
        correct = 0
        total = 0
        epoch_loss = 0
        scheduler.step()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            labels = labels.squeeze().long()
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            epoch_loss += loss.item()

            if (i + 1) % total_step == 0:
                # if (epoch + 1) % 100 == 0:
                # loss_train.append(loss.item())
                loss_train.append(epoch_loss/total)
                accuracy_train.append(correct / total)
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f} %'
                      .format(epoch + 1, num_epochs, i + 1, total_step, epoch_loss/total, 100 * correct / total))

        # Test the model
        model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            correct = 0
            total = 0
            epoch_loss = 0
            for i, (images, labels) in enumerate(valid_loader):
                images = images.to(device)
                labels = labels.to(device)
                labels = labels.squeeze().long()
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                epoch_loss += loss.item()

                if (i + 1) % total_step_valid == 0:
                    # if (epoch + 1) % 100 == 0:
                    # loss_valid.append(loss.item())
                    loss_valid.append(epoch_loss/total)
                    accuracy_valid.append(correct / total)
                    # stats_labels = labels.cpu().detach().clone().numpy()
                    # print('lab:', stats_labels)
                    # stats_predicted = predicted.cpu().detach().clone().numpy()
                    # # 0 * True == 0 * False == 0 so change categorie 0 to 32
                    # stats_predicted[stats_predicted == 0] = 32
                    # stats_labels[stats_labels == 0] = 32
                    # correct_categories = stats_predicted * (stats_predicted == stats_labels)
                    # print(correct_categories[np.nonzero(correct_categories)])
                    print('Test on validation set: Accuracy is {:.2f} % and Loss is {:.4f}'.format(100 * correct / total, epoch_loss/total))
        model.train()
# except Exception as e: print(e)
except:
    pass

plot_result(epochs, loss_train, loss_valid)

# Save the model checkpoint
torch.save(model.state_dict(), 'resnetm_valid.ckpt')

