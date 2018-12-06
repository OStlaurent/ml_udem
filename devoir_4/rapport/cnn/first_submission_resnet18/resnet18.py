# contains modified code from
# https://www.kaggle.com/leighplt/pytorch-starter-kit

#Resnet18

import torch
import torch.nn as nn
import torch.utils.data
import torchvision
from torchvision import transforms, utils
import numpy as np
import matplotlib.pyplot as plt
from label_transform import label_to_int

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
    plt.savefig("resnet_loss.png")
    plt.show()

    # accuracy graph
    plt.plot(epochs, accuracy_train, 'b', label='train')
    plt.plot(epochs, accuracy_valid, label='validation')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    # plt.figtext(0.99, 0.01, 'epoch={} batch_size={} learning_rate={} '.format(num_epochs, batch_size,learning_rate), horizontalalignment='right')
    plt.savefig("resnet_accuracy.png")
    plt.show()

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 38
num_classes = 31
batch_size = 16
learning_rate = 0.01

# Load numpy dataset
train_size = 8000
data_train = np.load('X_train_clean_bin_140_45.npy', encoding='latin1')
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
model = torchvision.models.resnet18(pretrained=False)

# change the model that is using 3 channels for input to 1 channel
def squeeze_weights(m):
    m.weight.data = m.weight.data.sum(dim=1)[:, None]
    m.in_channels = 1

model.conv1.apply(squeeze_weights)
model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
model.to(device)

criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True, weight_decay=0)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.01)
# multiply the learning rate by gamma after 10 epochs, 25 epochs, 35 epochs
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,25,35,50], gamma=0.1)

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
torch.save(model.state_dict(), 'resnet_valid.ckpt')

