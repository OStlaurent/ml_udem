# contains modified code from
# https://www.kaggle.com/leighplt/pytorch-starter-kit

#
# Résultats: loss ~3740, accuracy ~6.6%
#     (^ après une éternité.)
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision
from torchvision import transforms, utils
import numpy as np
import matplotlib.pyplot as plt
from label_transform import label_to_int

def plot_result(epochs, loss_train, loss_valid): #, accuracy_train, accuracy_valid):
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
    #plt.plot(epochs, accuracy_train, 'b', label='train')
    #plt.plot(epochs, accuracy_valid, label='validation')
    #plt.ylabel('accuracy')
    #plt.xlabel('epoch')
    #plt.legend(loc='upper right')
    # plt.figtext(0.99, 0.01, 'epoch={} batch_size={} learning_rate={} '.format(num_epochs, batch_size,learning_rate), horizontalalignment='right')
    #plt.savefig("resnet_accuracy.png")
    #plt.show()

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 37
num_classes = 31
batch_size = 16
learning_rate = 0.01

# Load numpy dataset
train_size = 8000
data_train = np.load('X_train.npy', encoding='latin1')
X_train = data_train[:train_size]
X_valid = data_train[train_size:]

train_labels = np.loadtxt('input/train_labels.csv', delimiter=',', skiprows=1, dtype='str')
y = np.array([[label_to_int(e[1])] for e in train_labels])
y_train = y[:train_size]
y_valid = y[train_size:]
def resize(x):
    #pad right and left
    pad_rl = np.zeros((100,2), dtype='float')
    x = np.hstack((pad_rl,x.reshape(100,100),pad_rl))
    # pad top and bottom
    pad_tb = np.zeros((2,104), dtype='float')
    x = np.vstack((pad_tb,x,pad_tb))
    return x

# Transform to torch tensors
tensor_X_train = torch.stack([torch.Tensor(resize(i)) for i in X_train])
tensor_y_train = torch.stack([torch.Tensor(i) for i in y_train])

tensor_X_valid = torch.stack([torch.Tensor(resize(i)) for i in X_valid])
tensor_y_valid = torch.stack([torch.Tensor(i) for i in y_valid])

# Transform to dataset
train_dataset = torch.utils.data.TensorDataset(tensor_X_train.unsqueeze(1),tensor_X_train)
valid_dataset = torch.utils.data.TensorDataset(tensor_X_valid.unsqueeze(1),tensor_X_valid)

train_dataset_1 = torch.utils.data.TensorDataset(tensor_X_train.unsqueeze(1),tensor_y_train)
valid_dataset_1 = torch.utils.data.TensorDataset(tensor_X_valid.unsqueeze(1),tensor_y_valid)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

train_loader_1 = torch.utils.data.DataLoader(dataset=train_dataset_1,
                                           batch_size=batch_size,
                                           shuffle=True)

valid_loader_1 = torch.utils.data.DataLoader(dataset=valid_dataset_1,
                                          batch_size=batch_size,
                                          shuffle=False)

# Not authorized to use pretrained/additionnal data
#model = torchvision.models.resnet18(pretrained=False)

# change the model that is using 3 channels for input to 1 channel
#def squeeze_weights(m):
#    m.weight.data = m.weight.data.sum(dim=1)[:, None]
#    m.in_channels = 1

#model.conv1.apply(squeeze_weights)
#model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
#model.to(device)

class NetWrap():
  
  def __init__(self, model):
    self.model = model
  
  def forward2(self, x):
    return self.model.forward2(x)

class Net0(nn.Module):
    """A medium sized network that performs very well on MNIST."""
    
    def __init__(self):
        super().__init__()
        # conv block 1
        self.conv1 = nn.Conv2d(1, 31, 5)
        self.bn1 = nn.BatchNorm2d(31)
        
        # fully connected layers
        self.fc1 = nn.Linear(31*50*50, 1000)
        self.bn5 = nn.BatchNorm1d(1000)
        
        self.fc2 = nn.Linear(1000, 10816)
        
    def forward(self, x):
        # x is [batch_size, channels, heigth, width] = [bs, 1, 104, 104]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2) # x is [bs, 31, 50, 50]
        #print(x.shape)
        x = x.view(x.size(0), -1) # flatten
        #print(x.shape)
        x = F.relu(self.bn5(self.fc1(x)))
        
        x = F.relu(self.fc2(x))
        
        return x
    
    def forward2(self, x):
        # x is [batch_size, channels, heigth, width] = [bs, 1, 104, 104]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2) # x is [bs, 31, 50, 50]
        #print(x.shape)
        x = x.view(x.size(0), -1) # flatten
        #print(x.shape)
        x = F.relu(self.bn5(self.fc1(x)))
        
        return x

class Net1(nn.Module):
    """A medium sized network that performs very well on MNIST."""
    
    def __init__(self, net0):
        super().__init__()
        
        # fully connected layers
        self.fc1 = nn.Linear(1000, 100)
        
        self.fc2 = nn.Linear(100, 31)
        
        self.net0 = net0
    
    def forward(self, x):
        
        x = self.net0.forward2(x)
        
        x = F.relu(self.fc1(x))
        
        x = F.relu(self.fc2(x))
        
        return x

# try makes it possible to send a keyboard interrupt to stop the learning and still have the graph and save the model
def do_it_0(model, name):
  criterion = nn.MSELoss()
  #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True, weight_decay=0)
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.01)
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
  
  try:
  #if True:
      print("Train loop...")
      
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
              labels = labels.squeeze()
              loss = criterion(outputs, labels.view((batch_size,-1)))
  
              # Backward and optimize
              optimizer.zero_grad()
              loss.backward()
              optimizer.step()
  
              #_, predicted = torch.max(outputs.data, 1)
              #total += labels.size(0)
              total += 1
              #correct += (predicted == labels).sum().item()
              epoch_loss += loss.item()
  
              if (i + 1) % 100 == 0:
                  # if (epoch + 1) % 100 == 0:
                  # loss_train.append(loss.item())
                  loss_train.append(epoch_loss/total)
                  #accuracy_train.append(correct / total)
                  #print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f} %'
                  #  .format(epoch + 1, num_epochs, i + 1, total_step, epoch_loss/total, 100 * correct / total))
                  print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch + 1, num_epochs, i + 1, total_step, epoch_loss/total))
  
          # Test the model
          model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
          with torch.no_grad():
              correct = 0
              total = 0
              epoch_loss = 0
              for i, (images, labels) in enumerate(valid_loader):
                  images = images.to(device)
                  labels = labels.to(device)
                  labels = labels.squeeze()
                  outputs = model(images)
                  loss = criterion(outputs, labels.view((batch_size,-1)))
                  #_, predicted = torch.max(outputs.data, 1)
                  #total += labels.size(0)
                  total += 1
                  #correct += (predicted == labels).sum().item()
                  epoch_loss += loss.item()
  
                  if (i + 1) % total_step_valid == 0:
                      # if (epoch + 1) % 100 == 0:
                      # loss_valid.append(loss.item())
                      loss_valid.append(epoch_loss/total)
                      #accuracy_valid.append(correct / total)
                      # stats_labels = labels.cpu().detach().clone().numpy()
                      # print('lab:', stats_labels)
                      # stats_predicted = predicted.cpu().detach().clone().numpy()
                      # # 0 * True == 0 * False == 0 so change categorie 0 to 32
                      # stats_predicted[stats_predicted == 0] = 32
                      # stats_labels[stats_labels == 0] = 32
                      # correct_categories = stats_predicted * (stats_predicted == stats_labels)
                      # print(correct_categories[np.nonzero(correct_categories)])
                      print('Test on validation set: Loss is {:.4f}'.format(epoch_loss/total))
          model.train()
  except Exception as e: print(e)
  #except:
  #    pass
  
  try:
      plot_result(epochs, loss_train, loss_valid)
  except:
      pass

  # Save the model checkpoint
  torch.save(model.state_dict(), (name+'.ckpt'))

def do_it_1(model, name):
  criterion = nn.CrossEntropyLoss()
  #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True, weight_decay=0)
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.01)
  # multiply the learning rate by gamma after 10 epochs, 25 epochs, 35 epochs
  scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,25,35,50], gamma=0.1)
  
  #variable for graph
  epochs = []
  loss_train = []
  loss_valid = []
  accuracy_train = []
  accuracy_valid = []
  
  # Train the model
  total_step = len(train_loader_1)
  total_step_valid = len(valid_loader_1)
  
  try:
  #if True:
      print("Train loop...")
      
      for epoch in range(num_epochs):
          epochs.append(epoch)
          correct = 0
          total = 0
          epoch_loss = 0
          scheduler.step()
          for i, (images, labels) in enumerate(train_loader_1):
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
  
              #_, predicted = torch.max(outputs.data, 1)
              #total += labels.size(0)
              total += 1
              #correct += (predicted == labels).sum().item()
              epoch_loss += loss.item()
  
              if (i + 1) % 100 == 0:
                  # if (epoch + 1) % 100 == 0:
                  # loss_train.append(loss.item())
                  loss_train.append(epoch_loss/total)
                  #accuracy_train.append(correct / total)
                  #print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f} %'
                  #  .format(epoch + 1, num_epochs, i + 1, total_step, epoch_loss/total, 100 * correct / total))
                  print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch + 1, num_epochs, i + 1, total_step, epoch_loss/total))
  
          # Test the model
          model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
          with torch.no_grad():
              correct = 0
              total = 0
              epoch_loss = 0
              for i, (images, labels) in enumerate(valid_loader_1):
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
                      print('Test on validation set: Accuracy is {:.4f}, Loss is {:.4f}'.format(correct/total, epoch_loss/total))
          model.train()
  except Exception as e: print(e)
  #except:
  #    pass
  
  try:
      plot_result(epochs, loss_train, loss_valid)
  except:
      pass

  # Save the model checkpoint
  torch.save(model.state_dict(), (name+'.ckpt'))

# DO IT!

n0 = Net0()
n1 = Net1(NetWrap(n0))

print("\tNet0")
do_it_0(n0,"Net0-0")

print("\tNet1")
do_it_1(n1,"Net1-0")
