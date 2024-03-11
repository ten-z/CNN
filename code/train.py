import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import torch.nn.functional as F

def drawplot(num_epochs, label1, data1, label2, data2, xlabel, ylabel, title):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs+1), data1, label=label1, color='blue')
    plt.plot(range(1, num_epochs+1), data2, label=label2, color='red')
    plt.legend()
    plt.grid(True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

have_cuda = torch.cuda.is_available()
print("have_cuda=",have_cuda)

gpu_count = torch.cuda.device_count()
print("gpu_count=",gpu_count)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((300, 300)),  # Resize images to a consistent size
    transforms.ToTensor(),          # Convert images to tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize image values
])

train_data = '/content/drive/MyDrive/AIML421/final/traindata'

custom_dataset = ImageFolder(root=train_data, transform=transform)

# *** Show loaded training data ***
# import matplotlib.pyplot as plt
# import numpy as np
#
# # functions to show an image
#
# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
# batch_size = 4
# data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)
#
# # get some random training images
# dataiter = iter(data_loader)
# images, labels = next(dataiter)
#
# classes = ('cherry', 'strawberry', 'tomato')
#
# # show images
# imshow(torchvision.utils.make_grid(images))
# # print labels
# print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

# *** the Baseline code Start ***
seed_value = 42
torch.manual_seed(seed_value)

train_size = int(0.8 * len(custom_dataset))
validate_size = len(custom_dataset) - train_size
train_dataset, validate_dataset = torch.utils.data.random_split(custom_dataset, [train_size, validate_size])

batch_size = 100
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False)

class BaselineModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hiddenL1 = nn.Linear(300*300*3, 512)
        self.hiddenL2 = nn.Linear(512, 256)
        self.hiddenL3 = nn.Linear(256, 3)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.hiddenL1(x))
        x = F.relu(self.hiddenL2(x))
        x = self.hiddenL3(x)
        return x

baseline_model = BaselineModule().to(device)
print(baseline_model)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(baseline_model.parameters(), lr=0.001)

num_epochs = 30
baseline_train_accuracy = []
baseline_train_losses = []
baseline_validate_accuracy = []
baseline_validate_losses = []

for epoch in range(num_epochs):
    baseline_model.train()
    baseline_loss = 0.0
    baseline_correct = 0.0
    for X, Y in train_loader:
        X = X.to(device)
        Y = Y.to(device)

        # Feedforward
        output = baseline_model(X)
        loss = criterion(output, Y)

        _, predict = torch.max(output, 1)
        baseline_correct += (predict == Y).sum().item()

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        baseline_loss += loss.item()

    epoch_train_accuracy = baseline_correct/len(train_dataset)
    baseline_train_accuracy.append(epoch_train_accuracy)

    epoch_train_loss = baseline_loss / len(train_dataset)
    baseline_train_losses.append(epoch_train_loss)

    # Validation on the validation set
    baseline_model.eval()  # Set to evaluate mode
    validate_correct = 0.0
    validate_loss = 0.0
    with torch.no_grad():
        for X, Y in validate_loader:
            X = X.to(device)
            Y = Y.to(device)
            outputs = baseline_model(X)
            loss = criterion(outputs, Y)
            validate_loss += loss.item()

            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, 1)
            validate_correct += (predicted == Y).sum().item()

    baseline_validate_accuracy.append(validate_correct / len(validate_dataset))
    baseline_validate_losses.append(validate_loss / len(validate_dataset))

drawplot(num_epochs, 'Training Accuracy', baseline_train_accuracy, 'Validation Accuracy', baseline_validate_accuracy, 'Epoch', 'Accuracy', 'BaseLine Training and Validation Accuracy')
drawplot(num_epochs, 'Training Loss', baseline_train_losses, 'Validation Loss', baseline_validate_losses, 'Epoch', 'Loss', 'BaseLine Training and Validation Loss')

print("Validation Accuracy", max(baseline_validate_accuracy))

# *** the Baseline code End ***

# *** the Base CNN code Start ***
seed_value = 42
torch.manual_seed(seed_value)

train_size = int(0.8 * len(custom_dataset))
validate_size = len(custom_dataset) - train_size
train_dataset, validate_dataset = torch.utils.data.random_split(custom_dataset, [train_size, validate_size])

batch_size = 100
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False)

class CNN_Simple_Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 32 channels * 150*150
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 64*75*75
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=3) # 128*25*25
        self.fc1 = nn.Linear(128 * 25 * 25, 128)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        return x

cnn_simple_model = CNN_Simple_Module().to(device)
print(cnn_simple_model)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(cnn_simple_model.parameters(), lr=0.001)

num_epochs = 50
cnn_simple_train_accuracy = []
cnn_simple_train_losses = []
cnn_simple_validate_accuracy = []
cnn_simple_validate_losses = []

for epoch in range(num_epochs):
    cnn_simple_model.train()
    cnn_simple_loss = 0.0
    cnn_simple_correct = 0.0
    for X, Y in train_loader:
        X = X.to(device)
        Y = Y.to(device)

        # Feedforward
        output = cnn_simple_model(X)
        loss = criterion(output, Y)

        _, predict = torch.max(output, 1)
        cnn_simple_correct += (predict == Y).sum().item()

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cnn_simple_loss += loss.item()

    epoch_train_accuracy = cnn_simple_correct/len(train_dataset)
    cnn_simple_train_accuracy.append(epoch_train_accuracy)

    epoch_train_loss = cnn_simple_loss / len(train_dataset)
    cnn_simple_train_losses.append(epoch_train_loss)

    # Validation on the validation set
    cnn_simple_model.eval()  # Set to evaluate mode
    cnn_simple_validate_correct = 0.0
    cnn_simple_validate_loss = 0.0
    with torch.no_grad():
        for X, Y in validate_loader:
            X = X.to(device)
            Y = Y.to(device)
            outputs = cnn_simple_model(X)
            loss = criterion(outputs, Y)
            cnn_simple_validate_loss += loss.item()

            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, 1)
            cnn_simple_validate_correct += (predicted == Y).sum().item()

    cnn_simple_validate_accuracy.append(cnn_simple_validate_correct / len(validate_dataset))
    cnn_simple_validate_losses.append(cnn_simple_validate_loss / len(validate_dataset))

drawplot(num_epochs, 'Training Accuracy', cnn_simple_train_accuracy, 'Validation Accuracy', cnn_simple_validate_accuracy, 'Epoch', 'Accuracy', 'CNN Simple Training and Validation Accuracy')
drawplot(num_epochs, 'Training Loss', cnn_simple_train_losses, 'Validation Loss', cnn_simple_validate_losses, 'Epoch', 'Loss', 'CNN Simple Training and Validation Loss')

# *** the Base CNN code End ***

# *** the Best CNN code Start ***

class CNN_Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 32 channels * 150*150
        # self.dropout1 = nn.Dropout(p=0.3) nn.Dropout(p=0.2) nn.Dropout(p=0.4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 64*75*75
        # self.dropout2 = nn.Dropout(p=0.3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=3) # 128*25*25
        # self.dropout3 = nn.Dropout(p=0.3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2) # 256*12*12
        # self.dropout4 = nn.Dropout(p=0.3)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU()
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2) # 512*6*6
        # self.dropout5 = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(512 * 6 * 6, 128)
        self.relu6 = nn.ReLU()
        self.dropout6 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(128, 3)

        # self.fc1 = nn.Linear(512 * 6 * 6, 256)
        # self.relu6 = nn.ReLU()
        # self.dropout6 = nn.Dropout(p=0.3)
        # self.fc2 = nn.Linear(256, 64)
        # self.relu7 = nn.ReLU()
        # self.dropout7 = nn.Dropout(p=0.3)
        # self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x) # torch.sigmoid(x), torch.tanh(x)
        x = self.maxpool1(x)
        # x = self.dropout1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        # x = self.dropout2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        # x = self.dropout3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)
        # x = self.dropout4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.maxpool5(x)
        # x = self.dropout5(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu6(x)
        x = self.dropout6(x)
        x = self.fc2(x)
        return x

from sklearn.model_selection import KFold

num_folds = 5
seed_value = 42
batch_size = 50
# batch_size = 100
# batch_size = 20

# K-Fold Cross-Validation
k_fold = KFold(n_splits=num_folds, shuffle=True, random_state=seed_value)

cnn_train_accuracy = []
cnn_validate_accuracy = []
for fold, (train_index, validate_index) in enumerate(k_fold.split(custom_dataset)):

    train_dataset = torch.utils.data.Subset(custom_dataset, train_index)
    validate_dataset = torch.utils.data.Subset(custom_dataset, validate_index)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False)

    cnn_model = CNN_Module().to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
    # optimizer = optim.Adam(cnn_model.parameters(), lr=0.01)
    # optimizer = optim.Adam(cnn_model.parameters(), lr=0.1)
    # optimizer = optim.SGD(cnn_model.parameters(),lr=0.001, momentum=0.8)
    # optimizer = optim.SGD(cnn_model.parameters(),lr=0.001, momentum=0.9)
    # optimizer = optim.SGD(cnn_model.parameters(),lr=0.01, momentum=0.8)
    # optimizer = optim.SGD(cnn_model.parameters(),lr=0.01, momentum=0.9)
    # optimizer = optim.RMSprop(cnn_model.parameters(),lr=0.001, momentum=0.8)
    # optimizer = optim.RMSprop(cnn_model.parameters(),lr=0.001, momentum=0.9)
    # optimizer = optim.RMSprop(cnn_model.parameters(),lr=0.01, momentum=0.8)
    # optimizer = optim.RMSprop(cnn_model.parameters(),lr=0.01, momentum=0.9)

    num_epochs = 15
    cnn_t_acc = []

    for epoch in range(num_epochs):
        cnn_model.train()
        cnn_train_correct = 0.0
        for X, Y in train_loader:
            X = X.to(device)
            Y = Y.to(device)

            # Feedforward
            output = cnn_model(X)
            loss = criterion(output, Y)

            _, predict = torch.max(output, 1)
            cnn_train_correct += (predict == Y).sum().item()

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        cnn_t_acc.append(cnn_train_correct/len(train_dataset))

    # last epoch's acc
    cnn_train_accuracy.append(cnn_t_acc[-1])

    # after 15 epoch training, Validation on the validation set
    cnn_model.eval()  # Set to evaluate mode
    cnn_validate_correct = 0.0
    with torch.no_grad():
        for X, Y in validate_loader:
            X = X.to(device)
            Y = Y.to(device)
            outputs = cnn_model(X)
            loss = criterion(outputs, Y)

            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, 1)
            cnn_validate_correct += (predicted == Y).sum().item()

    cnn_validate_accuracy.append(cnn_validate_correct / len(validate_dataset))

drawplot(num_folds, 'Training Accuracy', cnn_train_accuracy, 'Validation Accuracy', cnn_validate_accuracy, 'Folds', 'Accuracy', 'CNN Training and Validation Accuracy')

print("Average validation accuracy:", np.mean(cnn_validate_accuracy))

# *** the Best CNN code End ***


# *** the Final CNN Model Start (For safe the model) ***
class CNN_Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 32 channels * 150*150
        # self.dropout1 = nn.Dropout(p=0.3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 64*75*75
        # self.dropout2 = nn.Dropout(p=0.3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=3) # 128*25*25
        # self.dropout3 = nn.Dropout(p=0.3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2) # 256*12*12
        # self.dropout4 = nn.Dropout(p=0.3)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU()
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2) # 512*6*6
        # self.dropout5 = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(512 * 6 * 6, 128)
        self.relu6 = nn.ReLU()
        self.dropout6 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        # x = self.dropout1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        # x = self.dropout2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        # x = self.dropout3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)
        # x = self.dropout4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.maxpool5(x)
        # x = self.dropout5(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu6(x)
        x = self.dropout6(x)
        x = self.fc2(x)
        return x

seed_value = 42
batch_size = 50
num_epochs = 15

train_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

cnn_model = CNN_Module().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    cnn_model.train()
    for X, Y in train_loader:
        X = X.to(device)
        Y = Y.to(device)

        # Feedforward
        output = cnn_model(X)
        loss = criterion(output, Y)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

PATH = '/content/drive/MyDrive/AIML421/final/final_train_model.pth'
torch.save(cnn_model.state_dict(), PATH)


# *** the Final CNN Model End (For safe the model) ***
