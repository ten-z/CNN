import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

have_cuda = torch.cuda.is_available()
print("have_cuda=",have_cuda)

gpu_count = torch.cuda.device_count()
print("gpu_count=",gpu_count)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((300, 300)),  # Resize images
    transforms.ToTensor(),          # Convert images to tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize image values
])

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

cnn_model = CNN_Module()

test_data = '/Users/teng/Documents/Victoria/AIML421/assignment_final/commit/testdata'

custom_dataset = ImageFolder(root=test_data, transform=transform)

batch_size = 50
test_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

PATH = '/Users/teng/Documents/Victoria/AIML421/assignment_final/commit/final_train_model.pth'
cnn_model.load_state_dict(torch.load(PATH) if torch.cuda.is_available() else torch.load(PATH, map_location=torch.device('cpu')))

cnn_model.to(device)
cnn_model.eval()  # Set to evaluate mode
cnn_correct = 0.0
with torch.no_grad():
    for X, Y in test_loader:
        X = X.to(device)
        Y = Y.to(device)
        outputs = cnn_model(X)

        probs = F.softmax(outputs, dim=1)
        _, predicted = torch.max(probs, 1)
        cnn_correct += (predicted == Y).sum().item()

print("Final CNN accuracy on test set:", (cnn_correct / len(custom_dataset)))
