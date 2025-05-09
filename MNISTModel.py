import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

class MNISTNN(nn.Module):
    def __init__(self):
        super(MNISTNN,self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(64*7*7,128),
            nn.ReLU(),
            nn.Linear(128,10)
        )
    def forward(self,x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
    
def train_model():
    transform = transforms.ToTensor()
    train_dataset = MNIST(root='./data',train=True,transform=transform,download=True)
    trainloader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )

    model = MNISTNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.001)

    epochs = 10 
    for epoch in range(epochs):
        total_loss = 0
        for images,labels in trainloader:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output,labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} / {epochs}: Loss = {total_loss:.4f}")
    torch.save(model.state_dict(),"MNISTNN.pth")
    print("The model is saved as MNISTNN.pth")

if __name__ == "__main__":
    train_model()