from typing import Literal
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

class CustomNeuralNetwork(nn.Module):
    acc = []
    avg_loss = []
    # ネットワークのレイヤーを定義
    def __init__(self, device:Literal['cuda', 'cpu']):
        self.device = device
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    
    def forward(self,x:Tensor):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
    def get_loss(self):
        return nn.CrossEntropyLoss()
    
    def get_optimizer(self):
        return torch.optim.SGD(self.parameters(), lr=1e-3)
    
    def train_custom(self,dataloader:DataLoader):
        loss_fn = self.get_loss()
        optimizer = self.get_optimizer()
        size = len(dataloader.dataset)
        self.train()
        for batch, (X,y) in enumerate(dataloader):
            X,y = X.to(self.device), y.to(self.device)

            pred = self(X)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    def test_custom(self, dataloader:DataLoader):
        loss_fn = self.get_loss()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        self.eval()
        test_loss, correct = 0,0
        with torch.no_grad():
            for X, y in dataloader: 
                X, y = X.to(self.device), y.to(self.device)
                pred = self(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        self.acc.append(f"{(100*correct):>0.1f}")
        self.avg_loss.append(f"{test_loss:>8f}")
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    def exec(self, epochs: int, train_loader: DataLoader, test_loader: DataLoader):
        for t in range(epochs):
            print(f"Epoch: {t+1}--------------")
            self.train_custom(train_loader)
            self.test_custom(test_loader)
        print("DONE")

    def save_model(self, model_name:str):
        torch.save(self.state_dict(), f"{model_name}.pth")
        print(f"Saved PyTorch Model State to {model_name}.pth")

