from typing import Literal
import torch
import torchvision
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

device = "cuda" if torch.cuda.is_available() else "cpu"

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






class CustomFashionMNIST:
    BATCH_SIZE = 64
    def __init__(self):
        pass

    def get_training_data(self)-> torchvision.datasets.FashionMNIST:
        training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
        )
        return training_data
    
    def get_test_data(self) -> torchvision.datasets.FashionMNIST:
        test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
        )
        return test_data   
    
    def creatre_dataloader(self, data):
        return DataLoader(data, batch_size=self.BATCH_SIZE)
    
    def x_shape_y_shape_dtypes(self, dataloader:DataLoader):
        for X, y in dataloader:
            print(f"Shape of X [N, C, H, W]: {X.shape}")
            print(f"Shape of y: {y.shape} {y.dtype}")
            break



    
if __name__ == "__main__":
    customFashion = CustomFashionMNIST()
    training_data, test_data = customFashion.get_training_data(), customFashion.get_test_data()
    train_dataloader = customFashion.creatre_dataloader(training_data)
    test_dataloader = customFashion.creatre_dataloader(test_data)
    model = CustomNeuralNetwork(device).to(device)
    model.exec(5,train_loader=train_dataloader, test_loader=test_dataloader)
    for acc, avg_loss in zip(model.acc, model.avg_loss):
        print("Accuracy: ",acc, "AVG: ", avg_loss)
    # model.save_model("test_model")
    # model = CustomNeuralNetwork(device)
    # model.load_state_dict(torch.load("test_model.pth"))
    # classes = [
    # "T-shirt/top",
    # "Trouser",
    # "Pullover",
    # "Dress",
    # "Coat",
    # "Sandal",
    # "Shirt",
    # "Sneaker",
    # "Bag",
    # "Ankle boot",
    # ]
    # model.eval()
    # x, y = test_data[0][0], test_data[0][1]
    # with torch.no_grad():
    #     pred = model(x)
    #     predicted, actual = classes[pred[0].argmax(0)], classes[y]
    #     print(f'Predicted: "{predicted}", Actual: "{actual}"')