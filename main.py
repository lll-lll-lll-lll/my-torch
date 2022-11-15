import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from network.hellworld import CustomNeuralNetwork


device = "cuda" if torch.cuda.is_available() else "cpu"

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