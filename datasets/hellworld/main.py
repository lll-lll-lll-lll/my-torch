import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

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
