import pytest
from datasets.hellworld import CustomFashionMNIST
from network.hellworld import CustomNeuralNetwork
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def test_custom_network_load():
    path = "models/helloworld/test_model.pth"
    customFashion = CustomFashionMNIST()
    test_data = customFashion.get_test_data()
    model = CustomNeuralNetwork(device)
    model.load_state_dict(torch.load(path))
    classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
    ]
    model.eval()
    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')
        assert predicted == actual