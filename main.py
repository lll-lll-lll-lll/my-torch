import torch
from datasets.hellworld import CustomFashionMNIST

from network.hellworld import CustomNeuralNetwork

device = "cuda" if torch.cuda.is_available() else "cpu"
    
if __name__ == "__main__":
    customFashion = CustomFashionMNIST()
    training_data, test_data = customFashion.get_training_data(), customFashion.get_test_data()
    train_dataloader = customFashion.creatre_dataloader(training_data)
    test_dataloader = customFashion.creatre_dataloader(test_data)
    model = CustomNeuralNetwork(device).to(device)
    model.exec(5,train_loader=train_dataloader, test_loader=test_dataloader)
    for acc, avg_loss in zip(model.acc, model.avg_loss):
        print("Accuracy: ",acc, "AVG: ", avg_loss)
    