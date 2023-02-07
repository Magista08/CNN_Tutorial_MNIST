# General
import os
import time

# ML
import torch.optim
from torch.utils.data import DataLoader
from torch import nn
from torchvision.datasets import mnist
from torchvision.transforms import ToTensor
from structure import NeuralNetwork

# Plotting
import matplotlib.pyplot as plt

# Global
if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
TRAIN_LOCATION = "./train_data"
TEST_LOCATION = "./test_data"
BATCH_SIZE = 256
EPOCHS = 100


def train(data_loader, nn_model, loss_fn_train, optimizer_train):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    nn_model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(DEVICE), y.to(DEVICE)

        # Compute prediction error
        pred = nn_model(X)
        loss = loss_fn_train(pred, y)

        # Computation
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer_train.zero_grad()
        loss.backward()
        optimizer_train.step()

        # Info Showing
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    train_loss /= num_batches
    correct /= size
    return train_loss, correct


def test(data_loader, nn_model, loss_fn_test):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    nn_model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in data_loader:
            # Test
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = nn_model(X)

            # Errors
            test_loss += loss_fn_test(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    # Calculate errors
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")
    return test_loss, correct


if __name__ == '__main__':
    # Download the train data and test data
    if not os.path.isdir("train_data"):
        train_get_control = True
    else:
        train_get_control = False
    train_dataset = mnist.MNIST(root=TRAIN_LOCATION, train=True, download=train_get_control, transform=ToTensor())

    if not os.path.isdir("test_data"):
        test_get_control = True
    else:
        test_get_control = False
    test_dataset = mnist.MNIST(root=TEST_LOCATION, train=False, download=test_get_control, transform=ToTensor())

    # Load Data
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Init the model
    model = NeuralNetwork().to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Init data list
    train_losses, train_correct = list(), list()
    test_losses,  test_correct = list(), list()

    # Start Train
    print("Start Trainning")

    for t in range(EPOCHS):
        # Train
        print(f"Epoch {t + 1}\n-----------------------------------------")
        beginning_time = time.time()
        temp_loss, temp_correct = train(train_loader, model, loss_fn, optimizer)
        temp_training_time = time.time()
        print(f"Training time:{temp_training_time - beginning_time}")

        # Record
        train_losses.append(temp_loss)
        train_correct.append(1 - temp_correct)

        # Test
        temp_loss, temp_correct = test(test_loader, model, loss_fn)

        # Record
        test_losses.append(temp_loss)
        test_correct.append(1 - temp_correct)

    # Init figure
    fig = plt.figure()
    axes1 = plt.subplot(1, 2, 1)
    axes2 = plt.subplot(1, 2, 2)

    # Draw training data
    axes1.errorbar(range(EPOCHS), train_losses, yerr=train_correct, label="Train Data")
    axes1.set_xlabel("Epoches")
    axes1.set_ylabel("Traindata")
    axes1.set_title("Training Data Records")

    # Draw testing data
    axes2.errorbar(range(EPOCHS), test_losses, yerr=test_correct, label="Test Data")
    axes2.set_xlabel("Epoches")
    axes2.set_ylabel("Test Losses")
    axes2.set_title("Testing Data Records")

    # Show the picture
    plt.show()

    # Finish the code
    print("DONE!")
