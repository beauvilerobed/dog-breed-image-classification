#TODO: Import your dependencies.
import io
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

#TODO: Import dependencies for Debugging andd Profiling

def test(model, test_loader):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )

def train(model, train_loader, valid_loader, criterion, optimizer, epoch):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''

    # train the model
    for i in range(epoch):
        print("START TRAINING")
        model.train()
        train_loss = 0
        for _, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print("START VALIDATING")
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for _, (inputs, targets) in enumerate(valid_loader):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        print(
            "Epoch %d: train loss %.3f, val loss %.3f"
            % (i, train_loss, val_loss)
        )

def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 10))
    return model    

def main():
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    batch_size=10

    training_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
            download=True, transform=training_transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
            shuffle=True)
    
    transform_valid = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]
    )

    validset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_valid
    )

    valid_loader = torch.utils.data.DataLoader(
        validset,
        batch_size=batch_size,
        shuffle=False
    )

    model=train(model, train_loader, valid_loader, loss_criterion, optimizer)

    '''
    TODO: Test the model to see its accuracy
    '''
    
    testing_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
        download=True, transform=testing_transform)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
        shuffle=False)

    test(model, test_loader, loss_criterion)
    
    '''
    TODO: Save the trained model
    '''
    buffer = io.BytesIO()
    torch.save(model, buffer)

if __name__=='__main__':
    main()
