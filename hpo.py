import argparse
import json
import logging
import os
import sys


import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
from torchvision import transforms, models

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

from smdebug import modes
from smdebug.pytorch import get_hook

def net():
    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features
    model.fc = nn.Sequential(
                   nn.Linear(num_features, 133))
    return model

def _get_train_data_loader(batch_size, training_dir):
    logger.info("Get train data loader")
    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224,224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    trainset = torchvision.datasets.ImageFolder(root=training_dir,
            transform=transform_train)

    return torch.utils.data.DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True
        )


def _get_test_data_loader(batch_size, test_dir):
    logger.info("Get test data loader")
    testing_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    testset = torchvision.datasets.ImageFolder(root=test_dir, 
            transform=testing_transform)
    return torch.utils.data.DataLoader(
            testset, 
            batch_size=batch_size,
            shuffle=False
        )

def train(args):
    logger.info("Hyperparameters: epoch: {}, lr: {}, batch size: {}, momentum: {}".format(
                    args.epochs, args.lr, args.batch_size, args.momentum)
    )
    train_loader = _get_train_data_loader(args.batch_size, args.train_dir)
    test_loader = _get_test_data_loader(args.test_batch_size, args.test_dir)

    model = net()
    loss_optim = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    hook = get_hook(create_if_not_exists=True)
    if hook:
        hook.register_loss(loss_optim)

    for epoch in range(1, args.epochs + 1):
        if hook:
            hook.set_mode(modes.TRAIN)
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_optim(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
        test(model, test_loader)
    save_model(model, args.model_dir)


def test(model, test_loader):
    print("Testing Model on Whole Testing Dataset")
    model.eval()
    running_loss=0
    running_corrects=0

    loss_optim = nn.CrossEntropyLoss()
    
    for inputs, labels in test_loader:
        outputs=model(inputs)
        loss=loss_optim(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()

    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects/ len(test_loader.dataset)
    logger.info(f"Testing Accuracy: {100*total_acc}, Testing Loss: {total_loss}")


def model_fn(model_dir):
    model = net()
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model

def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        metavar="N",
        help="input batch size for training (default: 10)",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=10,
        metavar="N",
        help="input batch size for testing (default: 10)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)"
    )

    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
    train(parser.parse_args())
