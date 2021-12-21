import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import model_urls

model_urls['resnet18'] = model_urls['resnet18'].replace('https://', 'http://')

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

if __name__=='__main__':
    net()