import torch.nn as nn
from torchvision import models

def get_model(num_classes):
    model = models.efficientnet_b1(weights="DEFAULT")
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model
