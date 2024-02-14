import torch
from torchvision import models
import torch.nn as nn


class ModelFactory:
    @staticmethod
    def create_model(name, dataset):
        model = None

        # add yourself model here
        if name == "resnet18":
            model = models.resnet18()
        elif name == "resnet50":
            model = models.resnet50()
        elif name == "densenet121":
            model = models.densenet121()

        # 1. change the conv according to input feature
        channel_in = dataset[0][0].shape[0]
        model.conv1 = torch.nn.Conv2d(in_channels=channel_in,
                                      out_channels=64,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1,
                                      bias=False)

        # 2. change the fc layer by out feature
        num_labels = max(dataset.targets)+1
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_labels)

        if torch.cuda.is_available():
            model = model.cuda()
        else:
            model = model
        
        return model
