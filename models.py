# -*- coding: UTF-8 -*-
import torch.nn as nn
import settings
import torchvision.models as models
import torchvision
# CNN Model (2 conv layer)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU(),
            nn.MaxPool2d(2))
        # self.layer_vgg = nn.Sequential(*list(vgg.features._modules.values())[:-1])
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU(),
            nn.MaxPool2d(2))
            # nn.AdaptiveAvgPool2d((1,1)))
        self.fc = nn.Sequential(
            # nn.AdaptiveAvgPool2d((1,1)),
            nn.Linear((settings.IMAGE_WIDTH//8)*(settings.IMAGE_HEIGHT//8)*64, 1024),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU())
        self.rfc = nn.Sequential(
            nn.Linear(1024, settings.MAX_CAPTCHA*settings.ALL_CHAR_SET_LEN),
        )

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = out3.view(out3.size(0), -1)
        out5 = self.fc(out4)
        out6 = self.rfc(out5)
        return out6



class RES18(nn.Module):
    def __init__(self):
        super(RES18, self).__init__()
        self.num_cls = settings.MAX_CAPTCHA*settings.ALL_CHAR_SET_LEN
        self.base = torchvision.models.resnet18(pretrained=False)
        self.base.fc = nn.Linear(self.base.fc.in_features, self.num_cls)
    def forward(self, x):
        out = self.base(x)
        return out

class RES34(nn.Module):
    def __init__(self):
        super(RES34, self).__init__()
        self.num_cls = settings.MAX_CAPTCHA*settings.ALL_CHAR_SET_LEN
        self.base = torchvision.models.resnet34(pretrained=False)
        self.base.fc = nn.Linear(self.base.fc.in_features, self.num_cls)
    def forward(self, x):
        out = self.base(x)
        return out

class RES50(nn.Module):
    def __init__(self):
        super(RES50, self).__init__()
        self.num_cls = settings.MAX_CAPTCHA*settings.ALL_CHAR_SET_LEN
        self.base = torchvision.models.resnet50(pretrained=False)
        self.base.fc = nn.Linear(self.base.fc.in_features, self.num_cls)
    def forward(self, x):
        out = self.base(x)
        return out

class RES101(nn.Module):
    def __init__(self):
        super(RES101, self).__init__()
        self.num_cls = settings.MAX_CAPTCHA*settings.ALL_CHAR_SET_LEN
        self.base = torchvision.models.resnet101(pretrained=False)
        self.base.fc = nn.Linear(self.base.fc.in_features, self.num_cls)
    def forward(self, x):
        out = self.base(x)
        return out

class RES152(nn.Module):
    def __init__(self):
        super(RES152, self).__init__()
        self.num_cls = settings.MAX_CAPTCHA*settings.ALL_CHAR_SET_LEN
        self.base = torchvision.models.resnet152(pretrained=False)
        self.base.fc = nn.Linear(self.base.fc.in_features, self.num_cls)
    def forward(self, x):
        out = self.base(x)
        return out

class ALEXNET(nn.Module):
    def __init__(self):
        super(ALEXNET, self).__init__()
        self.num_cls = settings.MAX_CAPTCHA*settings.ALL_CHAR_SET_LEN
        self.base = torchvision.models.alexnet(pretrained=False)
        self.base.classifier[-1] = nn.Linear(4096, self.num_cls)
    def forward(self, x):
        out = self.base(x)
        return out

class VGG11(nn.Module):
    def __init__(self):
        super(VGG11, self).__init__()
        self.num_cls = settings.MAX_CAPTCHA*settings.ALL_CHAR_SET_LEN
        self.base = torchvision.models.vgg11(pretrained=False)
        self.base.classifier[-1] = nn.Linear(4096, self.num_cls)
    def forward(self, x):
        out = self.base(x)
        return out

class VGG13(nn.Module):
    def __init__(self):
        super(VGG13, self).__init__()
        self.num_cls = settings.MAX_CAPTCHA*settings.ALL_CHAR_SET_LEN
        self.base = torchvision.models.vgg13(pretrained=False)
        self.base.classifier[-1] = nn.Linear(4096, self.num_cls)
    def forward(self, x):
        out = self.base(x)
        return out
        
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.num_cls = settings.MAX_CAPTCHA*settings.ALL_CHAR_SET_LEN
        self.base = torchvision.models.vgg16(pretrained=False)
        self.base.classifier[-1] = nn.Linear(4096, self.num_cls)
    def forward(self, x):
        out = self.base(x)
        return out

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.num_cls = settings.MAX_CAPTCHA*settings.ALL_CHAR_SET_LEN
        self.base = torchvision.models.vgg19(pretrained=False)
        self.base.classifier[-1] = nn.Linear(4096, self.num_cls)
    def forward(self, x):
        out = self.base(x)
        return out

class SQUEEZENET(nn.Module):
    def __init__(self):
        super(SQUEEZENET, self).__init__()
        self.num_cls = settings.MAX_CAPTCHA*settings.ALL_CHAR_SET_LEN
        self.base = torchvision.models.squeezenet1_0(pretrained=False)
        self.base.classifier[-3] = nn.Linear(512, self.num_cls)
    def forward(self, x):
        out = self.base(x)
        return out

class DENSE161(nn.Module):
    def __init__(self):
        super(DENSE161, self).__init__()
        self.num_cls = settings.MAX_CAPTCHA*settings.ALL_CHAR_SET_LEN
        self.base = torchvision.models.densenet161(pretrained=False)
        self.base.classifier = nn.Linear(self.base.classifier.in_features, self.num_cls)
    def forward(self, x):
        out = self.base(x)
        return out

class MOBILENET(nn.Module):
    def __init__(self):
        super(MOBILENET, self).__init__()
        self.num_cls = settings.MAX_CAPTCHA*settings.ALL_CHAR_SET_LEN
        self.base = torchvision.models.mobilenet_v2(pretrained=False)
        self.base.classifier = nn.Linear(self.base.last_channel, self.num_cls)
    def forward(self, x):
        out = self.base(x)
        return out
