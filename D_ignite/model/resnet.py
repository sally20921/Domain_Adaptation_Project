import torchvision

def ResNet(name, pretrained=False):
    resnets = {
            "resnet18": torchvision.models.resnet18(pretrained=pretrained),
            "resnet50": torchvision.models.resnet50(pretrained=pretrained),
    }

    return resnets[name]

