import torch.optim as optim

from .Unet3D import UNet3D
from .Vnet import VNet, VNetLight
from .models.DMFNet_16x import DMFNet
from .unet3d.unet3d import ResidualUNet3D

def create_model(args):
    model_name = args.model
    optimizer_name = args.opt
    lr = args.lr
    in_channels = args.inChannels
    num_classes = args.classes
    weight_decay = 0.0000000001
    print("Building Model . . . . . . . ." + model_name)

    if model_name == 'VNET2':
        model = VNetLight(in_channels=in_channels, elu=False, classes=num_classes)
    elif model_name == 'VNET':
        model = VNet(in_channels=in_channels, elu=False, classes=num_classes)
    elif model_name == 'UNET3D':
        model = UNet3D(in_channels=in_channels, n_classes=num_classes, base_n_filter=8)
    elif model_name == "DMFnet":
        model = DMFNet(num_classes=num_classes)
    elif model_name == "RESIDUALUNET":
        model = ResidualUNet3D(in_channels=in_channels, out_channels=num_classes, f_maps=16)

    print(model_name, 'Number of params: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    if optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5, weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

    return model, optimizer
