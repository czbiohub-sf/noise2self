from models.babyunet import BabyUnet
from models.dncnn import DnCNN
from models.singleconv import SingleConvolution
from models.unet import Unet


def get_model(name, in_channels, out_channels, **kwargs):
    if name == 'unet':
        return Unet(in_channels, out_channels)
    if name == 'baby-unet':
        return BabyUnet(in_channels, out_channels)
    if name == 'dncnn':
        return DnCNN(in_channels, out_channels)
    if name == 'convolution':
        return SingleConvolution(in_channels, out_channels, kwargs['width'])
