import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from typing import Tuple
import numpy as np

class Unet(nn.Module):
    def __init__(self, in_channels=12*8, n_classes=6*8, depth=5, wf=6, padding=True, batch_norm=False):
        """
        Args:
                    in_channels (int): number of input channels
                    n_classes (int): number of output channels
                    depth (int): depth of the network
                    wf (int): number of filters in the first layer is 2**wf
                    padding (bool): if True, apply padding such that the input shape
                                    is the same as the output.
                                    This may introduce artifacts
                    batch_norm (bool): Use BatchNorm after layers with an
                                       activation function
                    up_mode (str): one of 'upconv' or 'upsample'.
                                   'upconv' will use transposed convolutions for
                                   learned upsampling.
                                   'upsample' will use bilinear upsampling.
        """
        super(Unet, self).__init__()
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UnetConvBlock(prev_channels, 2**(wf+i), self.padding, batch_norm))
            prev_channels = 2 ** (wf+i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth-1)):
            self.up_path.append(UnetUpBlock(prev_channels, 2 ** (wf+i), self.padding, batch_norm))
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=(1,1))

    # def extract_features(self, x, *args, **kwargs):
    #     for i, down in enumerate(self.down_path):
    #         x = down(x)
    #         if i != len(self.down_path) - 1:
    #             x = F.max_pool2d(x,2)
    #     return x

    def forward(self, x, *args, **kwargs):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x,2)
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])
        return self.last(x)

class UnetConvBlock(nn.Module):
    def __init__(self, input_size, output_size, padding=True, batch_norm=True):
        super(UnetConvBlock, self).__init__()
        block = []
        block.append(nn.Conv2d(input_size, output_size, kernel_size=(3, 3), padding=int(padding)))
        if batch_norm:
            block.append(nn.BatchNorm2d(output_size))
        block.append(nn.ReLU())
        block.append(nn.Conv2d(output_size, output_size, kernel_size=(3, 3), padding=int(padding)))
        if batch_norm:
            block.append(nn.BatchNorm2d(output_size))
        block.append(nn.ReLU())
        self.block = nn.Sequential(*block)

    def forward(self, x):
        output = self.block(x)
        return output

class UnetUpBlock(nn.Module):
    def __init__(self, input_size, output_size, padding=True, batch_norm=True):
        super(UnetUpBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(input_size, output_size, kernel_size=(2, 2), stride=(2, 2))
        self.conv_block = UnetConvBlock(input_size, output_size, padding, batch_norm)

    def center_crop(self, encoder_layer, target_size):
        _, _, h, w = encoder_layer.shape
        start_height = (h - target_size[0])//2
        start_width = (w - target_size[1])//2
        end_height = start_height + target_size[0]
        end_width = start_width + target_size[1]
        return encoder_layer[:, :, start_height:end_height, start_width:end_width]

    def forward(self, x, bridge):
        x = self.upconv(x)
        crop = self.center_crop(bridge, x.shape[2:])
        x = torch.cat([crop, x], 1)
        x = self.conv_block(x)
        return x

class UNetTransfomer():
        """Transformer `T4CDataset` <-> `UNet`.

        zeropad2d only works with
        """
        @staticmethod
        def Unet_pre_transform(data: torch.tensor,
                               zeropad2d:Optional[Tuple[int,int,int,int]] = None,
                               stack_channels_on_time: bool = True,
                               batch_dim: bool = False,
                               **kwargs
                               ):
            if not batch_dim:
                data = torch.unsqueeze(data,0)
            if stack_channels_on_time:
                data = UNetTransfomer.transform_stack_channels_on_time(data, batch_dim=True)
            if zeropad2d is not None:
                zeropad2d = torch.nn.ZeroPad2d(zeropad2d)
                data = zeropad2d(data)
            return data

        @staticmethod
        def transform_stack_channels_on_time(data: torch.Tensor, batch_dim: bool = False):
            if not batch_dim:
                # `(12, 495, 436, 8) -> (1, 12, 495, 436, 8)`
                data = torch.unsqueeze(data,0)
            num_time_steps = data.shape[1]
            num_channels = data.shape[4]
            # (k, 12, 495, 436, 8) -> (k, 12, 8, 495, 436)
            data = torch.moveaxis(data, 4, 2)
            # (k, 12, 8, 495, 436) -> (k, 12 * 8, 495, 436)
            data = torch.reshape(data,(data.shape[0], num_time_steps*num_channels, 495, 436))
            if not batch_dim:
                # `(1, 12, 495, 436, 8) -> (12, 495, 436, 8)`
                data = torch.squeeze(data, 0)
            return data

        @staticmethod
        def Unet_post_transform(data: torch.Tensor, crop: Optional[Tuple[int, int, int, int]] = None,
                                stack_channels_on_time: bool = False, batch_dim: bool = False, **kwargs):
            """
            Bring data from UNet back to `T4CDataset` format:
            - separats common dimension for time and channels
            - cropping
            """
            if not batch_dim:
                data = torch.unsqueeze(data, 0)

            if crop is not None:
                _,_,height,width = data.shape()
                left, right, top, bottom = crop
                right = width - right
                bottom = height - bottom
                data = data[:, :, top:bottom, left: right]
            if stack_channels_on_time:
                data = UNetTransfomer.transform_unstack_channels_on_time(data, batch_dim=True)
            if not batch_dim:
                data = torch.squeeze(data, 0)
            return data

        @staticmethod
        def transform_unstack_channels_on_time(data: torch.Tensor, num_channels=8, batch_dim: bool = False):
            """
            `(k, 12 * 8, 495, 436) -> (k, 12, 495, 436, 8)`
            """
            if not batch_dim:
                data = torch.unsqueeze(data, 0)
            num_time_steps = int(data.shape[1]/num_channels)
            # (k, 12 * 8, 495, 436) -> (k, 12, 8, 495, 436)
            data = torch.reshape(data, (data.shape[0], num_time_steps, num_channels, 495, 436))
            # (k, 12, 8, 495, 436) -> (k, 12, 495, 436, 8)
            data = torch.moveaxis(data, 2, 4)
            if not batch_dim:
                # `(1, 12, 495, 436, 8) -> (12, 495, 436, 8)`
                data = torch.squeeze(data, 0)
            return data






