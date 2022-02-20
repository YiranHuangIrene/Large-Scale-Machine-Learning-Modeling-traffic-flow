import math
import torch
from t4clab.utils.utils import extract_h5_data

def create_mask(static_data, num_slots=0):
    a = torch.ones_like(static_data[0])
    b = torch.zeros_like(static_data[0])
    static_mask_once = torch.where(static_data[0] > 0, a, b)
    assert static_mask_once.shape == (495, 436)
    if num_slots > 0:
        static_mask = torch.broadcast_to(static_mask_once, (num_slots, 495, 436))
        assert (static_mask[0] == static_mask_once).all()
        static_mask = torch.repeat_interleave(static_mask, 8).reshape(num_slots, 495, 436, 8)
        assert static_mask.shape == (num_slots, 495, 436, 8), f"{static_mask.shape}"
        assert (static_mask[0, :, :, 0] == static_mask_once).all()
    else:
        static_mask = torch.repeat_interleave(static_mask_once, 8).reshape(495, 436, 8)
        assert static_mask.shape == (495, 436, 8), f"{static_mask.shape}"
        assert (static_mask[:, :, 0] == static_mask_once).all()
    return static_mask

def get_city_static_map(city):
    static_data = extract_h5_data('/nfs/shared/traffic4cast' + '/' + city + '/' + city + '_static.h5')
    mask = create_mask(static_data)
    mask_torch_reshaped = torch.moveaxis(mask, 2, 0)
    mask_torch_unsqueezed = torch.unsqueeze(mask_torch_reshaped, 0)
    zeropad2d = (6, 6, 1, 0)
    padding = torch.nn.ZeroPad2d(zeropad2d)
    mask_torch_unsqueezed = padding(mask_torch_unsqueezed)
    summed_mask = torch.sum(mask_torch_unsqueezed[0], dim=0)
    mask_2d = torch.where(summed_mask > 0, 1, 0)
    return mask_2d
"""Transformer `T4CDataset` <-> `UNet`.
zeropad2d only works with
"""
def Unet_pre_transform(data: torch.tensor,
                       zeropad2d=(6,6,1,0),
                       stack_channels_on_time: bool = True,
                       batch_dim: bool = False,
                       **kwargs
                       ):
    if not batch_dim:
        data = torch.unsqueeze(data, 0)
    if stack_channels_on_time:
        data = transform_stack_channels_on_time(data, batch_dim=True)
    if zeropad2d is not None:
        zeropad2d = torch.nn.ZeroPad2d(zeropad2d)
        data = zeropad2d(data)
    if not batch_dim:
        data = torch.squeeze(data, 0)
    return data


def transform_stack_channels_on_time(data: torch.Tensor, batch_dim: bool = False):
    if not batch_dim:
        # `(12, 495, 436, 8) -> (1, 12, 495, 436, 8)`
        data = torch.unsqueeze(data, 0)

    num_time_steps = data.shape[1]
    num_channels = data.shape[4]

    # (k, 12, 495, 436, 8) -> (k, 12, 8, 495, 436)
    data = torch.moveaxis(data, 4, 2)

    # (k, 12, 8, 495, 436) -> (k, 12 * 8, 495, 436)
    data = torch.reshape(data, (data.shape[0], num_time_steps * num_channels, 495, 436))

    if not batch_dim:
        # `(1, 12, 495, 436, 8) -> (12, 495, 436, 8)`
        data = torch.squeeze(data, 0)
    return data


def Unet_post_transform(data: torch.Tensor, crop=None,
                        stack_channels_on_time: bool = False, batch_dim: bool = False, **kwargs):
    """
    Bring data from UNet back to `T4CDataset` format:
    - separats common dimension for time and channels
    - cropping
    """
    if not batch_dim:
        data = torch.unsqueeze(data, 0)

    if crop is not None:
        _, _, height, width = data.shape()
        left, right, top, bottom = crop
        right = width - right
        bottom = height - bottom
        data = data[:, :, top:bottom, left: right]
    if stack_channels_on_time:
        data = transform_unstack_channels_on_time(data, batch_dim=True)
    if not batch_dim:
        data = torch.squeeze(data, 0)
    return data


def transform_unstack_channels_on_time(data: torch.Tensor, num_channels=8):
    """
    `(k, 6 * 8, 496, 448) -> (496, 448, 8，6)`
    """
    #(k, 6 * 8, 496, 448) -> ( 6 * 8, 496, 448)
    data = torch.squeeze(data, 0)
    num_time_steps = int(data.shape[1] / num_channels)
    # ( 6 * 8, 496, 448) -> （ 6, 8, 495, 436)
    data = torch.reshape(data, (data.shape[0], num_time_steps, num_channels, 495, 436))
    # (6, 8, 496, 448) -> (6, 496, 448, 8)
    data = torch.moveaxis(data, 1, 4)
    # (6， 496, 448, 8) -> (496, 448, 8, 6)
    data = torch.moveaxis(data, 1, 4)
    return data


def build_sample(dynamicdata, padding =(6, 6, 1, 0) , start_time = None, day_of_week = None, input_size=12, include_timestamps=False):
    input_features = dynamicdata[:input_size]
    label = dynamicdata[input_size:][[0, 1, 2, 5, 8, 11]]

    # Per feature standardization
    """
    mean = all_node_features.mean(0)
    std = all_node_features.std(0)
    all_node_features = (all_node_features - mean) / std
    """

    # # Scaling to [0, 1]
    # input_features = input_features/255

    if include_timestamps:
        input_features = add_timestamp_to_features(input_features, start_time)
        input_features = add_day_of_week_to_features(input_features, day_of_week)
    # Transform data from `T4CDataset` be used by UNet
    # input.shape = (96, 496, 448)
    input_features = Unet_pre_transform(input_features, zeropad2d=padding, stack_channels_on_time=True, batch_dim=False)
    label = Unet_pre_transform(label, zeropad2d=(6, 6, 1, 0), stack_channels_on_time=True, batch_dim=False)

    sample = {"data": input_features, "label": label}

    return sample


def add_timestamp_to_features(input_features, timestamp):
    timestamp_radiant = 2 * math.pi * timestamp / 288.0
    timestamp_vector_sin = torch.ones((215820, 1)) * math.sin(timestamp_radiant)
    timestamp_vector_cos = torch.ones((215820, 1)) * math.cos(timestamp_radiant)
    input_features = torch.cat((input_features, timestamp_vector_sin, timestamp_vector_cos), 1)

    return input_features


def add_day_of_week_to_features(input_features, day_of_week):
    day_of_week_radiant = 2 * math.pi * day_of_week / 7.0
    day_of_week_vector_sin = torch.ones((215820, 1)) * math.sin(day_of_week_radiant)
    day_of_week_vector_cos = torch.ones((215820, 1)) * math.cos(day_of_week_radiant)
    input_features = torch.cat((input_features, day_of_week_vector_sin, day_of_week_vector_cos), 1)

    return input_features
