# Copyright (c) Facebook, Inc. and its affiliates.

from torch import nn
import torch


def initialize_module_params(module: nn.Module):
    for name, param in module.named_parameters():
        if "bias" in name:
            nn.init.constant_(param, 0)
        elif "weight" in name:
            nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

def save_image_tensor(input_tensor: torch.Tensor, filename):
    from torchvision import utils as vutils

    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    input_tensor = input_tensor.clone().detach()
    input_tensor = input_tensor.to(torch.device('cpu'))
    vutils.save_image(input_tensor, filename)