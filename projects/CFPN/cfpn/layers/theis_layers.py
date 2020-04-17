import torch
import torch.nn as nn

from typing import Tuple


def TheisConv(input=3, out=64, kernel=5, stride=1, activation=True) -> nn.Sequential:
    if activation:
        return nn.Sequential(
            nn.ReflectionPad2d((1, stride, 1, stride)),
            nn.Conv2d(
                in_channels=input,
                out_channels=out,
                kernel_size=(kernel, kernel),
                stride=(stride, stride),
            ),
            nn.LeakyReLU(),
        )
    else:
        return nn.Sequential(
            nn.ReflectionPad2d((1, stride, 1, stride)),
            nn.Conv2d(
                in_channels=input,
                out_channels=out,
                kernel_size=(kernel, kernel),
                stride=(stride, stride),
            ),
        )


def TheisResidual(first_activation=True, second_activation=False) -> nn.Sequential:
    return nn.Sequential(
        TheisConv(kernel=3, input=128, out=128, activation=first_activation),
        TheisConv(kernel=3, input=128, out=128, activation=second_activation),
    )


class ClipGradient(torch.autograd.Function):
    """
    Clips the output to [0, 255] and casts it to an integer
    """

    @staticmethod
    def forward(ctx, input):
        return torch.clamp(input, 0, 255).round()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class TheisRounding(torch.autograd.Function):
    """
    You compute r(x) = [x]. This a transformation from
    """

    @staticmethod
    def forward(ctx, input):
        with torch.no_grad():
            z = input.round()
        return z

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def Subpixel(input=96, out=512, scale=2) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(input, out, 3, stride=1, padding=1),
        nn.modules.PixelShuffle(scale),
    )
