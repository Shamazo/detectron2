import torch
import torch.nn as nn
import torch.nn.functional as F

import typing
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


class Encoder(nn.Module):
    """The Encoder module will take in 128x128x3 ('width'x'height'x'channel') patches from the 
    original image and compress it into a vector. 
    """

    def __init__(self):
        super().__init__()

        self.op_1: nn.Sequential = TheisConv(stride=2)
        self.op_2: nn.Sequential = TheisConv(stride=2, input=64, out=128)
        self.op_3: nn.Sequential = TheisResidual()
        self.op_4: nn.Sequential = TheisResidual()
        self.op_5: nn.Sequential = TheisResidual()
        self.op_6: nn.Sequential = TheisConv(stride=2, input=128, out=96)

        self.quantize = TheisRounding.apply

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.op_1(x)  # downsample
        z = self.op_2(z)

        z = z + self.op_3(z)  # residual
        z = z + self.op_4(z)
        z = z + self.op_5(z)

        z = self.op_6(z)  # upsample

        # z = self.quantize(z)  # quantization trick
        return z


def Subpixel(input=96, out=512, scale=2) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(input, out, 3, stride=1, padding=1),
        nn.modules.PixelShuffle(scale),
    )


class Decoder(nn.Module):
    """The Decoder module will take in a compressed patch 
    and deconvolve it into an image.
    """

    def __init__(self):
        super().__init__()

        self.op_1 = Subpixel()

        use_padding = True
        self.op_2 = TheisResidual()
        self.op_3 = TheisResidual()
        self.op_4 = TheisResidual()

        self.op_5 = Subpixel(input=128, out=256)
        self.op_5_activation = nn.LeakyReLU()
        self.op_6 = Subpixel(input=64, out=12)
        self.clip = ClipGradient.apply

    def forward(self, x):
        z = self.op_1(x)  # downsample

        # print(z.shape)

        z = z + self.op_2(z)  # residual

        # print(z.shape)

        z = z + self.op_3(z)

        # print(z.shape)
        z = z + self.op_4(z)

        z = self.op_5_activation(self.op_5(z))  # upsample

        z = self.op_6(z)  # upsample
        z = z * 255  # returning to [0, 255]
        z = self.clip(z)  # round to nearest int and cast to byte
        return z


from torch.distributions import Normal, Uniform

class GSM(nn.Module):
    """The GSM provides an estimate of the entropy
    of the quantized distribution.
    """

    def __init__(self, s=6, in_channels=96, patch=128, bsz=None):
        super(GSM, self).__init__()
        self.s = s
        self.eps = 1e-7

        variance = torch.randn([1, in_channels, 1, 1, self.s])
        pi = torch.randn(1, in_channels, 1, 1, self.s)

        self.variance = torch.nn.Parameter(variance)
        self.pi = torch.nn.Parameter(pi)

        self.uni: Uniform = Uniform(-0.5, 0.5)
        self.eps = 0.0001

    def forward(self, x: torch.Tensor):
        shape: Tuple[int, int, int, int] = x.shape  # type: ignore
        batch, k, i, j = shape

        pi = F.softmax(self.pi, dim=-1).repeat(batch, 1, i, j, 1)
        u = (
            self.uni.rsample(sample_shape=(batch, k, i, j, 1))
            .repeat(1, 1, 1, 1, self.s)
            .to(list(self.parameters())[0].device)
        )
        x = x.unsqueeze(4).repeat(1, 1, 1, 1, self.s)
        variance = self.variance.repeat(batch, 1, i, j, 1)

        exp_terms = (-0.5 * ((x + u) ** 2) / variance.exp()).exp()
        leading_terms = 1 / (2 * 3.14159 * (self.eps + variance).exp()).sqrt()
        normal_pdfs = leading_terms * exp_terms
        total_pdfs = (pi * normal_pdfs).sum(axis=-1) + self.eps
        return total_pdfs.log2().mean(dim=(1, 2, 3))

    # def forward(self, x: torch.Tensor):
    #     shape: Tuple[int, int, int, int] = x.shape # type: ignore
    #     batch, k, i, j = shape

    #     pi = F.softmax(self.pi, dim=1)

    #     u = self.uni.rsample(sample_shape=(batch, k, i, j,1)).repeat(1,1,1,1,self.s).to(list(self.parameters())[0].device)
    #     x = x.unsqueeze(4).repeat(1,1,1,1,self.s)

    #     dist = Normal(torch.zeros((batch, k,i,j,self.s)).to(list(self.parameters())[0].device), self.variance.repeat(batch, 1,i,j,1)**2)
    #     probs = dist.log_prob(x+u).exp()

    #     temp_1 = torch.einsum('ks,bkxys->bkxy', pi, probs) #TODO: einsum has performance implications to consider here.
    #     temp_2 = torch.einsum('bkxy->b', temp_1.log2())

    #     return temp_2

    def forward_no_einsum(self, x: torch.Tensor):
        shape: Tuple[int, int, int, int] = x.shape  # type: ignore
        batch, k, i, j = shape

        self.pi = self.pi.abs() / self.pi.sum(1, keepdim=True).clamp(min=self.eps)

        u = self.uni.rsample(sample_shape=(batch, k, i, j, 1)).repeat(
            1, 1, 1, 1, self.s
        )
        x = x.unsqueeze(4).repeat(1, 1, 1, 1, self.s)

        dist = Normal(
            torch.zeros((batch, k, i, j, self.s)),
            self.variance.repeat(batch, 1, i, j, 1) ** 2,
        )
        probs = dist.log_prob(x + u).exp()

        temp = self.pi.view(1, k, 1, 1, self.s) * probs
        return temp.sum(-1).log2().sum((1, 2, 3))
