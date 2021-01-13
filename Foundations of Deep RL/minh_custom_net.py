"""
    Dec 7th
    Custom Deep Network ResNet
    By: Minh Nguyen
    ------------------------------------------------------------------------------------------------
    SubModule 2D since the gamebreaker environment "RawSC2Env" defines an observation space that looks like this:
    {
        "proc_units": (512, 130)
    }
    512 is the maximum possible number of units
    130 is the number of features (each unit has a feature vector)
    because the input space has two dimensions (512 and 130), the ModularNetwork will initialize the net2d class specified in the configuration (via the --net2d flag)
    ------------------------------------------------------------------------------------------------
    Credit: Fully Pre-activate ResNet - https://arxiv.org/pdf/1603.05027.pdf
    Todo:
        Fix Wrong Projection
        Fix Structure

"""
import torch
from adept.network import SubModule2D
from torch import nn
from torch.nn import functional as F
from adept.modules import Identity
from collections import OrderedDict

"""
Confirm
    Input = [2,129,512]
    Output = [2, 512, 16]
    [2,64,64]
    f = number of input features
    h = number of filters to used for convolution = number of output features
"""
# Residual block that has structure of batch norm, relu , conv1, batch norm, relu, conv2
class ResBlock(nn.Module):
    def __init__(
        self, input_features, output_features, stride=1
    ):  # number of class s be specified in the deepresnet class
        super(ResBlock, self).__init__()
        expansion = 1

        self.bn1 = nn.BatchNorm1d(input_features)
        self.conv1 = nn.Conv1d(
            input_features,
            output_features,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(output_features)
        self.conv2 = nn.Conv1d(
            output_features,
            output_features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        # if stride != 1 or the input shape != the output shape
        if stride != 1 or (input_features != expansion * output_features):
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    input_features,
                    expansion * output_features,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


# Deep ResNet class use ResBlock class to create multiple layers################
# Implement ResNet34
class DeepResNet(SubModule2D):
    args = {
        "normalize": "bn",
        "nb_hidden": 512,
    }
    expansion = 1

    def __init__(self, input_shape, id, normalize, nb_hidden):  # default
        # super().__init__(input_shape, id) # default
        super(DeepResNet, self).__init__(input_shape, id)
        self._nb_hidden = nb_hidden  # default

        self.input_feature, s = input_shape
        output_feature = nb_hidden

        # how does the input and output of layer work when the output of the layer is 256 but the next input is 512
        # 512
        self.layer1 = self.make_layer(
            ResBlock, 512, 3, stride=2
        )  # 512 since we want to reserve the output 512

        # 256
        self.layer2 = self.make_layer(ResBlock, 512, 3, stride=2)

        # 128
        self.layer3 = self.make_layer(ResBlock, 512, 3, stride=2)

        # 64
        self.layer4 = self.make_layer(ResBlock, 512, 3, stride=2)

        # 32
        self.layer5 = self.make_layer(ResBlock, 512, 3, stride=2)

        # 16

    # Function make conv layer with ResBlock
    def make_layer(self, block, output, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(self.input_feature, output, stride)
            )  # same block format as ResBlock
            self.input_feature = output
        return nn.Sequential(*layers)

    # default
    @classmethod
    def from_args(cls, args, input_shape, id):
        return cls(input_shape, id, args.normalize, args.nb_hidden)

    # default
    @property
    def _output_shape(self):
        f, s = self.input_shape
        output_seq_len = s
        for i in range(5):
            output_seq_len //= 2
        return self._nb_hidden, output_seq_len  # number of hidden neuron

    def _forward(self, xs, internals, **kwargs):
        print(f"debugging input: The shape of the input is: {xs.shape}")
        out = self.layer1(xs)
        print(f"debugging 1: The shape of the output is: {out.shape}")
        out = self.layer2(out)
        print(f"debugging 2: The shape of the output is: {out.shape}")
        out = self.layer3(out)
        print(f"debugging 3: The shape of the output is: {out.shape}")
        out = self.layer4(out)
        print(f"debugging 4: The shape of the output is: {out.shape}")
        out = self.layer5(out)
        print(f"debugging 5: The shape of the output is: {out.shape}")
        return out, {}

    # default
    def _new_internals(self):
        return {}


#####################################################################################################

# class DeepResNet(SubModule2D):
#     args = {
#         "normalize" : "bn",
#         "nb_hidden" : 512,
#     }
#     def __init__ (self, input_shape, id, normalize, nb_hidden):
#         super().__init__(input_shape, id)
#         self._nb_hidden = nb_hidden
#
#         f, s = input_shape # in_plane and planes?
#         h = nb_hidden
#         is_bias = normalize
#
#         # Sequence Length = S = 512
#         self.res_block1 = nn.Conv1d(f, h, kernel_size = 3, stride = 2, padding = 1, bias = is_bias)
#
#         # Sequence Length = S = 256
#         self.res_block2 = nn.Conv1d(h, h, kernel_size = 3, stride = 2, padding = 1, bias = is_bias)
#
#         # Sequence Length = S = 128
#         self.res_block3 = nn.Conv1d(h, h, kernel_size = 3, stride = 2, padding = 1, bias = is_bias)
#
#         # Sequence Length = S = 64
#         self.res_block4 = nn.Conv1d(h, h, kernel_size = 3, stride = 2, padding = 1, bias = is_bias)
#
#         # Sequence Length = S = 32
#         self.res_block5 = nn.Conv1d(h, h, kernel_size = 3, stride = 2, padding = 1, bias = is_bias)
#
#         # Sequence Length (output) = S = 16
#
#         self.linear_proj1 = nn.Linear(512, 256)
#         self.linear_proj2 = nn.Linear(256, 128)
#         self.linear_proj3 = nn.Linear(128, 64)
#         self.linear_proj4 = nn.Linear(64, 32)
#         self.linear_proj5 = nn.Linear(32, 16)
#
#         self.norm1 = nn.BatchNorm1d(h)
#         self.norm2 = nn.BatchNorm1d(h)
#         self.norm3 = nn.BatchNorm1d(h)
#         self.norm4 = nn.BatchNorm1d(h)
#         self.norm5 = nn.BatchNorm1d(h)
#
#     @classmethod
#     def from_args(cls, args, input_shape, id):
#         return cls(input_shape, id, args.normalize, args.nb_hidden)
#
#     @property # Still correct?
#     def _output_shape(self):
#         f, s = self.input_shape
#
#         output_seq_len = s
#         for i in range(5):
#             output_seq_len //= 2
#
#         return self._nb_hidden, output_seq_len
#
#     def _forward(self, xs, internals, **kwargs):
#         xs1 = self.linear_proj1(xs) # linear projection down to 256
#         xs = torch.add(self.res_block1(F.relu(self.norm1(xs))), xs1) # output dim = 256x256
#
#         xs2 = self.linear_proj2(xs) # linear projection down to 128
#         xs = torch.add(self.res_block2(F.relu(self.norm2(xs))), xs2) # output dim = 128x128
#
#         xs3 = self.linear_proj3(xs) # linear projection down to 64
#         xs = torch.add(self.res_block3(F.relu(self.norm3(xs))), xs3) # output dim = 64x64
#
#         xs4 = self.linear_proj4(xs) # linear projection down to 32
#         xs = torch.add(self.res_block4(F.relu(self.norm4(xs))), xs4) # output dim = 32x32
#
#         xs5= self.linear_proj5(xs) # linear projection down to 16
#         xs = torch.add(self.res_block5(F.relu(self.norm5(xs))), xs5) # output dim = 16x16
#
#         return xs, {}
#
#     def _new_internals(self):
#         return {}
