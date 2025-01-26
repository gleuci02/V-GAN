from pyod.utils.torch_utility import get_activation_by_name
import torch
from torch import nn
from ..models.Generator import upper_softmax
import numpy as np

"""Module to add addtional networks for VGAN

Networks are added as nn.netowrks and then utilized by the od_module to implement inside VGAN
 """


class MLPnet(torch.nn.Module):
    def __init__(self, n_features, n_hidden=[500, 100], n_output=20,
                 activation='ReLU', bias=False, batch_norm=False,
                 skip_connection=False):
        super(MLPnet, self).__init__()
        self.skip_connection = skip_connection
        self.n_output = n_output

        num_layers = len(n_hidden)

        if type(activation) == str:
            activation = [activation] * num_layers
            activation.append(None)

        assert len(activation) == len(
            n_hidden) + 1, 'activation and n_hidden are not matched'

        self.layers = []
        for i in range(num_layers + 1):
            in_channels, out_channels = \
                self.get_in_out_channels(i, num_layers, n_features,
                                         n_hidden, n_output, skip_connection)
            self.layers += [
                LinearBlock(in_channels, out_channels,
                            bias=bias, batch_norm=batch_norm,
                            activation=activation[i],
                            skip_connection=skip_connection if i != num_layers else False)
            ]
        self.network = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.network(x)
        return x

    @staticmethod
    def get_in_out_channels(i, num_layers, n_features, n_hidden, n_output,
                            skip_connection):
        if skip_connection is False:
            in_channels = n_features if i == 0 else n_hidden[i - 1]
            out_channels = n_output if i == num_layers else n_hidden[i]
        else:
            in_channels = n_features if i == 0 else np.sum(
                n_hidden[:i]) + n_features
            out_channels = n_output if i == num_layers else n_hidden[i]
        return in_channels, out_channels


class LinearBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels,
                 activation='Tanh', bias=False, batch_norm=False,
                 skip_connection=False):
        super(LinearBlock, self).__init__()

        self.skip_connection = skip_connection

        self.linear = torch.nn.Linear(in_channels, out_channels, bias=bias)

        if activation is not None:
            # self.act_layer = _instantiate_class("torch.nn.modules.activation", activation)
            self.act_layer = get_activation_by_name(activation)
        else:
            self.act_layer = torch.nn.Identity()

        self.batch_norm = batch_norm
        if batch_norm is True:
            dim = out_channels
            self.bn_layer = torch.nn.BatchNorm1d(dim, affine=bias)

    def forward(self, x):
        x1 = self.linear(x)
        x1 = self.act_layer(x1)

        if self.batch_norm is True:
            x1 = self.bn_layer(x1)

        if self.skip_connection:
            x1 = torch.cat([x, x1], axis=1)

        return x1

class Encoder(torch.nn.Module):
    def __init__(self, latent_size, img_size):

        super(Encoder, self).__init__()
        assert img_size % 16 == 0, "isize has to be a multiple of 16"
        
        # input is nc x isize x isize
        main = nn.Sequential()

        main.add_module('Initial layer', nn.Linear(img_size, 64))
        csize, cndf = img_size / 2, 64

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid_{0}-{1}_linear'.format(in_feat, out_feat), nn.Linear(in_feat, out_feat))
            cndf = cndf * 2
            csize = csize / 2

        main.add_module('final_{0}-{1}_conv'.format(in_feat, out_feat), nn.Linear(cndf, latent_size))
        self.main = main

    def forward(self, input):
        output = self.main(input)
        return output
    

class Decoder(torch.nn.Module):
    def __init__(self, latent_size, img_size):

        super(Decoder, self).__init__()
        assert img_size % 16 == 0, "img_size has to be a multiple of 16"

        cngf, tisize = 64 // 2, 4
        while tisize != img_size:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        main.add_module('initial_{0}_linear'.format(cngf), nn.Linear(latent_size, cngf))

        csize = 4
        while csize < img_size // 2:
            main.add_module('pyramid_{0}-{1}_linear'.format(cngf, cngf // 2), nn.Linear(cngf, cngf // 2))
            cngf = cngf // 2
            csize = csize * 2

        main.add_module('final_{0}-{1}_convt'.format(cngf, 1), nn.Linear(cngf, 1))

        self.main = main

    def forward(self, input):
        output = self.main(input)
        return output