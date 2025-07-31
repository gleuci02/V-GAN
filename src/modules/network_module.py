from pyod.utils.torch_utility import get_activation_by_name
import torch
from torch import nn
from ..models.Generator import upper_softmax, upper_lower_softmax
import numpy as np
from torch.nn.utils import spectral_norm

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

class ttEncoder(torch.nn.Module):
    def __init__(self, latent_size, img_size):

        super(ttEncoder, self).__init__()
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
    

class ttDecoder(torch.nn.Module):
    def __init__(self, latent_size, img_size):

        super(ttDecoder, self).__init__()
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


class Encoder(nn.Module):
    def __init__(self, latent_size, img_size, channel):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channel, img_size, 3, stride=2, padding=1),  # 32x32 -> 16x16   nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(img_size),
            nn.Conv2d(img_size, 2*img_size, 3, stride=2, padding=1),  # 16x16 -> 8x8    nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(2*img_size),
            nn.Conv2d(2*img_size, 4*img_size, 3, stride=2, padding=1),  # 8x8 -> 4x4   nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(4*img_size)
        )
        self.fc_mu = nn.Linear(int((img_size / 16)*img_size * img_size), 2*img_size)  # Latent vector

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten before FC
        z = self.fc_mu(x)  # Latent space output
        return z

class Decoder(nn.Module):
    def __init__(self, latent_size, img_size, channel):
        super(Decoder, self).__init__()
        self.img_size = img_size
        self.fc = nn.Linear(2*img_size, int((img_size/16)*img_size * img_size))  # Expand to feature map
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4*img_size, 2*img_size, 3, stride=2, padding=1, output_padding=1),  # 4x4 -> 8x8 nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(2*img_size),
            nn.ConvTranspose2d(2*img_size, img_size, 3, stride=2, padding=1, output_padding=1),  # 8x8 -> 16x16 nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
            nn.ReLU(),
            nn.BatchNorm2d(img_size),
            nn.ConvTranspose2d(img_size, channel, 3, stride=2, padding=1, output_padding=1),  # 16x16 -> 32x32 nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1)
            nn.Tanh()  # CIFAR-10 pixel values are between 0-1
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 4*self.img_size, int(self.img_size / 8), int(self.img_size / 8))  # Reshape to feature map
        x = self.decoder(x)
        return x

    
    
class Detector(nn.Module): #Using an ELM as the embedding worked for LUNAR and Deep IForest
    def __init__(self, latent_size, img_size, channel, encoder, decoder):
        super(Detector, self).__init__()
        self.encoder = encoder(latent_size, img_size, channel)
        self.decoder = decoder(latent_size, img_size, channel)

    def forward(self, input):
        enc_X = self.encoder(input)
        dec_X = self.decoder(enc_X)

        #enc_X = enc_X.view(input.size(0), -1)
        dec_X = dec_X.view(input.size(0), -1)
        return enc_X, dec_X


class Generator_big(nn.Module):
    def __init__(self, latent_size, img_size):
        super(Generator_big, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(latent_size, 2*latent_size),
            GaussianNoise(),
            nn.BatchNorm1d(2*latent_size, eps=0.01),
            nn.Tanh(),
            nn.Linear(2*latent_size, 4*latent_size),
            GaussianNoise(),
            nn.BatchNorm1d(4*latent_size, eps=0.01),
            nn.Tanh(),
            nn.Linear(4*latent_size, 8*latent_size),
            GaussianNoise(),
            nn.BatchNorm1d(8*latent_size, eps=0.01),
            nn.Tanh(),
            nn.Linear(8*latent_size, 16*latent_size),
            GaussianNoise(),
            nn.BatchNorm1d(16*latent_size, eps=0.01),
            nn.Tanh(),
            nn.Linear(16*latent_size, img_size),
            GaussianNoise(),
            nn.BatchNorm1d(img_size, eps=0.01),
            nn.Tanh(),
            nn.Sigmoid(),
            upper_softmax(),
            #nn.Softmax(),
        )

    def forward(self, z):
        return self.main(z)

class GaussianNoise(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = nn.Parameter(torch.tensor(std))

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x

class BatchDiscrimination(nn.Module):
    def __init__(self, in_features, B, C):
        """
        in_features: length of f(x) (A)
        B, C: dimensions of the learned tensor T (A × B × C)
        """
        super().__init__()
        # T will be of shape (A, B, C)
        self.T = nn.Parameter(torch.randn(in_features, B, C))

    def forward(self, f):
        # f: (N, A)  batch of feature vectors
        N, A = f.shape
        # 1) Project: (N, A) × (A, B, C) → (N, B, C)

        M = f @ self.T.view(A, -1)           # (N, 1, A) @ (A, B*C) → (N, 1, B*C)
        M = M.view(N, -1, self.T.size(2))                 # → (N, B, C)

        # 2) Compute pairwise L1 distances along B:
        #    We want a tensor of shape (N, C, N) containing distances M[i,:,c] vs M[j,:,c].
        M_i = M.unsqueeze(3)            # (N, B, C, 1)
        M_j = M.permute(1,2,0).unsqueeze(0)  # (1, B, C, N)
        abs_diff = torch.abs(M_i - M_j)  # (N, B, C, N)
        o = abs_diff.sum(dim=1)          # sum over B → (N, C, N)

        # 3) Similarity and sum over other examples:
        #    Exclude j=i if you like; here we include self for simplicity.
        d = torch.exp(-o).sum(dim=2)     # (N, C)

        return d

# Main V-GAN Head Network
class VGANHead(nn.Module):
    def __init__(self, latent_size=1024, img_size=1024):
        super().__init__()
        print("Hi I am VGANHead")

        self.main = nn.Sequential(
            nn.Linear(latent_size, 2*latent_size),
            GaussianNoise(std=0.1),
            nn.BatchNorm1d(2*latent_size, eps=0.01),
            nn.LeakyReLU(0.2),

            nn.Linear(2*latent_size, 4*latent_size),
            GaussianNoise(std=0.1),
            nn.BatchNorm1d(4*latent_size, eps=0.01),
            nn.LeakyReLU(0.2),

            nn.Linear(4*latent_size, 8*latent_size),
            GaussianNoise(std=0.1),
            nn.BatchNorm1d(8*latent_size, eps=0.01),
            nn.LeakyReLU(0.2), #0.2

            nn.Linear(8*latent_size, 16*latent_size),
            GaussianNoise(std=0.1),
            nn.BatchNorm1d(16*latent_size, eps=0.01),
            nn.LeakyReLU(0.2),
        )
        
        self.discr = BatchDiscrimination(16*latent_size, 50, 10) # 50, 10

        self.last_lin = nn.Linear(16*latent_size + 10, img_size) # + 10    9*1024   / 4   9
    
        self.act = nn.Sequential(
            #nn.Sigmoid(),
            upper_softmax(),
            #nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.main(x.flatten(1, -1))
        
        batch_feats = self.discr(x)  # → (N, C)
        combined = torch.cat([x, batch_feats], dim=1)
        res = self.last_lin(combined)

        return self.act(res)
