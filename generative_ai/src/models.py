import math

import torch
import torch.nn as nn


class VAE(nn.Module):

    def __init__(self, image_size, latent_size, hidden_dimension,
                 conditional=False, num_params=0):

        super().__init__()

        if conditional:
            assert num_params > 0

        assert type(image_size) == tuple
        assert type(hidden_dimension) == int
        assert type(latent_size) == int

        self.image_size_flatten = math.prod(image_size)
        self.latent_size = latent_size

        self.encoder = Encoder(
            [self.image_size_flatten, hidden_dimension], latent_size, conditional, num_params+1)
        self.decoder = Decoder(
            [hidden_dimension, self.image_size_flatten], latent_size, conditional, num_params+1)

    def forward(self, x, c=None):

        if x.dim() > 2:
            x = x.view(-1, self.image_size_flatten)

        means, log_var = self.encoder(x, c)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z, c)

        return recon_x, means, log_var, z

    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

    def inference(self, z, c=None):

        recon_x = self.decoder(z, c)

        return recon_x


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_params):

        super().__init__()

        self.conditional = conditional
        if self.conditional:
            self.cMLP = nn.Linear(num_params, 32)
            layer_sizes[0] += 32

        self.MLP = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c=None):

        if self.conditional:
            c = self.cMLP(c)
            x = torch.cat((x, c), dim=-1)

        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_params):

        super().__init__()

        self.conditional = conditional
        if self.conditional:
            self.cMLP = nn.Linear(num_params, 32)
            input_size = latent_size + 32
        else:
            input_size = latent_size

        self.MLP = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                self.MLP.add_module(name="activation", module=nn.Hardtanh(-5, 5))

    def forward(self, z, c):

        if self.conditional:
            c = self.cMLP(c)
            z = torch.cat((z, c), dim=-1)

        x = self.MLP(z)

        return x
