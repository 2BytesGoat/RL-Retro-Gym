import torch
import torch.nn as nn
import torch.nn.functional as F
from piq import SSIMLoss # image quality losses

def get_encoder(enc_type, input_shape, latent_dim):
    if enc_type == 'pca':
        encoder = PCA(input_shape, latent_dim)
    elif enc_type == 'mlp':
        encoder = MLP(input_shape, latent_dim)
    return encoder

class EncoderTemplate(nn.Module):
    def __init__(self, input_shape, latent_dim=100):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.img_size = self._get_size_from_shape(input_shape)

        self.encoder = None
        self.decoder = None
        self.optimizer = None
        self.loss = nn.MSELoss()

    def encode(self, frames):
        return self.encoder(frames)

    def decode(self, encoding):
        return self.decoder()

    def _get_size_from_shape(self, input_shape):
        rez = 1
        for dim in input_shape:
            rez *= dim
        return rez

    def forward(self, frames):
        encoding = self.encoder(frames)
        decoding = self.decoder(encoding).view(-1, *self.input_shape)
        return encoding, decoding

    def calculate_loss(self, x_t, decoding):
        return self.loss(decoding, x_t)

    def train(self, x_t):
        encoding, decoding = self.forward(x_t)
        loss = self.calculate_loss(x_t, decoding)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self, x_t):
        with torch.no_grad():
            encoding, decoding = self.forward(x_t)
        loss = self.calculate_loss(x_t, decoding)
        return loss.item(), encoding, decoding


class PCA(EncoderTemplate):
    def __init__(self, input_shape, latent_dim=100):
        super().__init__(input_shape, latent_dim)
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.img_size, self.latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.img_size),
        )
        self.optimizer = torch.optim.Adam(self.parameters())

class MLP(EncoderTemplate):
    def __init__(self, input_shape, latent_dim=100):
        super().__init__(input_shape, latent_dim)
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.img_size, 512),
            nn.ReLU(),
            nn.Linear(512, self.latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.img_size),
            nn.Sigmoid()
        )

        self.optimizer = torch.optim.Adam(self.parameters())
        
    