import torch
import torch.nn as nn
import torch.nn.functional as F

class PCA(nn.Module):
    def __init__(self, img_shape, latent_dim=100):
        super().__init__()
        self.image_shape = img_shape
        self.latent_dim = latent_dim
        self.img_size = self._get_size_from_shape(img_shape)

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.img_size, self.latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.img_size),
        )
    
    def forward(self, frames):
        encoding = self.encoder(frames)
        decoding = self.decoder(encoding).view(-1, *self.image_shape)
        return encoding, decoding

    def _get_size_from_shape(self, img_shape):
        rez = 1
        for dim in img_shape:
            rez *= dim
        return rez

class MLP(nn.Module):
    def __init__(self, img_shape, latent_dim=100):
        super().__init__()
        self.image_shape = img_shape
        self.latent_dim = latent_dim
        self.img_size = self._get_size_from_shape(img_shape)

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
    
    def forward(self, frames):
        encoding = self.encoder(frames)
        decoding = self.decoder(encoding).view(-1, *self.image_shape)
        return encoding, decoding

    def _get_size_from_shape(self, img_shape):
        rez = 1
        for dim in img_shape:
            rez *= dim
        return rez