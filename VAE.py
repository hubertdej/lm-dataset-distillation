import torch
import torch.nn as nn

import numpy as np

from tqdm import tqdm
from torch.optim import Adam


cuda = True
DEVICE = torch.device("cuda" if cuda else "cpu")

x_dim  = 784
hidden_dim = 400
latent_dim = 200
lr = 1e-3
epochs = 30


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
        self.FC_var   = nn.Linear (hidden_dim, latent_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
        self.training = True
        
    def forward(self, x):
        h_       = self.LeakyReLU(self.FC_input(x))
        h_       = self.LeakyReLU(self.FC_input2(h_))
        mean     = self.FC_mean(h_)
        log_var  = self.FC_var(h_)
        
        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        h     = self.LeakyReLU(self.FC_hidden(x))
        h     = self.LeakyReLU(self.FC_hidden2(h))
        
        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat
        

class ModelVAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=200, device=DEVICE):
        super(ModelVAE, self).__init__()
        self.device = device

        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)
        z = mean + var * epsilon
        return z
        
                
    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat = self.decoder(z)
        
        return x_hat, mean, log_var
     
    def encode(self, x):
        mean, logvar = self.encoder(x)
        return mean, logvar
    
    def decode(self, x):
        return self.decoder(x)
    

class VAE():
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=200, lr=1e-3, device=DEVICE):
        super(VAE, self).__init__()
        self.device = device
        self.model = ModelVAE(input_dim, hidden_dim, latent_dim, device)
        self.optimizer = Adam(self.model.parameters(), lr=lr)
    
    def loss_function(self, x, x_hat, mean, log_var):
        reproduction_loss = nn.functional.l1_loss(x_hat, x, reduction='mean')
        # norm_x = torch.norm(x, dim=1)
        # norm_x_hat = torch.norm(x_hat, dim=1)

        # dot_product = torch.bmm(x_hat.unsqueeze(1), x.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        # cosine_similarity = dot_product / (norm_x * norm_x_hat)
        
        # reproduction_loss = -torch.mean(cosine_similarity)
        KLD = - 0.5 * torch.mean(1+ log_var - mean.pow(2) - log_var.exp())
        # print(reproduction_loss, KLD)

        return reproduction_loss #+ KLD
    
    def train(self, train_loader):
        self.model.train()

        for epoch in range(epochs):
            overall_loss = 0
            L = 0

            for x in train_loader:
                x = x.to(self.device)

                self.optimizer.zero_grad()

                x_hat, mean, log_var = self.model(x)
                loss = self.loss_function(x, x_hat, mean, log_var)
                
                overall_loss += loss.item()
                L += len(x)
                
                loss.backward()
                self.optimizer.step()

            print(f"Epoch {epoch}, training loss: {overall_loss/L}")
                



class AE():
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=200, lr=1e-3, device=DEVICE):
        super(AE, self).__init__()
        self.device = device
        self.model = NN = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU(),
        )
        self.optimizer = Adam(self.model.parameters(), lr=lr)
    
    def train(self, train_loader):
        self.model.train()

        for epoch in range(epochs):
            overall_loss = 0
            L = 0

            for x in train_loader:
                x = x.to(self.device)

                self.optimizer.zero_grad()

                x_hat = self.model(x)
                loss = nn.functional.l1_loss(x_hat, x, reduction='mean')
                
                overall_loss += loss.item()
                L += len(x)
                
                loss.backward()
                self.optimizer.step()

            print(f"Epoch {epoch}, training loss: {overall_loss/L}")
                