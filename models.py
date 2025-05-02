import torch
import torch.nn as nn

"""
GradConCAE and GradConVAE are two classes that define a Convolutional Autoencoder and a Variational Autoencoder respectively.
The GradConCAE class is a Convolutional Autoencoder with 4 downsampling and 4 upsampling layers.
The GradConVAE class is a Variational Autoencoder with 4 downsampling and 4 upsampling layers.
The GradConCAE class is used to reconstruct the input image from the encoded representation.
"""
class GradConCAE(nn.Module):
    def __init__(self, in_channel=3): 
        super(GradConCAE, self).__init__()

        """
        Convolutional Autoencoder with 4 downsampling and 4 upsampling layers.
        The input is a 28x28 image with 3 channels (RGB).
        The downsampling layers use 4x4 kernels with stride 2 and padding 2.
        The upsampling layers use 4x4 kernels with stride 2 and padding 1.
        The activation function used is ReLU for the downsampling layers and Sigmoid for the upsampling layers.

        The model is initialized with normal distribution with mean 0 and std 0.02.
        The model is trained using the Adam optimizer with a learning rate of 0.001.
        The model is evaluated using the Mean Squared Error (MSE) loss function.
        The model is trained for 10 epochs with a batch size of 64.
        """
        self.down = nn.Sequential(
            nn.Conv2d(in_channel, 32, 4, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, stride=2, padding=2),
            nn.ReLU()
        )

        self.up = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=2),  # output 4x4
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # output 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=2),  # output 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channel, 4, stride=2, padding=1),  # output 28x28
            nn.Sigmoid()
        )

        self.sigmoid = nn.Sigmoid() # Sigmoid activation function for the output layer

    """
    Forward pass of the model.
    """
    def forward(self, x):
        z = self.down(x)
        return self.up(z)

"""
GradConVAE is a class that defines a Variational Autoencoder.
The GradConVAE class is a Variational Autoencoder with 4 downsampling and 4 upsampling layers.
The input is a 28x28 image with 3 channels (RGB).
"""
class GradConVAE(nn.Module):
    def __init__(self, in_channel=3):
        super(GradConVAE, self).__init__()

        self.down = nn.Sequential(
            nn.Conv2d(in_channel, 32, 4, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, stride=2, padding=2),
            nn.ReLU()
        )

        """
        3 fully connected layers with 3x3x64 input and output.
        The first two layers are used to compute the mean and log variance of the latent space.
        The third layer is used to compute the latent space representation.
        """
        self.fc11 = nn.Linear(3 * 3 * 64, 3 * 3 * 64)
        self.fc12 = nn.Linear(3 * 3 * 64, 3 * 3 * 64)
        self.fc2 = nn.Linear(3 * 3 * 64, 3 * 3 * 64)


        self.up = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=2),  # output 4x4
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # output 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=2),  # output 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channel, 4, stride=2, padding=1),  # output 28x28
            nn.Sigmoid()
        )

        self.sigmoid = nn.Sigmoid()

    """
    Encodes the input image into a latent space representation.
    """
    def encode(self, x):
        h4 = self.down(x)
        return self.fc11(h4.view(-1, 3 * 3 * 64)), self.fc12(h4.view(-1, 3 * 3 * 64))

    """
    Reparameterization trick to sample from the latent space.
    This is done by adding noise to the mean and log variance of the latent space.
    The noise is sampled from a normal distribution with mean 0 and std 1.
    The reparameterization trick allows us to backpropagate through the sampling process.
    """
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    """
    The decoder takes the latent space representation and reconstructs the input image.
    The decoder is a series of transposed convolutional layers that upsample the latent space representation.
    The output is a 28x28 image with 3 channels (RGB).
    The output is passed through a sigmoid activation function to ensure the output is between 0 and 1.
    """
    def decode(self, z):
        h4_d = self.fc2(z)
        recon = self.up(h4_d.view(-1, 64, 3, 3))
        return recon

    """
    Forward pass of the model.
    The forward pass takes the input image and passes it through the encoder to get the mean and log variance of the latent space.
    The mean and log variance are then passed through the reparameterization trick to get the latent space representation.
    The latent space representation is then passed through the decoder to get the reconstructed image.
    The output is the reconstructed image, the mean and log variance of the latent space.
    The output is passed through a sigmoid activation function to ensure the output is between 0 and 1.
    """
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

"""
This function initializes the weights of the model using a normal distribution with mean and std.
The function takes a model, mean and std as input.
The function iterates through the model and initializes the weights of the model using a normal distribution with mean and std before training.
"""
def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

"""
This function initializes the weights of the model using a normal distribution with mean 0 and std 0.02.
The function takes a model as input.
The function is used to initialize the weights of the model before training.
"""
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
