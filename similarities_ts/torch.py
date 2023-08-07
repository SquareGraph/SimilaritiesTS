"""
Package that implements neural-network based methods for dimensionality reduction
"""
from .utils import WindowTransform, BracketAccess

from typing import List, Literal, Tuple, Sequence
from enum import Enum

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.transformer import MultiheadAttention
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
from pandas import DataFrame
from numpy import vstack
from tqdm.autonotebook import tqdm

__all__ = ['DEVICE', 'MultivariateDataset','Encoder','Decoder','Metrics','HybridVAE','Optimizers','Reducer']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultivariateDataset(Dataset):
    """
    A class for turning a DataFrame of series into multivariate multiple (windowed) time series datasets,
    of given chunk sizes. The class inherits from PyTorch's Dataset class.

    Methods
    -------
    __new__(cls, mode: Literal['sliding_window', 'non_overlapping_window'], df: pd.DataFrame,
            window_length: int, transform=None) -> 'MultivariateDataset'
        Create a new instance of the class.

    __init__(self, mode: Literal['sliding_window', 'non_overlapping_window'], df: pd.DataFrame,
             window_length: int, transform=None)
        Initialize the instance.

    __len__(self) -> int
        Get the length of the dataset.

    __getitem__(self, index) -> torch.Tensor
        Get an item from the dataset.
    """

    def __new__(cls,
                mode: Literal['sliding_window', 'non_overlapping_window'],
                df: DataFrame,
                window_length: int,
                transform=None):
        """
        Create a new instance of the class.

        Parameters
        ----------
        mode : Literal['sliding_window', 'non_overlapping_window']
            The window mode.
        df : pd.DataFrame
            The input DataFrame.
        window_length : int
            The length of the window.
        transform : callable, optional
            The transform to apply to the data.

        Returns
        -------
        MultivariateDataset
            The new instance of the class.
        """

        assert mode in WindowTransform.all(), AssertionError(f'mode must be one of {WindowTransform.methods()}')

        return super().__new__(cls)

    def __init__(self,
                 mode: Literal['sliding_window', 'non_overlapping_window'],
                 df: DataFrame,
                 window_length: int,
                 transform=None) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        mode : Literal['sliding_window', 'non_overlapping_window']
            The window mode.
        df : pd.DataFrame
            The input DataFrame.
        window_length : int
            The length of the window.
        transform : callable, optional
            The transform to apply to the data.
        """

        global DEVICE

        self.tensors = torch.FloatTensor(WindowTransform[mode](df, window_length)).to(DEVICE)
        self.transform = transform

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.
        """
        return len(self.tensors)

    def __getitem__(self, index) -> torch.Tensor:
        """
        Get an item from the dataset.

        Parameters
        ----------
        index : int
            The index of the item.

        Returns
        -------
        torch.Tensor
            The item from the dataset.
        """
        sample = self.tensors[index]

        if self.transform:
            sample = self.transofrm(sample)

        return sample


class Encoder(nn.Module):
    """
    Original inspiration: https://arxiv.org/pdf/2303.07048, credits to Borui Cai, Shuiqiao Yang, Longxiang Gao, Yong Xiang
    Implementational changes: LeakyReLU instead of ReLu for conv transpose layers, customized number of layers, upsampling in Decoder.

    Encoder class for a Variational Autoencoder (VAE) Neural Network.
    Inherits from PyTorch's nn.Module class.

    The encoder applies a series of convolutional layers, a multi-head self-attention mechanism,
    and fully connected layers to output the mean and standard deviation vectors for the VAE's latent space.

    Methods
    -------
    __init__(self, input_dim, conv_filters, conv_kernel_size, conv_strides, attention_heads, latent_dim)
        Initialize the instance.

    add_conv_layer(self, in_channels, out_channels, kernel_size, stride)
        Add a convolutional layer followed by a ReLU activation to the encoder.

    forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]
        Forward pass through the encoder.

    reparameterize(self, mu, logvar) -> torch.Tensor
        Generate a random sample from the distribution defined by mu and logvar.
    """

    def __init__(self,
                 input_dim: int,
                 conv_filters: Sequence[int],
                 conv_kernel_size: Sequence[int],
                 conv_strides: Sequence[int],
                 attention_heads: int,
                 latent_dim: int) -> None:

        """
        Initialize the instance.

        Parameters
        ----------
        input_dim : int
            The dimension of the input data.
        conv_filters : List[int]
            The number of filters for each convolutional layer.
        conv_kernel_size : List[int]
            The kernel size for each convolutional layer.
        conv_strides : List[int]
            The stride for each convolutional layer.
        attention_heads : int
            The number of attention heads for the multi-head self-attention mechanism.
        latent_dim : int
            The dimension of the latent space.
        """

        super(Encoder, self).__init__()

        # Initialize lists for convolutional layers
        self.conv_layers = nn.ModuleList()

        # Add convolutional layers
        for i in range(len(conv_filters)):
            self.add_conv_layer(input_dim if i == 0 else conv_filters[i - 1],
                                conv_filters[i], conv_kernel_size[i], conv_strides[i])

        # Multi-head self-attention mechanism
        self.self_attention = MultiheadAttention(embed_dim=conv_filters[-1], num_heads=attention_heads)

        # Fully connected layers to output the mean and standard deviation vectors
        self.fc_mu = nn.Linear(conv_filters[-1], latent_dim)
        self.fc_logvar = nn.Linear(conv_filters[-1], latent_dim)

    def add_conv_layer(self,
                       in_channels: int,
                       out_channels: int,
                       kernel_size: int,
                       stride: int) -> None:
        """
        Add a convolutional layer followed by a ReLU activation to the encoder.

        Parameters
        ----------
        in_channels : int
            The number of input channels.
        out_channels : int
            The number of output channels.
        kernel_size : int
            The size of the kernel.
        stride : int
            The stride of the convolution.
        """
        # Function to add a Convolutional layer followed by a ReLU activation
        self.conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, stride))
        self.conv_layers.append(nn.ReLU())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Forward pass through the encoder.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The mean and standard deviation vectors.
        """

        # Pass through each Convolutional layer
        for layer in self.conv_layers:
            x = layer(x)

        # Store the output shape of the last Convolutional layer
        self.last_conv_output_shape = x.shape

        # Reshape x to match what the multi-head attention layer expects
        x = x.permute(2, 0, 1)  # shape becomes (L, N, E)

        # Apply self-attention
        x, _ = self.self_attention(x, x, x)

        # Fully connected layers to output the mean and standard deviation vectors
        x = x.mean(dim=0)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar

    def reparameterize(self,
                       mu: torch.Tensor,
                       logvar: torch.Tensor) -> torch.Tensor:
        """
        Generate a random sample from the distribution defined by mu and logvar.

        Parameters
        ----------
        mu : torch.Tensor
            The mean vector.
        logvar : torch.Tensor
            The log variance vector.

        Returns
        -------
        torch.Tensor
            The generated sample.
        """
        # Function to generate a random sample from the distribution defined by mu and logvar
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class Decoder(nn.Module):
    """
    Original inspiration: https://arxiv.org/pdf/2303.07048, credits to Borui Cai, Shuiqiao Yang, Longxiang Gao, Yong Xiang
    Implementational changes: LeakyReLU instead of ReLu for conv transpose layers, customized number of layers, upsampling in Decoder.

    Decoder class for a Variational Autoencoder (VAE) Neural Network.
    Inherits from PyTorch's nn.Module class.

    The decoder applies a fully connected layer, an upsample layer, and a series of convolutional transpose layers
    to the input from the VAE's latent space.

    Methods
    -------
    __init__(self, latent_dim: int, hidden_dim: int, conv_transpose_filters: Sequence[int],
             conv_transpose_kernel_sizes: Sequence[int], conv_transpose_strides: Sequence[int], upsample: int)
        Initialize the instance.

    add_conv_transpose_layer(self, in_channels, out_channels, kernel_size, stride)
        Add a convolutional transpose layer followed by a LeakyReLU activation to the decoder.

    forward(self, z) -> torch.Tensor
        Forward pass through the decoder.
    """

    def __init__(self,
                 latent_dim: int,
                 hidden_dim: int,
                 conv_transpose_filters: Sequence[int],
                 conv_transpose_kernel_sizes: Sequence[int],
                 conv_transpose_strides: Sequence[int],
                 upsample: int) -> None:

        """
        Initialize the instance.

        Parameters
        ----------
        latent_dim : int
            The dimension of the latent space.
        hidden_dim : int
            The dimension of the hidden layer.
        conv_transpose_filters : Sequence[int]
            The number of filters for each convolutional transpose layer.
        conv_transpose_kernel_sizes : Sequence[int]
            The kernel size for each convolutional transpose layer.
        conv_transpose_strides : Sequence[int]
            The stride for each convolutional transpose layer.
        upsample : int
            The scale factor for the upsample layer.
        """

        super(Decoder, self).__init__()

        # Fully connected layer
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.hidden_dim = hidden_dim

        # Upsample layer
        self.upsample = nn.Upsample(scale_factor=upsample)  # adjust this value as needed

        # Initialize list for Convolutional Transpose layers
        self.conv_transpose_layers = nn.ModuleList()

        # Add Convolutional Transpose layers
        for i in range(len(conv_transpose_filters)):
            self.add_conv_transpose_layer(hidden_dim if i == 0 else conv_transpose_filters[i - 1],
                                          conv_transpose_filters[i], conv_transpose_kernel_sizes[i],
                                          conv_transpose_strides[i])

    def add_conv_transpose_layer(self,
                                 in_channels: int,
                                 out_channels: int,
                                 kernel_size: int,
                                 stride: int) -> None:

        """
        Add a convolutional transpose layer followed by a LeakyReLU activation to the decoder.

        Parameters
        ----------
        in_channels : int
            The number of input channels.
        out_channels : int
            The number of output channels.
        kernel_size : int
            The size of the kernel.
        stride : int
            The stride of the convolution transpose.
        """

        # Function to add a Convolutional Transpose layer followed by a ReLU activation
        self.conv_transpose_layers.append(nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride))
        self.conv_transpose_layers.append(nn.LeakyReLU())

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder.

        Parameters
        ----------
        z : torch.Tensor
            The input tensor from the VAE's latent space.

        Returns
        -------
        torch.Tensor
            The output tensor.
        """
        # Fully connected layer
        z = F.relu(self.fc(z))

        # Reshape to 3D tensor
        z = z.view(-1, self.hidden_dim, 1)

        # Upsample
        z = self.upsample(z)

        # Pass through each Convolutional Transpose layer
        for layer in self.conv_transpose_layers:
            z = layer(z)

        return z


class HybridVAE(nn.Module):
    """
    Original inspiration: https://arxiv.org/pdf/2303.07048, credits to Borui Cai, Shuiqiao Yang, Longxiang Gao, Yong Xiang
    Implementational changes: LeakyReLU instead of ReLu for conv transpose layers, customized number of layers, upsampling in Decoder.

    Hybrid Variational Autoencoder (VAE) class that combines an encoder and decoder,
    inheriting from PyTorch's nn.Module class.

    The HybridVAE applies the encoder to the input data to generate a latent representation,
    and then applies the decoder to the latent representation to generate the output data.

    Methods
    -------
    __init__(self, input_dim, latent_dim, encoder_params, decoder_params)
        Initialize the instance.

    forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Forward pass through the HybridVAE.
    """

    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 encoder_params: Sequence[List[int], List[int], List[int], int],
                 decoder_params: Sequence[int, List[int], List[int], List[int], int]) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        input_dim : int
            The dimension of the input data.
        latent_dim : int
            The dimension of the latent space.
        encoder_params : tuple
            The parameters for the encoder.
        decoder_params : tuple
            The parameters for the decoder.
        """

        super(HybridVAE, self).__init__()

        self.encoder = Encoder(input_dim, *encoder_params, latent_dim)
        self.decoder = Decoder(latent_dim, *decoder_params)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the HybridVAE.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            The output from the decoder, and the mean and log variance vectors from the encoder.
        """

        mu, logvar = self.encoder(x)
        z = self.encoder.reparameterize(mu, logvar)

        return self.decoder(z), mu, logvar


class Metrics(metaclass=BracketAccess):
    """
    A collection of methods for calculating metrics. The class uses the BracketAccess metaclass
    to allow bracket notation for method access.

    Methods
    -------
    mae(y_true, y_pred) -> torch.Tensor
        Calculate the Mean Absolute Error (MAE).

    mse(y_true, y_pred) -> torch.Tensor
        Calculate the Mean Squared Error (MSE).

    mase(y_true, y_pred, y_naive) -> torch.Tensor
        Calculate the Mean Absolute Scaled Error (MASE).

    all() -> List[str]
        Get a list of all public methods.
    """

    @staticmethod
    def mae(y_true: torch.Tensor,
            y_pred: torch.Tensor,
            **kwargs) -> torch.Tensor:
        """
        Calculate the Mean Absolute Error (MAE).

        Parameters
        ----------
        y_true : torch.Tensor
            The ground truth tensor.
        y_pred : torch.Tensor
            The predicted tensor.

        Returns
        -------
        torch.Tensor
            The MAE.
        """
        return torch.mean(torch.abs(y_true - y_pred))

    @staticmethod
    def mse(y_true: torch.Tensor,
            y_pred: torch.Tensor,
            **kwargs) -> torch.Tensor:
        """
        Calculate the Mean Squared Error (MSE).

        Parameters
        ----------
        y_true : torch.Tensor
            The ground truth tensor.
        y_pred : torch.Tensor
            The predicted tensor.

        Returns
        -------
        torch.Tensor
            The MSE.
        """
        return torch.mean((y_true - y_pred) ** 2)

    @staticmethod
    def mase(y_true: torch.Tensor,
             y_pred: torch.Tensor,
             y_naive: torch.Tensor,
             **kwargs) -> torch.Tensor:
        """
        Calculate the Mean Absolute Scaled Error (MASE).

        Parameters
        ----------
        y_true : torch.Tensor
            The ground truth tensor.
        y_pred : torch.Tensor
            The predicted tensor.
        y_naive : torch.Tensor
            The naive forecast.

        Returns
        -------
        torch.Tensor
            The MASE.
        """
        mae = Metrics.mae(y_true, y_pred)
        scale = Metrics.mae(y_true, y_naive)
        return mae / scale

    @staticmethod
    def all() -> List[str]:
        """
        Get a list of all public methods.

        Returns
        -------
        List[str]
            The list of all public methods.
        """
        return [key for key in Metrics.__dict__.keys() if not key.startswith("_")][:-1]


Optimizers = Enum("Optimizers", {
 'Adadelta': torch.optim.Adadelta,
 'Adagrad': torch.optim.Adagrad,
 'Adam': torch.optim.Adam,
 'AdamW': torch.optim.AdamW,
 'SparseAdam': torch.optim.SparseAdam,
 'Adamax': torch.optim.Adamax,
 'ASGD': torch.optim.ASGD,
 'SGD': torch.optim.SGD,
 'RAdam': torch.optim.RAdam,
 'Rprop': torch.optim.Rprop,
 'RMSprop': torch.optim.RMSprop,
 'Optimizer': torch.optim.Optimizer,
 'NAdam': torch.optim.NAdam,
 'LBFGS': torch.optim.LBFGS,})


class Reducer:
    """
    Class for training a Variational Autoencoder (VAE) and obtaining the latent representation of a given dataset.

    The Reducer applies the fit method to train the VAE, and then applies the latent_rep method to get the latent representation.

    Methods
    -------
    __init__(self, dataset: MultivariateDataset, batch_size: int, optimizer: Literal[...], latent_dim: int,
             conv_filters: Sequence[int], conv_kernel_size: Sequence[int], conv_strides: Sequence[int],
             attention_heads: int, hidden_dim: int, conv_transpose_filters: Sequence[int],
             conv_transpose_kernel_sizes: Sequence[int], conv_transpose_strides: Sequence[int], upsample: int)
        Initialize the instance.

    loss_hybrid_vae(recon_x, x, mu, logvar) -> torch.Tensor
        Calculate the loss of the VAE.

    fit(self, epochs: int = 5, metrics: Sequence[str] = None, schedule: bool = False, **kwargs) -> None
        Fit the VAE model.

    generate(self) -> Sequence
        Generate new data.

    latent_rep(self, as_numpy: bool = True) -> Sequence
        Get the latent representation of the dataset.

    decode(latent_rep: List[torch.Tensor]) -> Sequence
        Decode the latent representation back to the original space.
    """

    def __new__(cls,
                dataset: MultivariateDataset,
                batch_size: int,
                optimizer: Literal[
                    'Adadelta', 'Adagrad', 'Adam', 'AdamW', 'SparseAdam', 'Adamax', 'ASGD', 'SGD', 'RAdam', 'Rprop', 'RMSprop', 'Optimizer', 'NAdam', 'LBFGS'],
                latent_dim: int,
                conv_filters: Sequence[int],
                conv_kernel_size: Sequence[int],
                conv_strides: Sequence[int],
                attention_heads: int,
                hidden_dim: int,
                conv_transpose_filters: Sequence[int],
                conv_transpose_kernel_sizes: Sequence[int],
                conv_transpose_strides: Sequence[int],
                upsample: int):

        assert len(conv_filters) == len(conv_kernel_size) == len(conv_strides), AssertionError(
            "All encoder arguments have to have same length")
        assert isinstance(dataset, MultivariateDataset), AssertionError(
            "Dataset have to be of type MultivariateDataset")

        return super().__new__(cls)

    def __init__(self,
                 dataset: MultivariateDataset,
                 batch_size: int,
                 optimizer: Literal[
                     'Adadelta', 'Adagrad', 'Adam', 'AdamW', 'SparseAdam', 'Adamax', 'ASGD', 'SGD', 'RAdam', 'Rprop', 'RMSprop', 'Optimizer', 'NAdam', 'LBFGS'],
                 latent_dim: int,
                 conv_filters: Sequence[int],
                 conv_kernel_size: Sequence[int],
                 conv_strides: Sequence[int],
                 attention_heads: int,
                 hidden_dim: int,
                 conv_transpose_filters: Sequence[int],
                 conv_transpose_kernel_sizes: Sequence[int],
                 conv_transpose_strides: Sequence[int],
                 upsample: int) -> None:

        """
        Initialize the instance.

        Parameters
        ----------
        dataset : MultivariateDataset
            The dataset to reduce.
        batch_size : int
            The size of the batches for training.
        optimizer : Literal[...]
            The optimizer to use for training.
        latent_dim : int
            The dimension of the latent space.
        conv_filters : Sequence[int]
            The number of filters for each convolutional layer in the encoder.
        conv_kernel_size : Sequence[int]
            The kernel size for each convolutional layer in the encoder.
        conv_strides : Sequence[int]
            The stride for each convolutional layer in the encoder.
        attention_heads : int
            The number of attention heads for the multi-head self-attention mechanism in the encoder.
        hidden_dim : int
            The dimension of the hidden layer in the decoder.
        conv_transpose_filters : Sequence[int]
            The number of filters for each convolutional transpose layer in the decoder.
        conv_transpose_kernel_sizes : Sequence[int]
            The kernel size for each convolutional transpose layer in the decoder.
        conv_transpose_strides : Sequence[int]
            The stride for each convolutional transpose layer in the decoder.
        upsample : int
            The scale factor for the upsample layer in the decoder.
        """

        global DEVICE

        self.data_loader = DataLoader(dataset, batch_size=batch_size)
        self.model = HybridVAE(input_dim=dataset.tensors.shape[1],
                               latent_dim=latent_dim,
                               encoder_params=(conv_filters, conv_kernel_size, conv_strides, attention_heads),
                               decoder_params=(hidden_dim, conv_transpose_filters,
                                               conv_transpose_kernel_sizes, conv_transpose_strides,
                                               upsample)).to(DEVICE)
        self.optimizer = Optimizers[optimizer].value(params=self.model.parameters())

    @staticmethod
    def loss_hybrid_vae(recon_x: torch.Tensor,
                        x: torch.Tensor,
                        mu: torch.Tensor,
                        logvar: torch.Tensor) -> torch.Tensor:
        """
        Calculate the loss of the VAE.

        Parameters
        ----------
        recon_x : torch.Tensor
            The reconstructed tensor.
        x : torch.Tensor
            The original input tensor.
        mu : torch.Tensor
            The mean vector.
        logvar : torch.Tensor
            The log variance vector.

        Returns
        -------
        torch.Tensor
            The computed loss.
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x)

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss
        loss = recon_loss + kl_loss

        return loss

    def fit(self,
            epochs: int = 5,
            metrics: None | Literal['mse', 'mase', 'mae'] = None,
            schedule: bool = False,
            **kwargs) -> None:

        """
        Fit the VAE model.

        Parameters
        ----------
        epochs : int, optional
            The number of epochs to train for. Default is 5.
        metrics :  None | Literal['mse','mase','mae']
            The metrics to compute during training. Default is None.
        schedule : bool, optional
            Whether to use a learning rate scheduler. Default is False.
        **kwargs
            Additional keyword arguments, accordingly to provided metrics

        Returns
        -------
        None
        """

        if metrics != None:
            assert hasattr(metrics, "__iter__"), AssertionError('If not none, metrics have to be iterable object')
            for name in metrics:
                assert name in Metrics.all(), AssertionError(f"{name} must be one of {Metrics.all()}.")

        desc = "Fitting VAE model on dataset..."

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')

        # Training loop
        for epoch in tqdm(range(epochs), total=epochs, desc=desc):
            summary = {'epoch': epoch + 1}
            for batch in self.data_loader:
                x = batch
                self.optimizer.zero_grad()
                recon_x, mu, logvar = self.model(x)
                loss = Reducer.loss_hybrid_vae(recon_x, x, mu, logvar)
                summary['loss'] = loss.item()  # .:4f
                if metrics is not None:
                    for name in metrics:
                        summary[name] = Metrics[name](recon_x, x, **kwargs).item()  #:.4f

                loss.backward()
                self.optimizer.step()
            print(summary)

            if schedule:
                scheduler.step(loss)

    def generate(self) -> Sequence:
        """
        NOT IMPLEMENTED YET

        Generate new data.

        Returns
        -------
        Sequence
            The generated data.
        """
        pass

    def latent_rep(self, as_numpy: bool = True) -> Sequence:

        """
        Get the latent representation of the dataset.

        Parameters
        ----------
        as_numpy : bool, optional
            Whether to return the latent representation as a NumPy array. Default is True.

        Returns
        -------
        Sequence
            The latent representation.
        """

        encoded = []
        with torch.no_grad():
            for batch in tqdm(self.data_loader):
                x = batch
                mu, logvar = self.model.encoder(x)
                z = self.model.encoder.reparameterize(mu, logvar)
                encoded.append(z)

        if as_numpy:
            return vstack([item.cpu().numpy() for item in encoded])

        return encoded

    def decode(self, latent_rep: List[torch.Tensor]) -> Sequence:
        """
        Decode the latent representation back to the original space.

        Parameters
        ----------
        latent_rep : List[torch.Tensor]
            The latent representation.

        Returns
        -------
        Sequence
            The decoded data.
        """

        return [self.model.decoder(entry) for entry in latent_rep]
