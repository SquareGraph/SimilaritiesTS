"""
Package for clustering latent dimension of input tensors
"""
from typing import Tuple, Union

import faiss
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, moment

__all__ = ['Clusters', 'SingleCluster', 'ClustersDescription']

class Clusters:
    """
    Class for generating clusters of a given latent representation of an initial dataset.

    The Clusters class applies the fit method to train the clustering model (using the k-means algorithm),
    and then applies the get method to obtain the clusters.

    Methods
    -------
    __init__(self, latent_rep: np.ndarray, gpu: bool = False)
        Initialize the instance.

    get(self, k:int, n_iter: int, seed: int = np.random.choice(9999))
        Obtain the clusters.
    """

    def __init__(self,
                 latent_rep: np.ndarray,
                 gpu: bool = False) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        latent_rep : np.ndarray
            The latent representation of the dataset.
        gpu : bool, optional
            Whether to use GPU. Default is False.
        """

        # initialize
        faiss.normalize_L2(latent_rep)

        self.latent = latent_rep
        self.index = faiss.IndexFlatL2(self.latent.shape[1])
        self.gpu = gpu

        if self.gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

    def get(self,
            k: int,
            n_iter: int,
            seed: int = np.random.choice(9999)) -> faiss.Kmeans:
        """
        Obtain the clusters.

        Parameters
        ----------
        k : int
            The number of clusters.
        n_iter : int
            The number of iterations for k-means.
        seed : int, optional
            The seed for random number generation. Default is a random number between 0 and 9999.

        Returns
        -------
        faiss.Kmeans
            The trained k-means model.
        """

        np.random.seed(seed)
        kmeans = faiss.Kmeans(d=self.latent.shape[1],
                              k=k,
                              niter=n_iter,
                              gpu=self.gpu)

        kmeans.train(self.latent)

        return kmeans


class SingleCluster:
    """
    Class that stores and computes data about a single k-means cluster.

    The SingleCluster class computes statistics about the cluster's centroid and provides a method to visualize its distribution.

    Methods
    -------
    __init__(self, cluster: int, centroid: np.ndarray, idxs: np.ndarray, subset: np.ndarray)
        Initialize the instance.

    __calc_data_stats(self, centroid) -> Tuple
        Calculate statistics about the centroid.

    plot(self)
        Plot a histogram of the centroid.

    __getitem__(self, key: str)
        Get the value of an attribute.

    __repr__(self)
        Get a string representation of the instance.
    """

    def __init__(self,
                 cluster: int,
                 centroid: np.ndarray,
                 idxs: np.ndarray,
                 subset: np.ndarray) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        cluster : int
            The cluster number.
        centroid : np.ndarray
            The centroid of the cluster.
        idxs : np.ndarray
            The indices of the elements in the cluster.
        subset : np.ndarray
            The subset of the data in the cluster.
        """

        self.cluster = cluster
        self.idxs = idxs
        self.n_elements = len(subset)
        self.centroid = centroid
        self.mean, self.std, self.skew, self.kutosis, self.moment = self.__calc_data_stats(subset)
        self.subset = subset

    def __calc_data_stats(self, centroid) -> Tuple:
        """
        Calculate statistics about the centroid.

        Parameters
        ----------
        centroid : np.ndarray
            The centroid of the cluster.

        Returns
        -------
        Tuple
            The mean, standard deviation, skewness, kurtosis, and nth moment of the centroid.
        """

        skew_ = skew(centroid)
        kurtosis_ = kurtosis(centroid)
        moment_ = moment(centroid)
        mean_ = centroid.mean(axis=0).mean()
        std_ = centroid.mean(axis=0).std()

        return mean_, std_, skew_, kurtosis_, moment_

    def plot(self) -> None:
        """
        Plot a histogram of the centroid.

        Returns
        -------
        None
        """
        plt.figure()
        plt.title(self.cluster)
        plt.hist(self.centroid,
                 color='orange',
                 label='Centroid values', alpha=0.6, )
        plt.grid()
        plt.legend()
        plt.show()

    def __getitem__(self, key: str):
        """
        Get the value of an attribute.

        Parameters
        ----------
        key : str
            The attribute to get the value of.

        Returns
        -------
        Any
            The value of the attribute.
        """
        return getattr(self, str(key), None)

    def __repr__(self):
        """
        Get a string representation of the instance.

        Returns
        -------
        str
            A string representation of the instance.
        """
        keys = ['n_elements', 'mean', 'std', 'skew', 'kurtosis', 'moment']
        vals = [self[key] for key in keys]
        name = self.__class__.__name__
        sufix = ", ".join(["=".join([k, str(v)]) for k, v in zip(keys, vals)])

        return f"{name}[{self.cluster}]({sufix})"


class ClustersDescription:
    """
    Class that describes a k-means object given by the Clusters.get() method.

    The ClustersDescription class provides details about the k-means clusters including their centroids, indices,
    and a subset of data in each cluster.

    Methods
    -------
    __init__(self, kmeans: faiss.Kmeans, latent: np.ndarray) -> None
        Initialize the instance.

    __getitem__(self, index: Union[int, str])
        Get a specific cluster.

    __len__(self)
        Get the number of clusters.

    __repr__(self)
        Get a string representation of the instance.
    """

    def __init__(self, kmeans: faiss.Kmeans, latent: np.ndarray) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        kmeans : faiss.Kmeans
            The k-means model.
        latent : np.ndarray
            The latent representation of the dataset.
        """

        self.proba, self.index = [a.flatten() for a in kmeans.index.search(latent, 1)]

        for cluster in np.unique(self.index):
            centroid = kmeans.centroids[cluster]
            idxs = np.argwhere(self.index == cluster).flatten()
            sub = latent[idxs]
            setattr(self, str(cluster), SingleCluster(cluster, centroid, idxs, sub))

    def __getitem__(self, index: Union[int, str]):
        """
        Get a specific cluster.

        Parameters
        ----------
        index : Union[int, str]
            The index or name of the cluster to get.

        Returns
        -------
        Any
            The cluster object.
        """
        return getattr(self, str(index), None)

    def __len__(self):
        """
        Get the number of clusters.

        Returns
        -------
        int
            The number of clusters.
        """
        return len(np.unique(self.index))

    def __repr__(self):
        """
        Get a string representation of the instance.

        Returns
        -------
        str
            A string representation of the instance.
        """
        name = self.__class__.__name__
        return f"{name}(n_clusters={len(self)})"
