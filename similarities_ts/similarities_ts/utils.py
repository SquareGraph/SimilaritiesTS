"""
A collection of supporting methods for Similarities TS
"""

from typing import Tuple, List, Literal
from pandas import DataFrame

import numpy as np

__all__ = ['BracketAccess', 'SampleMethods', 'WindowTransform']

class BracketAccess(type):
    """
    A metaclass that enables bracket notation for attribute access in its classes.

    Classes that use this metaclass can access their attributes using brackets ([]),
    like dictionary objects, instead of the conventional dot (.) notation.

    Attributes
    ----------
    key : str
        The attribute name that is being accessed.
    """

    def __getitem__(cls, key: str):
        """
        Retrieve an attribute of the class using bracket notation.

        Parameters
        ----------
        key : str
            The attribute name that is being accessed.

        Returns
        -------
        The attribute's value, if it exists; otherwise, `None`.
        """
        return getattr(cls, key, None)


class SampleMethods(metaclass=BracketAccess):
    """
    A collection of methods for generating random datasets and processing them.
    The class uses the BracketAccess metaclass to allow bracket notation for method access.

    Methods
    -------
    noise(scale: float, num_time_steps: int) -> np.ndarray
        Generate a random noise dataset.

    brownian_motion(num_time_steps: int, initial_value: int, drift=0.0,
                    volatility=1.0, dt=1.0) -> np.ndarray
        Generate a Brownian motion path.

    random_oscillator(uniform_range: Tuple[int], num_time_steps: int) -> np.ndarray
        Generate a random oscillator dataset.

    standardize(input_array: np.ndarray) -> np.ndarray
        Standardize an array.

    normalize(input_array: np.ndarray) -> np.ndarray
        Normalize an array.

    random_dataset(n_series: int,
                   series_types: List[Literal['noise', 'brownian_motion', 'random_oscillator']]) -> pd.DataFrame
        Generate a DataFrame of random datasets of specified types.

    all() -> List[str]
        Get a list of all public methods.
    """

    @staticmethod
    def noise(scale: float,
              num_time_steps: int,
              **kwargs) -> np.ndarray:
        """
        Generate a random noise dataset.

        Parameters
        ----------
        scale : float
            The standard deviation of the normal distribution.
        num_time_steps : int
            The number of time steps.

        Returns
        -------
        np.ndarray
            The generated random noise.
        """
        return np.random.normal(scale=scale, size=num_time_steps)

    @staticmethod
    def brownian_motion(num_time_steps: int,
                        initial_value: int,
                        drift=0.0,
                        volatility=1.0,
                        dt=1.0,
                        **kwargs):

        """
        Generate a Brownian motion path.

        Parameters
        ----------
        num_time_steps : int
            The number of time steps.
        initial_value : int
            The initial value of the path.
        drift : float, optional
            The drift of the Brownian motion, by default 0.0.
        volatility : float, optional
            The volatility of the Brownian motion, by default 1.0.
        dt : float, optional
            The time step size, by default 1.0.

        Returns
        -------
        np.ndarray
            The generated Brownian motion path.
        """

        increments = np.random.normal(loc=drift * dt, scale=volatility * np.sqrt(dt), size=num_time_steps)

        # Generate forward Brownian motion path
        path = np.cumsum(increments) + initial_value

        return path

    @staticmethod
    def random_oscillator(uniform_range: Tuple[int],
                          num_time_steps: int,
                          **kwargs) -> np.ndarray:
        """
        Generate a random oscillator dataset.

        Parameters
        ----------
        uniform_range : Tuple[int]
            The range of the uniform distribution to draw from.
        num_time_steps : int
            The number of time steps.

        Returns
        -------
        np.ndarray
            The generated random oscillator.
        """

        return np.cos(np.random.uniform(*uniform_range, num_time_steps))

    @staticmethod
    def standardize(input_array: np.ndarray) -> np.ndarray:

        """
        Standardize an array.

        Parameters
        ----------
        input_array : np.ndarray
            The input array.

        Returns
        -------
        np.ndarray
            The standardized array.
        """

        return (input_array - input_array.mean()) / input_array.std()

    @staticmethod
    def normalize(input_array: np.ndarray) -> np.ndarray:

        """
        Normalize an array.

        Parameters
        ----------
        input_array : np.ndarray
            The input array.

        Returns
        -------
        np.ndarray
            The normalized array.
        """

        return (input_array - input_array.min()) / (input_array.max() - input_array.min())

    @staticmethod
    def random_dataset(n_series: int,
                       series_types: List[Literal['noise', 'brownian_motion', 'random_oscillator']],
                       **kwargs) -> DataFrame:

        """
        Generate a DataFrame of random datasets of specified types.

        Parameters
        ----------
        n_series : int
            The number of series to generate.
        series_types : List[Literal['noise', 'brownian_motion', 'random_oscillator']]
            The types of series to generate.

        Returns
        -------
        pd.DataFrame
            The DataFrame of generated datasets.
        """

        assert len(series_types) == n_series, AssertionError('len(series_types) must be equal to n_series')
        for name in series_types:
            assert name in ['noise', 'brownian_motion', 'random_oscillator'], AssertionError(
                f"{name} must be one of ['noise', 'brownian_motion', 'ranom_oscillator']")

        dataset = {}

        for a, name in enumerate(series_types):
            data = SampleMethods[name](**kwargs)

            if 'transform' in kwargs.keys():
                match kwargs['transform']:
                    case 'normalize':
                        data = SampleMethods.normalize(data)
                    case 'standardize':
                        data = SampleMethods.standardize(data)
                    case other:
                        pass

            dataset[f"{name}_{a}"] = data

        return DataFrame(dataset)

    @staticmethod
    def all() -> List[str]:

        """
        Get a list of all public methods.

        Returns
        -------
        List[str]
            The list of all public methods.
        """

        return [key for key in SampleMethods.__dict__.keys() if not key.startswith('_')][:-1]


class WindowTransform(metaclass=BracketAccess):
    """
    A class for transforming 1D data into a 2D representation using window methods.
    The class uses the BracketAccess metaclass to allow bracket notation for method access.

    Methods
    -------
    sliding_window(df: pd.DataFrame, window_length: int) -> np.ndarray
        Create a sliding window view of the input DataFrame.

    non_overlapping_window(df: pd.DataFrame, window_length: int) -> np.ndarray
        Create a non-overlapping window view of the input DataFrame.

    all() -> List[str]
        Get a list of all public methods.
    """

    @staticmethod
    def sliding_window(df: DataFrame, window_length: int) -> np.ndarray:
        """
        Create a sliding window view of the input DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame.
        window_length : int
            The length of the window.

        Returns
        -------
        np.ndarray
            The 2D array of the sliding window view.
        """
        sw = np.squeeze(np.lib.stride_tricks.sliding_window_view(df.values, (window_length,df.shape[-1])))
        return np.swapaxes(sw, 1,-1)

    @staticmethod
    def non_overlapping_window(df: DataFrame, window_length: int) -> np.ndarray:
        """
        Create a non-overlapping window view of the input DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame.
        window_length : int
            The length of the window.

        Returns
        -------
        np.ndarray
            The 2D array of the non-overlapping window view.
        """
        idxs = np.arange(df.shape[0]//window_length) * window_length
        stacked = np.dstack([df.iloc[idx:idx+window_length].values for idx in idxs])
        return np.swapaxes(stacked, 0,-1)

    @staticmethod
    def all():
        """
        Get a list of all public methods.

        Returns
        -------
        List[str]
            The list of all public methods.
        """
        return [key for key in WindowTransform.__dict__.keys() if not key.startswith('_')][:-1]
