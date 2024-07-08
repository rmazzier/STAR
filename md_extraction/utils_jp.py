import numpy as np
import pandas as pd


def complex_to_real_vector(complex_vec):
    """Convert a complex-valued K-dimensional vector h_c into real-valued 2K-dimensional vector h_r, s.t.:
    h_r = [Re(h_c) -Im(h_c)]

    Input: Flattened, 1-dimensional complex-valued array of shape (..., n,);
    Output: Flattened, 1-dimensional real-valued array of shape (..., 2n).
    """

    if not (
        complex_vec.dtype == np.dtype("complex128")
        or complex_vec.dtype == np.dtype("complex64")
    ):
        raise Exception("Input 'complex_vec' is not complex.")

    real_vec = np.concatenate([complex_vec.real, -complex_vec.imag], -1)
    return real_vec


def min_max_freq(spec, eps=1e-8):
    """
    Min-max normalization for microDoppler spectrograms in the frequency domain

    Parameters
    ----------
    spec: numpy array, shape (n_time_frames, ndoppler)
        The spectrogram to be normalized

    eps: float (optional)
        Small positive number to avoid division by 0

    Returns
    ----------
    spec: numpy array
        normalized spectrogram with values in [0, 1], shape (n_time_frames, ndoppler)
    """

    return (spec - spec.min(1, keepdims=True)) / (
        spec.max(1, keepdims=True) - spec.min(1, keepdims=True) + eps
    )


def crop_doppler(x, bins):
    """
    Function that crops central frequency bins from spectrogram

    Parameters
    ----------
    x: numpy array, shape (n_time_frames, ndoppler)
        The spectrogram to be processed

    bins: list
        The bin indices list to be removed

    Returns
    ----------
    x: numpy array
        cropped microDoppler spectrogram (n_time_frames, ndoppler - len(bins))
    """
    x = np.delete(x, bins, axis=1)
    return x


def filter_spectrogram(spec):
    """
    Function that removes bad frames from spectrogram

    Parameters
    ----------
    spec: numpy array, shape (n_time_frames, ndoppler)
        The spectrogram to be processed

    Returns
    ----------
    s: numpy array
        cleaned microDoppler spectrogram [dB], with shape (n_time_frames, ndoppler)
    """
    s = np.copy(spec)

    idx = np.where(s.mean(1) > 750)[0]
    s[idx, :] = np.nan

    for dbin in range(spec.shape[1]):
        interp = np.array(pd.DataFrame(s[:, dbin]).interpolate(method="linear"))
        s[:, dbin] = interp.squeeze()
    return s


def complex_to_real_matrix(complex_mtx):
    """Convert complex-valued KxP matrix T_c, to 2Kx2P real-valued matrix T_r, where:
    T_r =   [Re(T) -Im(T)]
            [Im(T)  Re(T)]
    """

    if complex_mtx.dtype != np.dtype("complex128"):
        raise Exception("Input 'complex_vec' is not complex.")

    k = complex_mtx.shape[0]
    p = complex_mtx.shape[1]
    real_mtx = np.zeros((2 * k, 2 * p))

    real_mtx[:k, :p] = complex_mtx.real
    real_mtx[:k, p:] = -complex_mtx.imag
    real_mtx[k:, :p] = complex_mtx.imag
    real_mtx[k:, p:] = complex_mtx.real

    return real_mtx
