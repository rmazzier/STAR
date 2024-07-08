import numpy as np
from scipy.stats import iqr
import torch

from .utils_jp import *
from .sparsity_based import moving_iqr


def compute_mD(
    cir,
    params,
    filter_lines=False,
    crop=False,
    normalize=False,
    out_db=True,
    n_kept_bins=None,
):
    """
    Function that returns the microDoppler spectrogram of an input CIR sequence

    Parameters
    ----------
    cir: numpy array, shape (n_range_bins, n_packets, n_bp)
        The CIR sequence to be processed

    params: dict
        Dictionary of input parameters for the spectrogram

    filter_lines: bool
        Flag to enable filtering out bad frames

    crop: bool
        Flag to enable cropping central frequency bins

    normalize: bool
        Flag to enable frequency domain normalization

    Returns
    ----------
    mD: numpy array
        microDoppler spectrogram [dB], with shape (n_time_frames, ndoppler)
    """

    cir = cir[:, :, params["BP_SEL"]]
    cir -= cir.mean(1, keepdims=True)
    mD, chunks = mD_spectrum(
        cir,
        nwin=params["NWIN"],
        step=params["TREP"],
        ndoppler=params["Nd"],
        out_db=out_db,
        n_kept_bins=n_kept_bins,
    )

    # select bins of interest from the spectrum and sum
    mD = mD.sum(1)
    mD_shift = np.zeros_like(mD)
    mD_shift[:, : mD.shape[1] // 2] = mD[:, mD.shape[1] // 2 :]
    mD_shift[:, mD.shape[1] // 2 :] = mD[:, : mD.shape[1] // 2]
    mD = mD_shift

    if crop:
        mD = crop_doppler(mD, list(range(30, 35)))

    if filter_lines:
        mD = filter_spectrogram(mD)

    if normalize:
        mD = min_max_freq(mD)

    return mD, chunks


def mD_spectrum(complex_cir, nwin, step, ndoppler, out_db=True, n_kept_bins=None):
    """
    Function that computes the microDoppler spectrogram of an input CIR sequence

    Parameters
    ----------
    complex_cir: numpy array, shape (n_range_bins, n_packets, n_bp)
        The CIR sequence to be processed

    nwin: int
        Length of an input CIR window in samples

    step: int
        Length of a step for the spectrogram samples

    ndoppler: int
        Number of Doppler bins in the DFT

    Returns
    ----------
    spec: numpy array
        microDoppler spectrogram [dB], with shape (n_time_frames, ndoppler)
    """
    chunks = []
    spec = []
    for i in range(0, complex_cir.shape[1], step):
        chunk = complex_cir[:, i : i + nwin]
        win = np.hanning(chunk.shape[1]).reshape(1, -1)
        windowed = chunk * win
        spectrum = np.fft.fft(windowed, n=ndoppler, axis=1)

        if n_kept_bins is not None:
            # Tracking on the range bin, by selecting the top k rows with
            # the highest moving interquartile range

            real_chunk = torch.tensor(complex_to_real_vector(chunk))

            iqrs_r = moving_iqr(
                real_chunk[:, : real_chunk.shape[1] // 2], k=16, moving=False
            )
            iqrs_i = moving_iqr(
                real_chunk[:, real_chunk.shape[1] // 2 :], k=16, moving=False
            )
            iqrs = torch.norm(torch.stack((iqrs_r, iqrs_i), axis=0), dim=0)
            topn_idx = torch.topk(iqrs, n_kept_bins, largest=True, sorted=True)[1]

            if n_kept_bins == 1:
                chunks.append(np.expand_dims(chunk[topn_idx, :], 0))
                spec.append(np.expand_dims(spectrum[topn_idx, :], 0))
            else:
                chunks.append(chunk[topn_idx, :])
                spec.append(spectrum[topn_idx, :])

        else:
            chunks.append(chunk)
            spec.append(spectrum)
    spec = np.array(spec)

    if out_db:
        spec = 20 * np.log10(np.abs(spec))
    return spec, chunks
