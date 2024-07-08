import numpy as np
import torch
from scipy.linalg import dft
from scipy.stats import iqr

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import sys

sys.path.append("../")

from utils import real_to_complex_vector, process_cpx_crop

from md_extraction.utils_jp import *


def moving_iqr(X, moving=True, alpha=0.1, k=8):
    # compute moving standard deviation of differentiated signal
    X_diff = X[:, 1:] - X[:, :-1]
    # add a column of zeros to the left of the signal
    X_diff = torch.cat((torch.zeros(X.shape[0], 1), X_diff), dim=1)

    if moving == False:
        out = torch.tensor(iqr(X_diff, axis=1))
        return out
    else:
        out = []
        for i in range(X.shape[1]):
            # compute std of last k samples of differentiated signal and compute ema
            if i > k:
                diff_std = (
                    alpha * iqr(X_diff[:, i - k : i + 1], axis=1)
                    + (1 - alpha) * diff_std
                )
            else:
                diff_std = iqr(X_diff[:, : i + 2], axis=1)

            out.append(torch.tensor(diff_std))

        # return spikes, moving_std, moving_threshold
        return torch.stack(out, axis=1)


def compute_mD(cir, params, mask, filter_lines=False, crop=False, normalize=False):

    cir = cir[:, :, params["BP_SEL"]]
    cir -= cir.mean(1, keepdims=True)
    mD = mD_spectrum(cir, mask, nwin=params["NWIN"], step=params["TREP"])

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

    return mD


def compute_mD_(cir, params, mask, filter_lines=False, crop=False, normalize=False):

    cir = cir[:, :, params["BP_SEL"]]
    cir -= cir.mean(1, keepdims=True)
    _, mD = mD_spectrum_(cir, mask, nwin=params["NWIN"], step=params["TREP"])
    mD = np.abs(mD) ** 2

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
    return mD


def mD_spectrum(complex_cir, mask, nwin, step):
    spec = []
    for i in range(0, complex_cir.shape[1] - nwin, step):
        curr_mask = mask[i]
        keep_idx = np.argwhere(curr_mask).squeeze()
        chunk = complex_cir[:, i : i + nwin]
        partial_chunk = chunk[:, keep_idx]
        win = np.hanning(chunk.shape[1]).reshape(1, -1)
        partial_win = win[:, keep_idx]
        psi = partial_fourier(nwin, keep_idx)
        rep_psi = np.tile(psi, (complex_cir.shape[0], 1, 1))
        spectrum = iht(rep_psi, partial_chunk * partial_win)
        # print(spectrum)
        spec.append(spectrum)
    spec = np.array(spec).squeeze()
    # spec = 20 * np.log10(np.abs(spec))
    spec = np.abs(spec) ** 2
    return spec


def mD_spectrum_(complex_cir, nwin, step, n_kept_bins):

    chunks = []
    mD_columns = []
    full_spec = []

    for i in range(0, complex_cir.shape[1] - nwin, step):
        chunk = complex_cir[:, i : i + nwin]

        # ==== IHT on full X ===
        full_mask = np.ones(nwin)
        keep_idx = np.argwhere(full_mask).squeeze()
        full_chunk = chunk[:, keep_idx]
        win = np.hanning(chunk.shape[1]).reshape(1, -1)
        full_win = win[:, keep_idx]
        psi = partial_fourier(nwin, keep_idx)
        rep_psi = np.tile(psi, (complex_cir.shape[0], 1, 1))
        full_spectrum = iht(
            rep_psi, full_chunk * full_win, fixed_iters=False, n_iters=0
        )

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

        chunks.append(chunk[topn_idx, :])

        full_spec.append(full_spectrum[topn_idx, :].squeeze())

        mD_shift = process_cpx_crop(full_spectrum[topn_idx, :].squeeze())
        mD = min_max_freq(mD_shift[np.newaxis, :])
        mD_columns.append(mD.squeeze())

    chunks = np.array(chunks)
    full_spec = np.array(full_spec)
    mD_columns = np.array(mD_columns)

    return chunks, full_spec, mD_columns


def partial_fourier(N, idxs):
    F = np.conj(dft(N, scale="sqrtn"))
    F_part = F[idxs]  # * np.sqrt(N / len(self.keep_idxs))
    return F_part.squeeze()


def hard_thresholding(vector, sparsity_level):
    tozero = np.argpartition(np.abs(vector), -sparsity_level, axis=1)[
        :, : vector.shape[1] - sparsity_level, :
    ]
    rows = np.repeat(np.arange(len(tozero)), tozero.shape[1])
    cols = tozero.reshape(-1)
    vector[rows, cols] = complex(0, 0)
    return vector


def iht(psi, y, fixed_iters, n_iters, s=5, mu=1, maxit=300, change_conv=1e-2):
    it = 0
    end = False
    N = psi.shape[2]
    z_old = np.zeros((y.shape[0], N, 1), dtype=np.complex128)
    z = np.zeros((y.shape[0], N, 1), dtype=np.complex128)
    y = y[..., np.newaxis]
    residuals = []
    while not end:
        it += 1
        z += mu * np.transpose(np.conj(psi), (0, 2, 1)) @ (y - psi @ z)
        z = hard_thresholding(z, sparsity_level=s)
        change = np.linalg.norm(z - z_old, axis=1)
        z_old = np.copy(z)
        end_maxit = it >= maxit
        residuals.append(np.linalg.norm(psi @ z - y, axis=1).max())
        converge_thr = change_conv * np.linalg.norm(z_old, axis=1)
        # end_converged_change = change < change_conv * np.linalg.norm(z_old)
        end_conv_change = np.all(change < converge_thr)
        if fixed_iters:
            end = it == n_iters
        else:
            end = end_maxit or end_conv_change
    # print(end_maxit, it)
    # plt.plot(residuals)
    # plt.show()
    return z


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    P = partial_fourier(64, np.arange(64))
    P_real = complex_to_real_matrix(P)
    print(P_real.shape)

    plt.imshow(P_real)
    plt.show()
