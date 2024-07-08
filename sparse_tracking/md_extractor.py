import time

import matplotlib
import matplotlib.patches as patches
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.patches import Ellipse

from .cjpdaf import NN_CJPDAF
from .cs_stft import *
from .explicit_eot import KalmanTracker
from .utils import *

# def md_extraction(sel_tracking_list, chunk, params):
#     for sl in sel_tracking_list:
#         # micro-Doppler
#         ds = compute_mdspec(sl, chunk, params)
#         if ds is not None:
#             sl.micro_doppler.append(ds)


def ra2distbp_index(ra, dists, bp_norm, bp_ang):
    distbp = np.zeros_like(ra)
    rng, ang = ra
    range_diff = np.abs(dists - rng)
    distbp[0] = np.argmin(range_diff)
    restr_bp_ang = np.copy(bp_ang)
    restr_bp_ang[:, :60] = np.inf
    restr_bp_ang[:, 180:] = np.inf
    ang_diff = np.abs(bp_ang - ang)
    ang_idx = np.argmin(ang_diff)
    distbp[1] = np.argmax(bp_norm[:, ang_idx])
    return distbp.astype(int)


def s_method(stft, L):
    cross_term = np.zeros_like(stft)
    for k in range(len(stft)):
        start = k - L if (k - L >= 0) else 0
        end = k + L + 1 if (k + L + 1 < len(stft)) else len(stft) - 1
        w = stft[start:end]
        # print(k - L, k + L + 1, w)
        w_conj_flip = np.flip(np.conj(w))
        cross_term[k] = 2 * np.real((w * w_conj_flip).sum())
    squared_term = np.abs(stft) ** 2
    sm = squared_term + cross_term
    return sm


def compute_mrspec(t, chunk, params, space_interval=5):
    distbp = t.current_dbp_index
    # save_h = chunk[distbp[1], :params['Nd']//2, distbp[0]]
    # for el in save_h:
    #     t.h_signal.append(el)
    if chunk.shape[1] == 64:
        spec = chunk[distbp[1], 32, :]
        plt.plot(spec)
        plt.show()
        spec = 20 * np.log10(np.abs(spec) + 1e-8)
        # spec = spec.sum(-1)
        # mD_shift = np.zeros_like(spec)
        # mD_shift[:len(spec)//2] = spec[len(spec)//2:]
        # mD_shift[len(spec)//2:] = spec[:len(spec)//2]
        # spec = mD_shift
        # spec = np.delete(spec, [31,32,33])
        spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
        t.micro_range.append(spec)


def compute_mdspec_sparse(time_index, t, chunk, params):
    space_interval = int(np.floor(params["Q"] // 2))
    sparsity = params["OMEGA"]
    mask = params["SPARSE_PATTERN"]
    if mask is not None:
        mask_idx = np.argwhere(mask).squeeze()
        tks = params["INTER_PCKT"] * mask_idx
        sparsity_profile = (
            np.ones(2 * space_interval + 1, dtype=int) * sparsity
        )  # np.array([8, 4, 2, 2, 1, 1, 1, 2, 2, 4, 8])
        # G = spectral_window(tks, params['INTER_PCKT'])

        distbp = t.current_dbp_index
        save_h = chunk[distbp[1], : params["Nd"] // 2, distbp[0]]
        for el in save_h:
            t.h_signal.append(el)
        chunk = chunk[
            distbp[1], :, distbp[0] - space_interval : distbp[0] + space_interval + 1
        ]
        win = np.hanning(len(chunk))
        # if chunk.shape[0] == ndoppler:
        if chunk.shape[1] > 0 and chunk.shape[0] == params["Nd"]:
            spectrum = []
            cond = np.sum(mask) >= 4  # at least 4 samples in the window
            if cond:
                for j, x in enumerate(chunk.T):
                    cs_solver = cs_stft(
                        x * win,
                        mask,
                        t.last_h,
                        solver=params["SOLVER"],
                        sparsity=sparsity_profile[j],
                    )
                    start = time.time()
                    sol = cs_solver.solve()
                    end = time.time()
                    spectrum.append(sol)
                    if j == chunk.shape[1] // 2:
                        t.last_h = sol
            else:
                sol = np.zeros(chunk.shape[0], dtype=np.complex128)
                spectrum.append(sol)

            spectrum = np.array(spectrum).T

            spec = 20 * np.log10(np.abs(spectrum) + 1e-8)
            spec = np.abs(spectrum)
            if not t.body_md:
                spec = spec.sum(-1)
                mD_shift = np.zeros_like(spec)
                mD_shift[: len(spec) // 2] = spec[len(spec) // 2 :]
                mD_shift[len(spec) // 2 :] = spec[: len(spec) // 2]
                spec = mD_shift
                # spec = np.delete(spec, [62, 63, 64, 65, 66, 67])
                spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
                t.micro_doppler.append(spec)
            else:
                mD_shift = np.zeros_like(spec)
                mD_shift[: len(spec) // 2] = spec[len(spec) // 2 :]
                mD_shift[len(spec) // 2 :] = spec[: len(spec) // 2]
                spec = mD_shift
                # spec = np.delete(spec, [31,32,33])
                spec = (spec - spec.min(axis=0)) / (
                    spec.max(axis=0) - spec.min(axis=0) + 1e-8
                )
                for x in range(spec.shape[-1]):
                    t.micro_doppler[x].append(spec[..., x])
    else:
        # nsparse = np.random.choice(np.arange(params['NWIN']))
        nsparse = params["SPARSITY"]
        mask_idx = np.random.choice(
            np.arange(params["NWIN"]), size=nsparse, replace=False
        )
        mask = np.zeros((params["NWIN"],), dtype=bool)
        mask[mask_idx] = True
        sparsity_profile = np.ones(2 * space_interval + 1, dtype=int) * sparsity

        distbp = t.current_dbp_index
        save_h = chunk[distbp[1], : params["Nd"] // 2, distbp[0]]
        for el in save_h:
            t.h_signal.append(el)
        chunk = chunk[
            distbp[1], :, distbp[0] - space_interval : distbp[0] + space_interval + 1
        ]
        win = np.hanning(len(chunk))

        if chunk.shape[1] > 0 and chunk.shape[0] == params["Nd"]:
            spectrum = []
            cond = nsparse > 1  # at least 2 samples in the window
            if cond:
                for j, x in enumerate(chunk.T):
                    cs_solver = cs_stft(
                        x * win,
                        mask,
                        t.last_h,
                        solver=params["SOLVER"],
                        sparsity=sparsity_profile[j],
                    )
                    start = time.time()
                    sol = cs_solver.solve()
                    end = time.time()
                    spectrum.append(sol)
                    if j == chunk.shape[1] // 2:
                        t.last_h = sol
            else:
                sol = np.zeros(chunk.shape[0], dtype=np.complex128)
                spectrum.append(sol)

            spectrum = np.array(spectrum).T

            spec = 20 * np.log10(np.abs(spectrum) + 1e-8)
            spec = np.abs(spectrum)

            spec = spec.sum(-1)
            mD_shift = np.zeros_like(spec)
            mD_shift[: len(spec) // 2] = spec[len(spec) // 2 :]
            mD_shift[len(spec) // 2 :] = spec[: len(spec) // 2]
            spec = mD_shift
            # spec = np.delete(spec, [62, 63, 64, 65, 66, 67])
            spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
            t.micro_doppler.append(spec)


def compute_mdspec_us(time_index, t, chunk, params, space_interval=4):
    nsparse = params["SPARSITY"]
    mask_idx = np.random.choice(np.arange(params["NWIN"]), size=nsparse, replace=False)
    mask = np.zeros((params["NWIN"],), dtype=bool)
    mask[mask_idx] = True
    # print(chunk.shape)
    cond = (np.sum(mask) > 4) and (
        chunk.shape[1] == params["NWIN"]
    )  # at least 5 samples in the window
    # print(np.sum(mask))
    distbp = t.current_dbp_index
    start = max(0, distbp[0] - space_interval)
    end = min(distbp[0] + space_interval, chunk.shape[-1])
    # print(cond)
    if cond:
        chunk = chunk[distbp[1], mask, start:end]
        win = np.hanning(len(chunk)).reshape(-1, 1)
        spectrum = np.fft.fft(chunk * win, n=params["Nd"], axis=0)
    else:
        spectrum = np.zeros((params["NWIN"], end - start), dtype=np.complex128)
    spec = 20 * np.log10(np.abs(spectrum) + 1e-8)
    spec = spec.sum(-1)
    mD_shift = np.zeros_like(spec)
    mD_shift[: len(spec) // 2] = spec[len(spec) // 2 :]
    mD_shift[len(spec) // 2 :] = spec[: len(spec) // 2]
    spec = mD_shift
    # spec = np.delete(spec, [31,32,33])
    spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
    t.micro_doppler.append(spec)
    # return spec
