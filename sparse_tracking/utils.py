import pickle
from os import times
from pathlib import Path

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spectrum
from numpy.core.numeric import indices
from scipy import signal
from scipy.io import loadmat
from scipy.signal import butter, freqz, lfilter
from sklearn.decomposition import PCA


def get_empty_dict(npoints=100, typ="random"):
    if typ == "random":
        return {
            "id": None,
            "cardinality": npoints,
            "elements": np.random.normal(size=(npoints, 2)),
            "z_coord": np.random.normal(size=(npoints,)),
            "dopplers": np.random.normal(size=(npoints,)),
            "powers": np.random.normal(size=(npoints,)),
            "center": None,
            "ellipse": None,
        }
    elif typ == "zeros":
        return {
            "id": None,
            "cardinality": npoints,
            "elements": np.zeros((npoints, 2)),
            "z_coord": np.zeros((npoints,)),
            "dopplers": np.zeros((npoints,)),
            "powers": np.zeros((npoints,)),
            "center": None,
            "ellipse": None,
        }


def normalize(data):
    maxs = np.max(data, axis=0)
    mins = np.min(data, axis=0)
    if (maxs - mins) != 0:
        norm_data = (data - mins) / (maxs - mins)
    else:
        norm_data = data
    return norm_data


def isNaN(num):
    return num != num


def polar2cart(dist, ang):
    return np.vstack([dist * np.sin(np.deg2rad(ang)), dist * np.cos(np.deg2rad(ang))]).T


def cart2polar(x, y):
    return np.hstack([np.sqrt(x**2 + y**2), 90 - np.rad2deg(np.arctan2(y, x))])


def is_pos_def(A):
    M = np.matrix(A)
    return np.all(np.linalg.eigvals(M + M.transpose()) > 1e-10)


def get_moment(points, p, q):
    M = np.sum((points[:, 0] ** p) * (points[:, 1] ** q))
    return M


def get_ellipse(points, get_cov=True):
    cov = np.cov(points[:, :2].T) if get_cov else None

    M00 = get_moment(points, 0, 0)
    # print('M00 ', M00)
    x_bar = get_moment(points, 1, 0) / M00
    # print('x bar ', x_bar)
    y_bar = get_moment(points, 0, 1) / M00
    # print('y bar ', y_bar)

    mu20 = get_moment(points, 2, 0) / M00 - x_bar**2
    # print('mu20 ', mu20)
    mu11 = get_moment(points, 1, 1) / M00 - x_bar * y_bar
    # print('mu11 ', mu11)
    mu02 = get_moment(points, 0, 2) / M00 - y_bar**2
    # print('mu02 ', mu02)

    # print('operands ', mu20 + mu02, np.sqrt(2*mu11**2 + (mu20 - mu02)**2))

    l = np.sqrt(8 * (mu20 + mu02 + np.sqrt(2 * mu11**2 + (mu20 - mu02) ** 2)))
    # print('l ', l)
    w = np.sqrt(8 * (mu20 + mu02 - np.sqrt(2 * mu11**2 + (mu20 - mu02) ** 2)))
    # print('w', w)
    theta = 0.5 * np.arctan2(2 * mu11, (mu20 - mu02))

    return np.asarray([l, w, theta]), cov


def get_uncertainty(track):
    P = track.P[:2, :2]

    l = np.sqrt(
        8 * (P[0, 0] + P[1, 1] + np.sqrt(2 * P[0, 1] ** 2 + (P[0, 0] - P[1, 1]) ** 2))
    )
    w = np.sqrt(
        8 * (P[0, 0] + P[1, 1] - np.sqrt(2 * P[0, 1] ** 2 + (P[0, 0] - P[1, 1]) ** 2))
    )
    theta = 0.5 * np.arctan2(2 * P[0, 1], (P[0, 0] - P[1, 1]))

    return np.asarray([l, w, theta])


def get_obs(cluster):
    obs = np.concatenate([cluster["center"], cluster["ellipse"]]).reshape(-1, 1)
    return obs


def deg2rad_shift(angles):
    a = np.copy(angles)
    a = np.pi * a / 180
    a = -a + np.pi / 2
    return a


def shift_rad2deg(angles):
    a = np.copy(angles)
    # a = -a + np.pi/2
    a = 180 * a / np.pi
    return a


def entropy(pdf):
    return -(pdf * np.log2(pdf)).sum()


def interpolate_missing(cir, idx):
    nafter = 5
    new_idx = np.copy(idx)
    for i in range(len(idx) - nafter):
        if idx[i] == 1 and idx[i + 1] == 0:
            new_idx[i + 1 : i + nafter] = 1

    real = np.real(cir)
    # fig, ax = plt.subplots(3, 1)
    # ax[0].plot(real[50, 14600:15300])
    imag = np.imag(cir)
    real[:, new_idx == 1] = np.nan
    # ax[1].plot(real[50, 14600:15300])
    imag[:, new_idx == 1] = np.nan
    for rbin in range(real.shape[0]):
        interp = np.array(pd.DataFrame(real[rbin, :]).interpolate(method="cubic"))
        real[rbin] = interp.squeeze()
        interp = np.array(pd.DataFrame(imag[rbin, :]).interpolate(method="cubic"))
        imag[rbin] = interp.squeeze()
    cir = real + 1j * imag
    # ax[2].plot(np.real(cir[50, 14600:15300]))
    # plt.show()
    return cir


def refine_peaks(x, cand_pos, win=2):
    true_pos = []
    for pos in cand_pos:
        x_mod = -1e-5 * np.ones_like(x)
        x_mod[pos - win : pos + win] = x[pos - win : pos + win]
        newpos = np.argmax(x_mod)
        true_pos.append(newpos)
    return np.array(true_pos)


def filter_spectrogram(spec):
    s = np.copy(spec)

    idx = np.where(s.mean(1) > 750)[0]
    s[idx, :] = np.nan

    for dbin in range(spec.shape[1]):
        interp = np.array(pd.DataFrame(s[:, dbin]).interpolate(method="linear"))
        s[:, dbin] = interp.squeeze()
    return s


# def mD_spectrum(complex_cir, nwin, step, ndoppler, mode="power_dB"):
#     spec = []
#     for i in range(0, complex_cir.shape[1], step):
#         chunk = complex_cir[:, i : i + nwin]
#         win = np.hamming(chunk.shape[1]).reshape(1, -1)
#         spectrum = np.fft.fft(chunk * win, n=ndoppler, axis=1)
#         spec.append(spectrum)

#     spec = np.array(spec)

#     if mode == "power_dB":
#         spec = 20 * np.log10(np.abs(spec))
#         return spec
#     elif mode == "power_lin":
#         spec = np.abs(spec) ** 2
#         return spec
#     elif mode == "complex":
#         return spec


def angle_estimation(cir_path, BP_norm, BP_angles):
    # RSS-based angle estimation algorithm
    # Normalized RSS input
    p_m = cir_path**2
    norm_p_m = np.linalg.norm(p_m)
    p_m_N = p_m / norm_p_m

    npaths = p_m.shape[1]

    corr_pm_BP = np.zeros((BP_norm.shape[1], npaths))
    for j in range(npaths):
        for i in range(BP_norm.shape[1]):
            corr_pm_BP[i, j] = np.sum(p_m_N[:, j] * BP_norm[:, i])

    # Restrict searching only in the range[-60:60] index(61 to 301)
    corr_pm_BP[:60, :] = -np.inf
    corr_pm_BP[180:, :] = -np.inf
    max_CORR_pos = np.argmax(corr_pm_BP, axis=0)

    # BPang_rep = np.
    angle_est = BP_angles[0, max_CORR_pos]

    return angle_est


def setup_angles(angle_path, angle_offset, bp_used=64):
    angle_data = loadmat(angle_path)
    ANGLES_REAL, ANGLES_ROT, MAG_ANGLE = (
        angle_data["ANGLES_REAL"],
        angle_data["ANGLES_ROT"],
        angle_data["MAG_ANGLE"],
    )

    BP_used = np.arange(bp_used)
    BP_angles = -(ANGLES_REAL + angle_offset)

    # Index of TX BP used for measurements
    BP_codebook_used = np.arange(0, 64, 64 // bp_used)

    # Power of the selected BPs
    BP_m = MAG_ANGLE[BP_codebook_used, :] ** 2

    # Normalize the BPs per angle value
    norm_BP = np.linalg.norm(BP_m, axis=0)
    BP_norm = BP_m / norm_BP
    BP_norm = BP_norm[BP_used, :]
    return BP_norm, BP_angles


def get_cir(datapath, refpath, dist_ini, dist_end, return_ref=True):

    f = h5py.File(datapath, "r")
    if return_ref:
        g = h5py.File(refpath, "r")

    complex_cir_data = f["FRAMES"]["CIR"]
    if return_ref:
        complex_ref_data = g["FRAMES"]["CIR"]
    # cfo = f['FRAMES']['CFO_EST']
    # time = f['FRAMES']['TIME']
    bad_pckts = f["FRAMES"]["BAD_PACKET"]
    # print(complex_cir_data)
    cir_data = complex_cir_data[dist_ini:dist_end, :, :].view("complex")
    del complex_cir_data
    if return_ref:
        ref_data = complex_ref_data[dist_ini:dist_end, :, :].view("complex")
        del complex_ref_data
        diff = np.abs(cir_data)  # - np.mean(np.abs(ref_data), axis=1, keepdims=True)
        empty_diff = np.abs(ref_data) - np.mean(np.abs(ref_data), axis=1, keepdims=True)

        diff[diff < 0] = 0
        empty_diff[empty_diff < 0] = 0
        # return cir_data, ref_data, diff, empty_diff, time.value.squeeze(), bad_pckts.value.squeeze().view(np.float)
        return (
            cir_data,
            ref_data,
            diff,
            empty_diff,
            None,
            bad_pckts.value.squeeze().view(np.float),
        )
    else:
        return cir_data, bad_pckts.value.squeeze().view(np.float)


def polar2cart(dist, ang):
    return np.vstack([dist * np.sin(np.deg2rad(ang)), dist * np.cos(np.deg2rad(ang))]).T


def cart2polar(x, y):
    return np.hstack([np.sqrt(x**2 + y**2), 90 - np.rad2deg(np.arctan2(y, x))])


def min_max_normalize(x, axis=-1, keepdims=True):
    x = (x - np.min(x, axis=axis, keepdims=keepdims)) / (
        np.max(x, axis=axis, keepdims=keepdims)
        - np.min(x, axis=axis, keepdims=keepdims)
        + 1e-6
    )
    return x


def crop_doppler(x, bins):
    x = np.delete(x, bins, axis=1)
    return x


def slotted_resampling(samples, times, grid_step=0.27e-3):
    grid_length = int(times[-1] / grid_step)
    grid = np.arange(grid_length) * grid_step
    bins = np.array(
        [0]
        + [(i + 0.5) * grid_step for i in range(grid_length)]
        + [grid_length * grid_step]
    )
    dig = np.digitize(times, bins) - 1
    reg_samples = np.zeros(
        (samples.shape[0], len(grid), samples.shape[2]), dtype=complex
    )
    pattern = np.zeros(len(grid), dtype=bool)
    sidxs = []
    for j in range(len(grid)):
        current_time = grid[j]
        idx = np.array(dig == j)
        sel_indices = np.argwhere(idx)
        if idx.any():
            times_into_bin = times[idx]
            samples_into_bin = samples[:, idx, :]
            diff = np.abs(current_time - times_into_bin)
            closest_idx = np.argmin(diff)
            reg_samples[:, j, :] = samples_into_bin[:, closest_idx, :]
            pattern[j] = True
            sidxs.append(sel_indices[closest_idx])
        else:
            pass
    return reg_samples, pattern, np.array(sidxs).squeeze()
