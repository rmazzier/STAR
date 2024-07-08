import numpy as np
import os
from scipy.io import loadmat
import csv
import yaml
import torch

import hyperparams as hpm


def get_cir(datapath, dist_bounds=None):
    """
    Function that loads a CIR .mat file into a numpy array

    Parameters
    ----------
    datapath: str or datapath
        The path to the CIR file to be loaded

    dist_bounds: tuple or list or numpy array
        Initial and final indices indicating the start and end range bins to select

    Returns
    ----------
    cir_data: numpy array
        CIR data with shape (n_range_bins, n_packets, n_bp)

    """
    f = loadmat(datapath)
    complex_cir_data = f["FRAMES"]["CIR"][0][0]
    if dist_bounds:
        cir_data = complex_cir_data[:, :, dist_bounds[0]: dist_bounds[1]]
    else:
        cir_data = complex_cir_data
    cir_data = np.transpose(cir_data, (2, 1, 0))
    return cir_data


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


def real_to_complex_vector(real_vec):
    """Convert a real-valued 2K-dimensional vector h_r into complex-valued K-dimensional vector h_c, s.t.:
    h_c = h_r[:K] - j * h_r[K:]

    Input: Flattened, 1-dimensional real-valued array of shape (2n,);
    Output: Flattened, 1-dimensional complex-calued array of shape (n).
    """
    d = real_vec.shape[-1] // 2
    complex_vec = real_vec[..., :d] - 1j * real_vec[..., d:]
    return complex_vec


def process_cpx_crop(complex_crop):
    """Input: single complex crop of shape (110, NWIN)"""
    # 3.1) Take np.abs(crop) ** 2
    p = np.abs(complex_crop) ** 2

    # 3.2) Sum along range axis
    mD = p.sum(0)
    # 3.3) Make mD Shift
    mD_shift = np.roll(mD, mD.shape[0] // 2)
    return mD_shift


# def get_act_filenames(act_name):
#     all_filenames = os.listdir(hpm.DATA_PATH)
#     act_filenames = [f for f in all_filenames if act_name + "_" in f]
#     sorted_filenames = sorted(act_filenames, key=lambda s: int(s.split("_")[4][:-4]))
#     return sorted_filenames


# Following methods based on this naming Convention:
# {PASS_INDEX}_{SUBJECT}_{ACTIVITY}_{ACTIVITY_INDEX}_{CHUNK_INDEX}
def get_subj_from_filename(filename):
    return filename.split("_")[1]


def get_act_from_filename(filename):
    return filename.split("_")[2]


def get_actidx_from_filename(filename):
    return filename.split("_")[3][:-4]


def get_chunkidx_from_filename(filename):
    return filename.split("_")[4]


def parse_grid_search_csv(path):

    config_dict = {}
    keys = []
    with open(path) as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                keys = row

                for key in keys:
                    config_dict[key] = []
            else:
                for j, el in enumerate(row):
                    if keys[j] in ["EPOCHS", "N_LIHT_ITERS", "N_PAST_WINDOWS"]:
                        config_dict[keys[j]].append(int(el))
                    elif keys[j] in ["LOSS_WEIGHTS"]:
                        config_dict[keys[j]].append(
                            [float(x) for x in el[1:-1].split(",")]
                        )
                    elif keys[j] in ["SUBJECTS"]:
                        config_dict[keys[j]].append(
                            [int(x) for x in el[1:-1].split(",")]
                        )
                    elif keys[j] in ["ACTIVITIES"]:
                        config_dict[keys[j]].append(
                            [x for x in el[1:-1].split(",")])

                    elif keys[j] in ["MODEL_NAME", "MODEL_TYPE"]:
                        config_dict[keys[j]].append(el)
                    elif keys[j] in ["L_IHT_WEIGHT", "L_MD_WEIGHT", "W_D_REG_WEIGHT"]:
                        config_dict[keys[j]].append(float(el))
                    else:
                        if el == "TRUE":
                            config_dict[keys[j]].append(True)
                        elif el == "FALSE":
                            config_dict[keys[j]].append(False)

    return config_dict


def load_config_yaml(path):
    "Reads wandb config files and returns a dict with the values"
    with open(path, "r") as f:
        current_config = yaml.load(f, Loader=yaml.FullLoader)
    cfg = {}

    for k, v in current_config.items():
        if k != "wandb_version":
            cfg[k] = v["value"]
    return cfg


def print_model_params(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Param {name}: shape {param.data.size()}")

    # return number of parameters
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def IHT_to_mD(IHT_out_pred):
    """Take in input predictions from IHT of shape [n_windows, n_bins, W]"""
    cpx_IHT_out = IHT_out_pred.reshape(
        IHT_out_pred.shape[0], 2, IHT_out_pred.shape[1] // 2
    ).clone()
    cpx_IHT_out[:, 1, :] *= -1
    p = torch.norm(cpx_IHT_out, dim=1) ** 2
    mD = p.sum(0)
    mD_column_pred = (mD - mD.min()) / (mD.max() - mD.min() + 1e-8)
    return mD_column_pred


def crop(s, length, step):
    idxs = np.arange(len(s) - length, step=step)
    batchseq = np.zeros((len(idxs), length, s.shape[-1]), dtype=np.float32)
    for b in range(len(idxs)):
        batchseq[b] = s[idxs[b]: idxs[b] + length]
    return batchseq


if __name__ == "__main__":
    config = parse_grid_search_csv("data/final_grid_search.csv")
    pass
