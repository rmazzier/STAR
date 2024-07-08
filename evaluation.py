import torch
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import json
from ignite.metrics import SSIM, PSNR
from tqdm import tqdm

from sparse_dataset_gen import Sparse_MD_Dataset, Dataset_Rev1
from models import STAR
from utils import real_to_complex_vector, process_cpx_crop, get_cir, IHT_to_mD
from md_extraction.utils_jp import min_max_freq
from md_extraction.sparsity_based import (
    partial_fourier,
    iht,
    partial_fourier,
)

from md_extraction.fft_based import (
    compute_mD,
)

from utils import (
    get_subj_from_filename,
    get_act_from_filename,
    get_actidx_from_filename,
    load_config_yaml,
    complex_to_real_matrix,
)

import hyperparams as hpm


def load_model_and_config(model_dir):
    current_config = load_config_yaml(os.path.join(model_dir, "config.yaml"))

    mweights_fname = os.listdir(os.path.join(
        model_dir, "models", "STAR_GS"))[0]
    model_weights = torch.load(
        os.path.join(model_dir, "models", "STAR_GS", mweights_fname)
    )

    return current_config, model_weights


def MSE_noise(rec_spectrum, full_spectrum, thr=0.3):
    """Evaluation metric to see how well the reconstructed spectrum matches the full IHT
    spectrum, just on the background of the mD spectrum."""
    shape_mask = np.copy(full_spectrum)
    shape_mask[shape_mask < thr] = 0.0
    shape_mask[shape_mask >= thr] = 1.0

    # compute the MSE between reconstructed and full IHT only on the mask
    MSE_n = torch.mean(
        (rec_spectrum[shape_mask == 0] - full_spectrum[shape_mask == 0]) ** 2
    )

    return MSE_n.item()


def MSE_shape(rec_spectrum, full_spectrum, thr=0.3):
    """Evaluation metric to see how well the reconstructed spectrum matches the full IHT
    spectrum, just on the shape of interest of the mD spectrum."""
    shape_mask = np.copy(full_spectrum)
    shape_mask[shape_mask < thr] = 0.0
    shape_mask[shape_mask >= thr] = 1.0

    # compute the MSE between reconstructed and full IHT only on the mask
    MSE_s = torch.mean(
        (rec_spectrum[shape_mask == 1] - full_spectrum[shape_mask == 1]) ** 2
    )

    return MSE_s.item()


def compute_metrics(mD, IHT_full_mD):
    """Returns:
    mD_SSIM, mD_PSNR, shape_MSE, noise_MSE, MSE, MAE"""
    ssim = SSIM(data_range=1.0)
    psnr = PSNR(data_range=1.0)
    MSELoss = torch.nn.MSELoss()
    L1Loss = torch.nn.L1Loss()

    mD = torch.from_numpy(mD).float()
    IHT_full_mD = torch.from_numpy(IHT_full_mD).float()

    ssim.update((mD.unsqueeze(0).unsqueeze(
        0), IHT_full_mD.unsqueeze(0).unsqueeze(0)))
    psnr.update((mD.unsqueeze(0).unsqueeze(
        0), IHT_full_mD.unsqueeze(0).unsqueeze(0)))
    mD_SSIM = ssim.compute()
    mD_PSNR = psnr.compute()
    shape_MSE = MSE_shape(mD, IHT_full_mD)
    noise_MSE = MSE_noise(mD, IHT_full_mD)
    MSE = MSELoss(mD, IHT_full_mD).item()
    MAE = L1Loss(mD, IHT_full_mD).item()

    ssim.reset()
    psnr.reset()

    return mD_SSIM, mD_PSNR, shape_MSE, noise_MSE, MSE, MAE


def evaluate_sweep(sweep_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # loop over models downloaded from wandb
    models_dirs = os.listdir(sweep_dir)
    for model_dir in models_dirs:
        # Load config and model weights
        evaluate_run(
            os.path.join(sweep_dir, model_dir),
            out_dir=os.path.join(out_dir, model_dir),
            ps_remove=[0.0, 0.5, 0.75, 0.9],
        )


def evaluate_run(LIHT_model, model_config, out_dir, ps_remove, plot_results=True):
    def update_results_dictionary(
        out_dict,
        p_remove,
        LIHT_SSIMs,
        LIHT_PSNRs,
        LIHT_MSEs,
        LIHT_MSEs_shape,
        LIHT_MSEs_noise,
        LIHT_MAEs,
        IHT_SSIMs,
        IHT_PSNRs,
        IHT_MSEs,
        IHT_MSEs_shape,
        IHT_MSEs_noise,
        IHT_MAEs,
        IHT_SSIMs_conv,
        IHT_PSNRs_conv,
        IHT_MSEs_conv,
        IHT_MSEs_shape_conv,
        IHT_MSEs_noise_conv,
        IHT_MAEs_conv,
    ):
        def update_key_metric(results_dict, p_remove, key_model, key_metric, metric):
            results_dict[p_remove][key_model][key_metric].setdefault(
                "values", [])
            results_dict[p_remove][key_model][key_metric].setdefault(
                "mean", [])
            results_dict[p_remove][key_model][key_metric].setdefault("std", [])

            results_dict[p_remove][key_model][key_metric]["values"] = metric
            results_dict[p_remove][key_model][key_metric]["mean"] = np.mean(
                metric)
            results_dict[p_remove][key_model][key_metric]["std"] = np.std(
                metric)

        # === LIHT ===
        update_key_metric(out_dict, p_remove, "LIHT", "MSE", LIHT_MSEs)
        update_key_metric(out_dict, p_remove, "LIHT", "MAE", LIHT_MAEs)
        update_key_metric(out_dict, p_remove, "LIHT", "SSIM", LIHT_SSIMs)
        update_key_metric(out_dict, p_remove, "LIHT", "PSNR", LIHT_PSNRs)
        update_key_metric(out_dict, p_remove, "LIHT",
                          "MSE_shape", LIHT_MSEs_shape)
        update_key_metric(out_dict, p_remove, "LIHT",
                          "MSE_noise", LIHT_MSEs_noise)

        # === IHT converged ===
        update_key_metric(out_dict, p_remove,
                          "IHT_converged", "MSE", IHT_MSEs_conv)
        update_key_metric(out_dict, p_remove,
                          "IHT_converged", "MAE", IHT_MAEs_conv)
        update_key_metric(out_dict, p_remove, "IHT_converged",
                          "SSIM", IHT_SSIMs_conv)
        update_key_metric(out_dict, p_remove, "IHT_converged",
                          "PSNR", IHT_PSNRs_conv)
        update_key_metric(
            out_dict,
            p_remove,
            "IHT_converged",
            "MSE_shape",
            IHT_MSEs_shape_conv,
        )
        update_key_metric(
            out_dict,
            p_remove,
            "IHT_converged",
            "MSE_noise",
            IHT_MSEs_noise_conv,
        )

        # === IHT fixed iters ===
        update_key_metric(out_dict, p_remove, "IHT_fixed", "MSE", IHT_MSEs)
        update_key_metric(out_dict, p_remove, "IHT_fixed", "MAE", IHT_MAEs)
        update_key_metric(out_dict, p_remove, "IHT_fixed", "SSIM", IHT_SSIMs)
        update_key_metric(out_dict, p_remove, "IHT_fixed", "PSNR", IHT_PSNRs)
        update_key_metric(out_dict, p_remove, "IHT_fixed",
                          "MSE_shape", IHT_MSEs_shape)
        update_key_metric(out_dict, p_remove, "IHT_fixed",
                          "MSE_noise", IHT_MSEs_noise)

    os.makedirs(out_dir, exist_ok=True)

    # Define test dataset (same for all models)
    _, _, test_filenames = Sparse_MD_Dataset.make_splits(
        model_config["SUBJECTS"],
        model_config["ACTIVITIES"],
        subsample_factor=model_config["DATASET_SUBSAMPLE_FACTOR"],
        seed=model_config["DATASET_SPLIT_SEED"],
        train=model_config["SPLIT_PTGS"][0],
        valid=model_config["SPLIT_PTGS"][1],
        test=model_config["SPLIT_PTGS"][2],
    )

    random_test_set = Sparse_MD_Dataset(test_filenames, p_burst=0.0)

    out_dict = {}
    for p_remove in ps_remove:
        print(f"p_remove: {p_remove}")
        out_dict.setdefault(p_remove, {})
        out_dict[p_remove].setdefault("LIHT", {})
        out_dict[p_remove].setdefault("IHT_fixed", {})
        out_dict[p_remove].setdefault("IHT_converged", {})

        for _metric in ["SSIM", "PSNR", "MSE", "MSE_shape", "MSE_noise", "MAE"]:
            out_dict[p_remove]["LIHT"].setdefault(_metric, {})
            out_dict[p_remove]["IHT_fixed"].setdefault(_metric, {})
            out_dict[p_remove]["IHT_converged"].setdefault(_metric, {})

        with torch.no_grad():
            # Initialize lists to store metrics,
            # for LIHT and IHT both converged and fixed iterations
            LIHT_SSIMs = []
            LIHT_PSNRs = []
            LIHT_MSEs = []
            LIHT_MSEs_shape = []
            LIHT_MSEs_noise = []
            LIHT_MAEs = []

            IHT_SSIMs = []
            IHT_PSNRs = []
            IHT_MSEs = []
            IHT_MSEs_shape = []
            IHT_MSEs_noise = []
            IHT_MAEs = []

            IHT_SSIMs_conv = []
            IHT_PSNRs_conv = []
            IHT_MSEs_conv = []
            IHT_MSEs_shape_conv = []
            IHT_MSEs_noise_conv = []
            IHT_MAEs_conv = []

            for idx in tqdm(range(len(random_test_set))):
                X_test = random_test_set[idx][0].to(
                    hpm.DEVICE).float().squeeze()
                mD_columns_test = (
                    random_test_set[idx][2].to(hpm.DEVICE).float().squeeze()
                )
                # IHT_output_test = (
                #     random_test_set[idx][1].to(hpm.DEVICE).float().squeeze()
                # )

                past_wins_IHT_t = []
                reconstructions = []
                iht_spectrums_conv = []
                iht_spectrums_fixed = []
                gts_full_iht = []

                for j in range(X_test.shape[0]):
                    chunk = X_test[j]
                    mD_column_gt_test = mD_columns_test[j]
                    # IHT_out_gt_test = IHT_output_test[j]

                    chunk_mask = random_test_set.generate_mask(
                        chunk.shape[1] // 2, p_remove=p_remove
                    )

                    # apply mask on chunk
                    masked_chunk = chunk * chunk_mask.unsqueeze(0)

                    # update past windows
                    if j > 0:
                        past_wins_IHT_t.append(IHT_pred_test.detach())
                        if len(past_wins_IHT_t) > model_config["N_PAST_WINDOWS"]:
                            past_wins_IHT_t.pop(0)
                        past_wins_t = torch.stack(past_wins_IHT_t, dim=0)
                    else:
                        past_wins_t = []

                    mD_column_pred_test, IHT_pred_test = LIHT_model(
                        masked_chunk, past_wins_t
                    )
                    # mD_column_pred_test = IHT_to_mD(IHT_pred_test)

                    reconstructions.append(mD_column_pred_test.cpu())
                    gts_full_iht.append(mD_column_gt_test.cpu())

                    # === now run IHT on the masked chunk ===
                    complex_chunk = real_to_complex_vector(
                        masked_chunk).cpu().numpy()
                    chunk_mask = chunk_mask[: chunk_mask.shape[0] //
                                            2].cpu().numpy()

                    keep_idx = np.argwhere(chunk_mask).squeeze()
                    partial_chunk = complex_chunk[:, keep_idx]
                    win = np.hanning(complex_chunk.shape[1]).reshape(1, -1)
                    partial_win = win[:, keep_idx]
                    psi = partial_fourier(hpm.DATAGEN_PARAMS["NWIN"], keep_idx)
                    rep_psi = np.tile(psi, (complex_chunk.shape[0], 1, 1))

                    spectrum_conv = iht(
                        rep_psi,
                        partial_chunk * partial_win,
                        fixed_iters=False,
                        n_iters=model_config["N_LIHT_ITERS"],
                    )
                    spectrum_fixed = iht(
                        rep_psi,
                        partial_chunk * partial_win,
                        fixed_iters=True,
                        n_iters=model_config["N_LIHT_ITERS"],
                    )

                    iht_spectrums_fixed.append(spectrum_fixed)
                    iht_spectrums_conv.append(spectrum_conv)

                    # Compute the various mD Spectrums of whole sequence

                # process converged iht spectrum
                IHT_conv_mD = []
                for rec_crop in iht_spectrums_conv:
                    mD_shift = process_cpx_crop(rec_crop)
                    mD = min_max_freq(mD_shift[np.newaxis, :])
                    IHT_conv_mD.append(mD.squeeze())

                # process fixed iters iht spectrum
                IHT_fixed_mD = []
                for rec_crop in iht_spectrums_fixed:
                    mD_shift = process_cpx_crop(rec_crop)
                    mD = min_max_freq(mD_shift[np.newaxis, :])
                    IHT_fixed_mD.append(mD.squeeze())

                IHT_conv_mD = np.stack(IHT_conv_mD, 0)
                IHT_fixed_mD = np.stack(IHT_fixed_mD, 0)
                IHT_full_mD = np.stack(gts_full_iht, 0)

                LIHT_mD = np.stack(reconstructions, 0)
                LIHT_mD = np.roll(LIHT_mD, 32, axis=-1)

                # === Compute the metrics ===

                # On LIHT
                (
                    mD_SSIM,
                    mD_PSNR,
                    shape_MSE,
                    noise_MSE,
                    MSE,
                    MAE,
                ) = compute_metrics(LIHT_mD, IHT_full_mD)

                LIHT_MSEs_shape.append(shape_MSE)
                LIHT_MSEs_noise.append(noise_MSE)
                LIHT_MSEs.append(MSE)
                LIHT_MAEs.append(MAE)
                LIHT_SSIMs.append(mD_SSIM)
                LIHT_PSNRs.append(mD_PSNR)

                # On IHT converged

                (
                    mD_SSIM_conv,
                    mD_PSNR_conv,
                    shape_MSE_conv,
                    noise_MSE_conv,
                    MSE_conv,
                    MAE_conv,
                ) = compute_metrics(IHT_conv_mD, IHT_full_mD)

                IHT_MSEs_shape_conv.append(shape_MSE_conv)
                IHT_MSEs_noise_conv.append(noise_MSE_conv)
                IHT_MSEs_conv.append(MSE_conv)
                IHT_MAEs_conv.append(MAE_conv)
                IHT_SSIMs_conv.append(mD_SSIM_conv)
                IHT_PSNRs_conv.append(mD_PSNR_conv)

                # On IHT fixed iters

                (
                    mD_SSIM_fixed,
                    mD_PSNR_fixed,
                    shape_MSE_fixed,
                    noise_MSE_fixed,
                    MSE_fixed,
                    MAE_fixed,
                ) = compute_metrics(IHT_fixed_mD, IHT_full_mD)

                IHT_MSEs_shape.append(shape_MSE_fixed)
                IHT_MSEs_noise.append(noise_MSE_fixed)
                IHT_MSEs.append(MSE_fixed)
                IHT_MAEs.append(MAE_fixed)
                IHT_SSIMs.append(mD_SSIM_fixed)
                IHT_PSNRs.append(mD_PSNR_fixed)

                # Update results dict

            update_results_dictionary(
                out_dict,
                p_remove,
                LIHT_SSIMs,
                LIHT_PSNRs,
                LIHT_MSEs,
                LIHT_MSEs_shape,
                LIHT_MSEs_noise,
                LIHT_MAEs,
                IHT_SSIMs,
                IHT_PSNRs,
                IHT_MSEs,
                IHT_MSEs_shape,
                IHT_MSEs_noise,
                IHT_MAEs,
                IHT_SSIMs_conv,
                IHT_PSNRs_conv,
                IHT_MSEs_conv,
                IHT_MSEs_shape_conv,
                IHT_MSEs_noise_conv,
                IHT_MAEs_conv,
            )

    # save out dict as json file in results folder
    with open(os.path.join(out_dir, "results.json"), "w") as fp:
        json.dump(out_dict, fp)

    if plot_results:

        plot_metrics_barplot(
            p_remove, out_dict, os.path.join(out_dir, "metrics_barplot.png")
        )

        plot_sparsity_vs_metric(
            os.path.join(out_dir, "psnrs.png"), ps_remove, "PSNR", out_dict
        )
        plot_sparsity_vs_metric(
            os.path.join(out_dir, "ssims.png"), ps_remove, "SSIM", out_dict
        )
        plot_sparsity_vs_metric(
            os.path.join(out_dir, "mses.png"), ps_remove, "MSE", out_dict
        )
        plot_sparsity_vs_metric(
            os.path.join(
                out_dir, "mses_shape.png"), ps_remove, "MSE_shape", out_dict
        )
        plot_sparsity_vs_metric(
            os.path.join(
                out_dir, "mses_noise.png"), ps_remove, "MSE_noise", out_dict
        )

    return out_dict


def plot_sparsity_vs_metric(out_dir, ps_remove, metric, results_dict):
    n_samples = len(results_dict[ps_remove[0]]["LIHT"][metric]["values"])

    LIHT_values = [results_dict[p]["LIHT"][metric]["mean"] for p in ps_remove]
    LIHT_stds = [
        results_dict[p]["LIHT"][metric]["std"] / np.sqrt(n_samples) for p in ps_remove
    ]

    IHT_values = [results_dict[p]["IHT_fixed"][metric]["mean"]
                  for p in ps_remove]
    IHT_stds = [
        results_dict[p]["IHT_fixed"][metric]["std"] / np.sqrt(n_samples)
        for p in ps_remove
    ]

    IHT_c_values = [results_dict[p]["IHT_converged"]
                    [metric]["mean"] for p in ps_remove]
    IHT_c_stds = [
        results_dict[p]["IHT_converged"][metric]["std"] / np.sqrt(n_samples)
        for p in ps_remove
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(alpha=0.5)
    ax.bar(
        np.arange(len(ps_remove)) - 0.2,
        LIHT_values,
        width=0.2,
        yerr=LIHT_stds,
        label="LIHT",
    )
    ax.bar(np.arange(len(ps_remove)), IHT_values,
           width=0.2, yerr=IHT_stds, label="IHT")
    ax.bar(
        np.arange(len(ps_remove)) + 0.2,
        IHT_c_values,
        width=0.2,
        yerr=IHT_c_stds,
        label="IHT-Conv",
    )
    ax.legend()
    ax.set_xticks(np.arange(len(ps_remove)))
    ax.set_xticklabels(ps_remove)
    ax.set_xlabel("Sparsity")
    ax.set_ylabel(metric)
    plt.savefig(os.path.join(out_dir))


def plot_metrics_barplot(p_remove, results_dict, out_path):
    metrics = ["PSNR", "SSIM", "MSE", "MAE", "MSE_shape", "MSE_noise"]
    models = ["LIHT", "IHT_fixed", "IHT_converged"]
    mean_values = np.array(
        [
            [results_dict[p_remove][m][metric]["mean"] for metric in metrics]
            for m in models
        ]
    )
    stds = np.array(
        [
            [results_dict[p_remove][m][metric]["std"] for metric in metrics]
            for m in models
        ]
    )

    stds[:, 0] = stds[:, 0] / mean_values[:, 0].max()
    mean_values[:, 0] = mean_values[:, 0] / mean_values[:, 0].max()

    fig, ax = plt.subplots(figsize=(10, 6))
    # plt.rc("text", usetex=True)
    # plt.rcParams["font.family"] = "serif"
    # plt.rcParams["font.size"] = 25

    ax.grid(alpha=0.5)
    ax.bar(
        np.arange(len(metrics)) - 0.2,
        mean_values[0],
        width=0.2,
        label=models[0],
        yerr=stds[0],
    )
    ax.bar(
        np.arange(len(metrics)),
        mean_values[1],
        width=0.2,
        label=models[1],
        yerr=stds[1],
    )
    ax.bar(
        np.arange(len(metrics)) + 0.2,
        mean_values[2],
        width=0.2,
        label=models[2],
        yerr=stds[2],
    )
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_title(f"Metrics for p_remove={p_remove}")
    plt.savefig(out_path)


def get_rec_mD(
    model: torch.nn.Module,
    test_set: torch.utils.data.Dataset,
    cfg,
    p_remove,
    fix_iht_iters,
    idx=0,
    omp_coeffs=5,
    compute_lasso=True,
):

    from baselines import omp, l1

    """Returns Reconstructed microdopplers with the following methods:
    1. STAR (ours)
    2. IHT
    3. IHT on full window
    4. OMP
    5. L1 minimization (LASSO)
    """

    reconstructions = []
    iht_spectrums = []
    omp_spectrums = []
    lasso_spectrums = []
    gts_full_iht = []

    W = 64
    W_d_cpx = partial_fourier(W, np.arange(W))
    W_d_real = complex_to_real_matrix(W_d_cpx)
    lasso_lambda = 5

    with torch.no_grad():
        X_test = test_set[idx][0].to(hpm.DEVICE).float().squeeze()
        IHT_output_test = test_set[idx][1].to(hpm.DEVICE).float().squeeze()
        mD_columns_test = test_set[idx][2].to(hpm.DEVICE).float().squeeze()

        W = X_test.shape[-1] // 2
        W_d_cpx = partial_fourier(W, np.arange(W))

        past_wins_IHT_t = []

        for j in range(X_test.shape[0]):
            chunk = X_test[j]
            mD_column_gt_test = mD_columns_test[j]
            # IHT_out_gt_test = IHT_output_test[j]

            chunk_mask = test_set.generate_mask(
                chunk.shape[1] // 2, p_remove=p_remove)

            # apply mask on chunk
            masked_chunk = chunk * chunk_mask.unsqueeze(0)

            # update past windows
            if j > 0:
                past_wins_IHT_t.append(IHT_pred_test.detach())
                if len(past_wins_IHT_t) > cfg["N_PAST_WINDOWS"]:
                    past_wins_IHT_t.pop(0)
                past_wins_t = torch.stack(past_wins_IHT_t, dim=0)
            else:
                past_wins_t = []

            mD_column_pred_test, IHT_pred_test = model(
                masked_chunk, past_wins_t)

            reconstructions.append(mD_column_pred_test.cpu())
            gts_full_iht.append(mD_column_gt_test.cpu())

            # now run IHT on the masked chunk
            print(f"Processing window {j} / {X_test.shape[0]} ", end="\r")
            complex_chunk = real_to_complex_vector(masked_chunk).cpu().numpy()
            chunk_mask = chunk_mask[: chunk_mask.shape[0] // 2].cpu().numpy()

            keep_idx = np.argwhere(chunk_mask).squeeze()
            partial_chunk = complex_chunk[:, keep_idx]
            win = np.hanning(complex_chunk.shape[1]).reshape(1, -1)
            partial_win = win[:, keep_idx]
            psi = partial_fourier(hpm.DATAGEN_PARAMS["NWIN"], keep_idx)
            rep_psi = np.tile(psi, (complex_chunk.shape[0], 1, 1))

            spectrum = iht(
                rep_psi,
                partial_chunk * partial_win,
                fix_iht_iters,
                n_iters=cfg["N_LIHT_ITERS"],
            )
            iht_spectrums.append(spectrum.squeeze())

            # Now run OMP on the masked chunk

            recs_bin = []
            for y in masked_chunk:
                y = y.cpu().numpy()
                y_cpx = real_to_complex_vector(y)
                sol = omp(
                    psi=W_d_cpx,
                    y=y_cpx,
                    ncoef=omp_coeffs,
                    maxit=30,
                    tol=0.05,
                    ztol=0.005,
                )

                recs_bin.append(sol)

            recs_bin = np.stack(recs_bin, 0)
            omp_spectrums.append(recs_bin)

            # Now run LASSO on the masked chunk
            if compute_lasso:

                recs_bin_l1 = []
                for y in masked_chunk:
                    sol = l1(psi=W_d_real, y=y.cpu().numpy(), lbd=lasso_lambda)
                    sol_cpx = real_to_complex_vector(sol)

                    recs_bin_l1.append(sol_cpx)

                recs_bin_l1 = np.stack(recs_bin_l1, 0)
                lasso_spectrums.append(recs_bin_l1)

    rec_spectrum = np.stack(reconstructions, 0)

    # shift rec_spectrum
    rec_spectrum = np.roll(rec_spectrum, 32, axis=-1)

    # process iht spectrum
    processed_IHT_mD = []
    for rec_crop in iht_spectrums:
        mD_shift = process_cpx_crop(rec_crop)
        mD = min_max_freq(mD_shift[np.newaxis, :])
        # mD = mD_shift
        processed_IHT_mD.append(mD.squeeze())

    IHT_mD = np.stack(processed_IHT_mD, 0)

    # process omp spectrum
    processed_OMP_mD = []
    for rec_crop in omp_spectrums:
        mD_shift = process_cpx_crop(rec_crop)
        mD = min_max_freq(mD_shift[np.newaxis, :])
        # mD = mD_shift
        processed_OMP_mD.append(mD.squeeze())

    OMP_mD = np.stack(processed_OMP_mD, 0)
    # reverse omp md in the columns
    OMP_mD = np.flip(OMP_mD, axis=1)

    # process lasso spectrum
    if compute_lasso:
        processed_LASSO_mD = []
        for rec_crop in lasso_spectrums:
            mD_shift = process_cpx_crop(rec_crop)
            mD = min_max_freq(mD_shift[np.newaxis, :])
            # mD = mD_shift
            processed_LASSO_mD.append(mD.squeeze())

        LASSO_mD = np.stack(processed_LASSO_mD, 0)
        # reverse lasso md in the columns
        LASSO_mD = np.flip(LASSO_mD, axis=1)
    else:
        LASSO_mD = None

    IHT_full_spectrum = np.stack(gts_full_iht, 0)
    # 6) Get the spectrum obtained with FFT from the full CIR

    # based on the class of the test set, we have to load the correct raw data
    if isinstance(test_set, Sparse_MD_Dataset):
        s_idx = get_subj_from_filename(test_set.raw_filenames[idx])
        a_name = get_act_from_filename(test_set.raw_filenames[idx])
        a_idx = get_actidx_from_filename(test_set.raw_filenames[idx])

        raw_fname = os.path.join(
            hpm.RAW_DATA_PATH,
            f"PERSON{s_idx}",
            f"{a_name}_{s_idx}_{a_idx}.mat",
        )
        cir = get_cir(raw_fname, hpm.DATAGEN_PARAMS["DIST_BOUNDS"])
    elif isinstance(test_set, Dataset_Rev1):
        a_idx = get_actidx_from_filename(test_set.filenames[idx])
        a_name = get_act_from_filename(test_set.filenames[idx])
        nseq = int(a_idx.split("-")[0])
        ntrack = int(a_idx.split("-")[1])
        act_idx = list(test_set.activity_dict.values()).index(a_name)

        raw_fname = os.path.join(
            hpm.RAW_REV1_DATA_PATH,
            f"TEST{act_idx+1}_{nseq}_US_F2_track_{ntrack}.obj",
        )

        with open(raw_fname, "rb") as file:
            raw_seq = pickle.load(file)

        # pad with zeros all elements of raw_seq to make all sequences have the shape (64, 8)
        for i in range(len(raw_seq)):
            rbins = raw_seq[i].shape[1]
            raw_seq[i] = np.pad(
                raw_seq[i], ((0, 0), (0, 8 - rbins)), "constant")

        raw_seq = np.concatenate(raw_seq, axis=0)
        cir = np.transpose(raw_seq, (1, 0))[:, :, np.newaxis]
    else:
        raise ValueError("Dataset not recognized")

    fft_spectrum, _ = compute_mD(cir, hpm.DATAGEN_PARAMS, normalize=True)

    return (
        rec_spectrum,
        IHT_full_spectrum,
        IHT_mD,
        OMP_mD,
        LASSO_mD,
        fft_spectrum,
    )


if __name__ == "__main__":
    from models import init_model
    from hyperparams import CONFIG as config

    # test_set = Dataset_Rev1("test", regenerate=False)
    _, _, test_filenames = Sparse_MD_Dataset.make_splits(
        config["SUBJECTS"],
        config["ACTIVITIES"],
        subsample_factor=config["DATASET_SUBSAMPLE_FACTOR"],
        seed=config["DATASET_SPLIT_SEED"],
        train=config["SPLIT_PTGS"][0],
        valid=config["SPLIT_PTGS"][1],
        test=config["SPLIT_PTGS"][2],
    )

    test_set = Sparse_MD_Dataset(test_filenames, p_burst=0.0)

    config_whole_path = os.path.join(
        "models", "STAR_GS", "STAR_final_config.pkl"
    )
    model_weights_path = os.path.join(
        "models", "STAR_GS", "STAR_final.pt"
    )

    with open(config_whole_path, "rb") as f:
        config = pickle.load(f)

    model_weights = torch.load(model_weights_path)

    model_final = init_model(config)
    model_final.load_state_dict(model_weights)
    model_final.to("cuda")
    model_final.eval()

    p_remove = 0.9

    i = 73
    (
        rec_spectrum,
        IHT_full_spectrum,
        IHT_spectrum,
        OMP_spectrum,
        LASSO_spectrum,
        fft_spectrum,
    ) = get_rec_mD(
        model=model_final,
        test_set=test_set,
        cfg=config,
        p_remove=p_remove,
        fix_iht_iters=False,
        idx=i,
    )

    quit()

    mD_SSIM, _, _, _, MSE, _ = compute_metrics(rec_spectrum, IHT_full_spectrum)
    SSIMs.append(mD_SSIM)
    RMSEs.append(np.sqrt(MSE))

    SSIMs = np.array(SSIMs)
    RMSEs = np.array(RMSEs)
    np.save(os.path.join("results", "Dataset_Rev1", "SSIMs_STAR.npy"), SSIMs)
    np.save(os.path.join("results", "Dataset_Rev1", "RMSEs_STAR.npy"), RMSEs)

    print(f"SSIMs: {SSIMs.mean()} +- {SSIMs.std() / np.sqrt(len(SSIMs))}")
    print(f"RMSEs: {RMSEs.mean()} +- {RMSEs.std() / np.sqrt(len(RMSEs))}")

    # Results on NEW ROOM:
    # SSIMs: 0.47908065619030477 +- 0.006226859367983648
    # RMSEs: 0.11455523849335202 +- 0.004336352384438636

    from models import init_model

    config_whole_path = os.path.join(
        "models", "STAR_GS", "STAR_final_config.pkl"
    )
    model_weights_path = os.path.join(
        "models", "STAR_GS", "STAR_final.pt"
    )

    with open(config_whole_path, "rb") as f:
        config = pickle.load(f)

    model_weights = torch.load(model_weights_path)

    _, _, test_filenames = Sparse_MD_Dataset.make_splits(
        config["SUBJECTS"],
        config["ACTIVITIES"],
        subsample_factor=config["DATASET_SUBSAMPLE_FACTOR"],
        seed=config["DATASET_SPLIT_SEED"],
        train=config["SPLIT_PTGS"][0],
        valid=config["SPLIT_PTGS"][1],
        test=config["SPLIT_PTGS"][2],
    )

    test_set = Sparse_MD_Dataset(test_filenames)

    model_final = init_model(config)
    model_final.load_state_dict(model_weights)
    model_final.to("cuda")
    model_final.eval()

    i = 73
    p_remove = 0.9
    rec_spectrum_90, IHT_full_spectrum, IHT_spectrum_90, fft = get_rec_mD(
        model=model_final,
        test_set=test_set,
        cfg=config,
        p_remove=p_remove,
        fix_iht_iters=False,
        idx=i,
    )
    pass
