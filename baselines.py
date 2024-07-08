"""
Evaluation of the baseline algorithms (i.e. Lasso and IHT) on the test set.

    1. Compute the MSE and SSIM on the test set. 
    Results are saved in a json dictionary in "results/baselines/results.json"

    2. Compute the CNN (already trained) predictions on the mds provided by the baselines.
    
    """

import time
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from scipy.linalg import dft
from matplotlib import pyplot as plt
import cvxpy as cp
import json

from sparse_dataset_gen import Sparse_MD_Dataset
from evaluation import compute_metrics
from models import init_model
from evaluation import get_rec_mD
from utils import process_cpx_crop, real_to_complex_vector, complex_to_real_matrix
from md_extraction.utils_jp import min_max_freq
from hyperparams import CONFIG as config
from md_extraction.sparsity_based import (
    partial_fourier,
)


def l1(psi, y, lbd=0.1):
    # Define and solve the CVXPY problem.
    n = psi.shape[1]
    x = cp.Variable(n)

    cost = cp.sum_squares(psi @ x - y) + lbd * cp.norm(x, 1)
    prob = cp.Problem(cp.Minimize(cost))
    prob.solve(solver=cp.OSQP)
    return x.value


def omp(psi, y, ncoef=5, maxit=30, tol=0.05, ztol=0.005):
    active = []
    coef = np.zeros(psi.shape[1], dtype=np.complex128)
    residual = y
    ynorm = np.linalg.norm(y)  # / np.sqrt(len(y))
    err = np.zeros(maxit, dtype=float)

    tol = tol * ynorm
    ztol = ztol * ynorm

    for it in range(maxit):
        rcov = np.abs(psi.T @ residual)
        i = np.argmax(rcov)

        if i not in active:
            active.append(i)

        selected_idxs = np.unique(active)
        coefi, _, _, _ = np.linalg.lstsq(psi[:, selected_idxs], y)
        coef[selected_idxs] = coefi

        residual = y - np.dot(psi[:, selected_idxs], coefi)
        err[it] = np.linalg.norm(residual) / (np.sqrt(len(residual)) * ynorm)

        if err[it] < tol:
            break
        if len(selected_idxs) >= ncoef:
            break
        if it == maxit - 1:
            break

    return coef


def batch_omp(psi, ys, ncoef=5, maxit=30, ztol=0.005):
    """
    Batched version of the OMP algorithm.
    Parameters
    ----------
    psi : torch tensor, shape (m, n)
        Dictionary. In our case is the inverse DFT matrix.
    ys : torch tensor, shape (n_bins, m)
        Batch of measurements.
    ncoef : int, optional
        Number of nonzero components of the solution. The default is 5.

    Returns
    -------
    coef : array, shape (n_bins, n)
        Sparse solution.

    """
    active = []
    psi = psi.unsqueeze(0)
    residuals = ys
    coef = torch.zeros((ys.shape[0], psi.shape[1]),
                       dtype=torch.complex128).to("cuda")
    ynorm = torch.linalg.norm(ys, axis=1)  # / np.sqrt(len(y))
    # err = np.zeros(maxit, dtype=float)

    # tol = tol * ynorm
    ztol = ztol * ynorm

    for it in range(maxit):
        psis_transposed = torch.transpose(psi, 1, 2)
        rcov = torch.abs(psis_transposed @ residuals.unsqueeze(-1)).squeeze()
        idxs = torch.argmax(rcov, dim=1).squeeze()

        active.append(idxs)

        active_tensor = torch.stack(active, dim=1)

        # repeat psi batch_size times along new axis
        psi_rep = psi.repeat(ys.shape[0], 1, 1)

        # For each psi in the batch, select the respective indexes
        # from the active_tensor.
        # psi active will have shape (batch, m, n_active)
        idx_tensor = active_tensor.unsqueeze(1).repeat(1, psi.shape[1], 1)
        psi_active = torch.gather(psi_rep, 2, idx_tensor)

        # remove eventually repeated columns from psi_active
        psi_active = torch.unique(psi_active, dim=2)

        # Solve least squares problem
        coefis, _, _, _ = torch.linalg.lstsq(psi_active, ys)

        # Update the coefficients
        coef.scatter_(dim=1, index=active_tensor, src=coefis.to("cuda"))

        # Update the residuals
        x_star = torch.matmul(
            psi_active, coefis.unsqueeze(-1).to("cuda")).squeeze()
        residuals = ys - x_star

        if len(active) >= ncoef:
            break
        if it == maxit - 1:
            break

    return coef


def save_baseline_mds(
    l1,
    omp,
    sparsity_levels,
    out_dir,
    ds_type="test",
):
    train_filenames, valid_filenames, test_filenames = Sparse_MD_Dataset.make_splits(
        config["SUBJECTS"],
        config["ACTIVITIES"],
        subsample_factor=config["DATASET_SUBSAMPLE_FACTOR"],
        seed=config["DATASET_SPLIT_SEED"],
        train=config["SPLIT_PTGS"][0],
        valid=config["SPLIT_PTGS"][1],
        test=config["SPLIT_PTGS"][2],
    )

    if ds_type == "train":
        dataset = Sparse_MD_Dataset(train_filenames)
    elif ds_type == "valid":
        dataset = Sparse_MD_Dataset(valid_filenames)
    elif ds_type == "test":
        dataset = Sparse_MD_Dataset(test_filenames)
    else:
        raise ValueError("ds_type must be either train or test")

    W = 64
    omp_ncoef = 5
    lasso_lambda = 5

    W_d_cpx = partial_fourier(W, np.arange(W))
    W_d_real = complex_to_real_matrix(W_d_cpx)

    # Initialize output directories for the mds
    os.makedirs(os.path.join(out_dir, "omp"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "lasso"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "mds_plots"), exist_ok=True)

    for sparsity_level in sparsity_levels:
        for idx in range(len(dataset)):
            # for idx in range(1):
            X = dataset[idx][0]
            mD_columns = dataset[idx][2]

            recs_omp = []
            recs_l1 = []

            tic = time.time()

            for j in range(X.shape[0]):
                print(f"{j}/{X.shape[0]}", end="\r", flush=True)
                chunk = X[j]

                # generate random mask where element is equal to zero with probability
                # equal to sparsity_level
                chunk_mask = dataset.generate_mask(
                    chunk.shape[1] // 2, sparsity_level
                ).to("cpu")

                # apply mask on chunk
                masked_chunk = chunk * chunk_mask.unsqueeze(0)

                # OMP
                recs_bin = []
                for y in masked_chunk:
                    y = y.numpy()
                    y_cpx = real_to_complex_vector(y)
                    sol = omp(
                        psi=W_d_cpx,
                        y=y_cpx,
                        ncoef=omp_ncoef,
                        maxit=30,
                        tol=0.05,
                        ztol=0.005,
                    )

                    recs_bin.append(sol)

                recs_bin = np.stack(recs_bin, 0)
                recs_omp.append(recs_bin)

                # lasso
                recs_bin_l1 = []
                for y in masked_chunk:
                    sol = l1(psi=W_d_real, y=y.numpy(), lbd=lasso_lambda)
                    sol_cpx = real_to_complex_vector(sol)

                    recs_bin_l1.append(sol_cpx)

                recs_bin_l1 = np.stack(recs_bin_l1, 0)
                recs_l1.append(recs_bin_l1)

            mDs_omp = []
            for rec_crop in recs_omp:
                # convert reconstructions to mD columns
                mD = process_cpx_crop(rec_crop)

                mD = min_max_freq(mD[np.newaxis, :])
                mDs_omp.append(mD.squeeze())

            omp_md = np.stack(mDs_omp, 0)

            # reverse omp1_spectrum in the columns
            omp_md = np.flip(omp_md, axis=1)

            # save md to file
            np.save(
                os.path.join(
                    out_dir,
                    "omp",
                    f"omp_md_{sparsity_level}_ncoef{omp_ncoef}_{idx}.npy",
                ),
                omp_md,
            )

            mDs_l1 = []
            for rec_crop in recs_l1:
                # convert reconstructions to mD columns
                mD = process_cpx_crop(rec_crop)

                mD = min_max_freq(mD[np.newaxis, :])
                mDs_l1.append(mD.squeeze())

            l1_md = np.stack(mDs_l1, 0)

            toc = time.time()
            print(f"Elapsed time: {toc - tic}")

            # reverse l1_spectrum in the columns
            l1_md = np.flip(l1_md, axis=1)
            # save md to file
            np.save(
                os.path.join(
                    out_dir,
                    "lasso",
                    f"lasso_md_{sparsity_level}_lambda{lasso_lambda}_{idx}.npy",
                ),
                l1_md,
            )

            # Plot the microdoppler side by side with the ground truth
            fig, ax = plt.subplots(1, 3, figsize=(10, 5))
            ax[0].imshow(omp_md.T, aspect="auto")
            ax[0].set_title(f"OMP (for loop), sparsity = {sparsity_level}")

            ax[1].imshow(l1_md.T, aspect="auto")
            ax[1].set_title(f"Lasso, sparsity = {sparsity_level}")

            ax[2].imshow(mD_columns.T, aspect="auto")
            ax[2].set_title(f"Ground Truth, sparsity = {sparsity_level}")
            plt.savefig(
                os.path.join(out_dir, "mds_plots",
                             f"md_{sparsity_level}_{idx}.png"),
            )
            plt.close()


def update_results_dir(
    results_dir,
    sparsity_level,
    omp_SSIMs,
    omp_PSNRs,
    omp_shape_MSEs,
    omp_noise_MSEs,
    omp_MSEs,
    omp_MAEs,
    l1_SSIMs,
    l1_PSNRs,
    l1_shape_MSEs,
    l1_noise_MSEs,
    l1_MSEs,
    l1_MAEs,
):
    results_dir.setdefault(str(sparsity_level), {})
    results_dir[str(sparsity_level)].setdefault("OMP", {})
    results_dir[str(sparsity_level)]["OMP"].setdefault("SSIM", {})
    results_dir[str(sparsity_level)]["OMP"].setdefault("PSNR", {})
    results_dir[str(sparsity_level)]["OMP"].setdefault("MSE", {})
    results_dir[str(sparsity_level)]["OMP"].setdefault("MSE_shape", {})
    results_dir[str(sparsity_level)]["OMP"].setdefault("MSE_noise", {})
    results_dir[str(sparsity_level)]["OMP"].setdefault("MAE", {})

    results_dir[str(sparsity_level)]["OMP"]["SSIM"].setdefault("values", [])
    results_dir[str(sparsity_level)]["OMP"]["SSIM"]["values"].append(omp_SSIMs)
    results_dir[str(sparsity_level)
                ]["OMP"]["SSIM"]["mean"] = np.mean(omp_SSIMs)
    results_dir[str(sparsity_level)]["OMP"]["SSIM"]["std"] = np.std(omp_SSIMs)

    results_dir[str(sparsity_level)]["OMP"]["PSNR"].setdefault("values", [])
    results_dir[str(sparsity_level)]["OMP"]["PSNR"]["values"].append(omp_PSNRs)
    results_dir[str(sparsity_level)
                ]["OMP"]["PSNR"]["mean"] = np.mean(omp_PSNRs)
    results_dir[str(sparsity_level)]["OMP"]["PSNR"]["std"] = np.std(omp_PSNRs)

    results_dir[str(sparsity_level)]["OMP"]["MSE"].setdefault("values", [])
    results_dir[str(sparsity_level)]["OMP"]["MSE"]["values"].append(omp_MSEs)
    results_dir[str(sparsity_level)]["OMP"]["MSE"]["mean"] = np.mean(omp_MSEs)
    results_dir[str(sparsity_level)]["OMP"]["MSE"]["std"] = np.std(omp_MSEs)

    results_dir[str(sparsity_level)]["OMP"]["MSE_shape"].setdefault(
        "values", [])
    results_dir[str(sparsity_level)]["OMP"]["MSE_shape"]["values"].append(
        omp_shape_MSEs
    )
    results_dir[str(sparsity_level)]["OMP"]["MSE_shape"]["mean"] = np.mean(
        omp_shape_MSEs
    )
    results_dir[str(sparsity_level)]["OMP"]["MSE_shape"]["std"] = np.std(
        omp_shape_MSEs)

    results_dir[str(sparsity_level)]["OMP"]["MSE_noise"].setdefault(
        "values", [])
    results_dir[str(sparsity_level)]["OMP"]["MSE_noise"]["values"].append(
        omp_noise_MSEs
    )
    results_dir[str(sparsity_level)]["OMP"]["MSE_noise"]["mean"] = np.mean(
        omp_noise_MSEs
    )
    results_dir[str(sparsity_level)]["OMP"]["MSE_noise"]["std"] = np.std(
        omp_noise_MSEs)

    results_dir[str(sparsity_level)]["OMP"]["MAE"].setdefault("values", [])
    results_dir[str(sparsity_level)]["OMP"]["MAE"]["values"].append(omp_MAEs)
    results_dir[str(sparsity_level)]["OMP"]["MAE"]["mean"] = np.mean(omp_MAEs)
    results_dir[str(sparsity_level)]["OMP"]["MAE"]["std"] = np.std(omp_MAEs)

    # Update Lasso results
    results_dir.setdefault(str(sparsity_level), {})
    results_dir[str(sparsity_level)].setdefault("LASSO", {})
    results_dir[str(sparsity_level)]["LASSO"].setdefault("SSIM", {})
    results_dir[str(sparsity_level)]["LASSO"].setdefault("PSNR", {})
    results_dir[str(sparsity_level)]["LASSO"].setdefault("MSE", {})
    results_dir[str(sparsity_level)]["LASSO"].setdefault("MSE_shape", {})
    results_dir[str(sparsity_level)]["LASSO"].setdefault("MSE_noise", {})
    results_dir[str(sparsity_level)]["LASSO"].setdefault("MAE", {})

    results_dir[str(sparsity_level)]["LASSO"]["SSIM"].setdefault("values", [])
    results_dir[str(sparsity_level)
                ]["LASSO"]["SSIM"]["values"].append(l1_SSIMs)
    results_dir[str(sparsity_level)
                ]["LASSO"]["SSIM"]["mean"] = np.mean(l1_SSIMs)
    results_dir[str(sparsity_level)]["LASSO"]["SSIM"]["std"] = np.std(l1_SSIMs)

    results_dir[str(sparsity_level)]["LASSO"]["PSNR"].setdefault("values", [])
    results_dir[str(sparsity_level)
                ]["LASSO"]["PSNR"]["values"].append(l1_PSNRs)
    results_dir[str(sparsity_level)
                ]["LASSO"]["PSNR"]["mean"] = np.mean(l1_PSNRs)
    results_dir[str(sparsity_level)]["LASSO"]["PSNR"]["std"] = np.std(l1_PSNRs)

    results_dir[str(sparsity_level)]["LASSO"]["MSE"].setdefault("values", [])
    results_dir[str(sparsity_level)]["LASSO"]["MSE"]["values"].append(l1_MSEs)
    results_dir[str(sparsity_level)]["LASSO"]["MSE"]["mean"] = np.mean(l1_MSEs)
    results_dir[str(sparsity_level)]["LASSO"]["MSE"]["std"] = np.std(l1_MSEs)

    results_dir[str(sparsity_level)]["LASSO"]["MSE_shape"].setdefault(
        "values", [])
    results_dir[str(sparsity_level)]["LASSO"]["MSE_shape"]["values"].append(
        l1_shape_MSEs
    )
    results_dir[str(sparsity_level)]["LASSO"]["MSE_shape"]["mean"] = np.mean(
        l1_shape_MSEs
    )
    results_dir[str(sparsity_level)]["LASSO"]["MSE_shape"]["std"] = np.std(
        l1_shape_MSEs
    )

    results_dir[str(sparsity_level)]["LASSO"]["MSE_noise"].setdefault(
        "values", [])
    results_dir[str(sparsity_level)]["LASSO"]["MSE_noise"]["values"].append(
        l1_noise_MSEs
    )
    results_dir[str(sparsity_level)]["LASSO"]["MSE_noise"]["mean"] = np.mean(
        l1_noise_MSEs
    )
    results_dir[str(sparsity_level)]["LASSO"]["MSE_noise"]["std"] = np.std(
        l1_noise_MSEs
    )

    results_dir[str(sparsity_level)]["LASSO"]["MAE"].setdefault("values", [])
    results_dir[str(sparsity_level)]["LASSO"]["MAE"]["values"].append(l1_MAEs)
    results_dir[str(sparsity_level)]["LASSO"]["MAE"]["mean"] = np.mean(l1_MAEs)
    results_dir[str(sparsity_level)]["LASSO"]["MAE"]["std"] = np.std(l1_MAEs)

    return results_dir


def compute_baseline_metrics(update_results_dir):
    md_dir = os.path.join("results", "baselines", "mds")

    # load test set
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

    results_dir = {}

    sparsity_levels = [0.5, 0.75, 0.9]

    for sparsity_level in sparsity_levels:
        omp_SSIMs = []
        omp_PSNRs = []
        omp_shape_MSEs = []
        omp_noise_MSEs = []
        omp_MSEs = []
        omp_MAEs = []

        l1_SSIMs = []
        l1_PSNRs = []
        l1_shape_MSEs = []
        l1_noise_MSEs = []
        l1_MSEs = []
        l1_MAEs = []

        for idx in range(len(test_set)):
            X = test_set[idx][0]
            IHT_output = test_set[idx][1]
            mD_columns = test_set[idx][2].numpy()

            omp_md = np.load(
                os.path.join(
                    md_dir,
                    "omp",
                    f"omp_md_{sparsity_level}_ncoef5_{idx}.npy",
                )
            )

            omp_md_gt = np.load(
                os.path.join(
                    md_dir,
                    "omp",
                    f"omp_md_0.0_ncoef5_{idx}.npy",
                )
            )

            l1_md = np.load(
                os.path.join(
                    md_dir,
                    "lasso",
                    f"lasso_md_{sparsity_level}_lambda5_{idx}.npy",
                )
            )

            l1_md_gt = np.load(
                os.path.join(
                    md_dir,
                    "lasso",
                    f"lasso_md_0.0_lambda5_{idx}.npy",
                )
            )

            (
                omp_SSIM,
                omp_PSNR,
                omp_shape_MSE,
                omp_noise_MSE,
                omp_MSE,
                omp_MAE,
            ) = compute_metrics(omp_md, omp_md_gt)

            omp_SSIMs.append(omp_SSIM)
            omp_PSNRs.append(omp_PSNR)
            omp_shape_MSEs.append(omp_shape_MSE)
            omp_noise_MSEs.append(omp_noise_MSE)
            omp_MSEs.append(omp_MSE)
            omp_MAEs.append(omp_MAE)

            (
                l1_SSIM,
                l1_PSNR,
                l1_shape_MSE,
                l1_noise_MSE,
                l1_MSE,
                l1_MAE,
            ) = compute_metrics(l1_md, l1_md_gt)

            l1_SSIMs.append(l1_SSIM)
            l1_PSNRs.append(l1_PSNR)
            l1_shape_MSEs.append(l1_shape_MSE)
            l1_noise_MSEs.append(l1_noise_MSE)
            l1_MSEs.append(l1_MSE)
            l1_MAEs.append(l1_MAE)

            # - dict_keys(['0.1', '0.3', '0.5', '0.75', '0.9'])
            #     - dict_keys(['LIHT', 'IHT_fixed', 'IHT_converged'])
            #         - dict_keys(['SSIM', 'PSNR', 'MSE', 'MSE_shape', 'MSE_noise', 'MAE'])
            #             - dict_keys(['values', 'mean', 'std'])

        # Update OMP results
        results_dir = update_results_dir(
            results_dir,
            sparsity_level,
            omp_SSIMs,
            omp_PSNRs,
            omp_shape_MSEs,
            omp_noise_MSEs,
            omp_MSEs,
            omp_MAEs,
            l1_SSIMs,
            l1_PSNRs,
            l1_shape_MSEs,
            l1_noise_MSEs,
            l1_MSEs,
            l1_MAEs,
        )

        print()

    # Save results_dir to json
    with open(os.path.join("results", "baselines", "results_2.json"), "w") as f:
        json.dump(results_dir, f, indent=4)


if __name__ == "__main__":
    import pickle
    from cnn_dataset import CNN_Dataset
    from base_cnn import train_cnn_baselines, test_cnn_baselines
    from hyperparams import CONFIG as cfg

    # Execute to recompute all test set microdopplers from OMP and Lasso
    # save_baseline_mds(
    #     l1,
    #     omp,
    #     sparsity_levels=[0.0],
    #     ds_type="valid",
    #     out_dir=os.path.join("results", "baselines", "valid_set_mds"),
    # )

    # Compute metrics on the test set
    # load md from file
    # compute_baseline_metrics(update_results_dir)

    # CNN Part
    (
        train_filenames,
        valid_filenames,
        test_filenames,
    ) = Sparse_MD_Dataset.make_splits(
        cfg["SUBJECTS"],
        cfg["ACTIVITIES"],
        subsample_factor=cfg["DATASET_SUBSAMPLE_FACTOR"],
        seed=cfg["DATASET_SPLIT_SEED"],
        train=cfg["SPLIT_PTGS"][0],
        valid=cfg["SPLIT_PTGS"][1],
        test=cfg["SPLIT_PTGS"][2],
    )

    # Generate OMP and LASSO microdopplers for the training of the CNN
    for algorithm in ["omp", "lasso"]:
        train_dir = os.path.join(
            "data", "cnn_dataset", "baselines", algorithm, "train_set"
        )
        valid_dir = os.path.join(
            "data", "cnn_dataset", "baselines", algorithm, "valid_set"
        )
        test_dir_50 = os.path.join(
            "data", "cnn_dataset", "baselines", algorithm, "test_set_0.5"
        )
        test_dir_75 = os.path.join(
            "data", "cnn_dataset", "baselines", algorithm, "test_set_0.75"
        )
        test_dir_90 = os.path.join(
            "data", "cnn_dataset", "baselines", algorithm, "test_set_0.9"
        )

        generate_dataset = True
        if generate_dataset:

            # Train set
            CNN_Dataset.generate_dataset_baseline(
                filenames=train_filenames,
                in_folder=os.path.join(
                    "results", "baselines", "train_set_mds", algorithm
                ),
                out_folder=train_dir,
                sparsity_level="0.0",
            )

            # Valid set
            CNN_Dataset.generate_dataset_baseline(
                filenames=valid_filenames,
                in_folder=os.path.join(
                    "results", "baselines", "valid_set_mds", algorithm
                ),
                out_folder=valid_dir,
                sparsity_level="0.0",
            )

            # Test set 50
            CNN_Dataset.generate_dataset_baseline(
                filenames=test_filenames,
                in_folder=os.path.join(
                    "results", "baselines", "test_set_mds", algorithm
                ),
                out_folder=test_dir_50,
                sparsity_level="0.5",
            )

            # Test set 75
            CNN_Dataset.generate_dataset_baseline(
                filenames=test_filenames,
                in_folder=os.path.join(
                    "results", "baselines", "test_set_mds", algorithm
                ),
                out_folder=test_dir_75,
                sparsity_level="0.75",
            )

            # Test set 90
            CNN_Dataset.generate_dataset_baseline(
                filenames=test_filenames,
                in_folder=os.path.join(
                    "results", "baselines", "test_set_mds", algorithm
                ),
                out_folder=test_dir_90,
                sparsity_level="0.9",
            )

        class_dict = {
            0: "WALKING",
            1: "RUNNING",
            2: "SITTING",
            3: "HANDS",
        }

        train_set = CNN_Dataset(train_dir)
        valid_set = CNN_Dataset(valid_dir)
        test_set_50 = CNN_Dataset(test_dir_50)
        test_set_75 = CNN_Dataset(test_dir_75)
        test_set_90 = CNN_Dataset(test_dir_90)

        # Train CNN on train set
        cnn_model_path = os.path.join(
            "results", "baselines", f"cnn_{algorithm}.pt")
        (
            training_stats,
            test_preds_50,
            test_preds_75,
            test_preds_90,
            test_labels,
        ) = train_cnn_baselines(
            train_set,
            valid_set,
            test_set_50,
            test_set_75,
            test_set_90,
            out_model_path=cnn_model_path,
        )

        # save training stats as pickle
        with open(
            os.path.join("results", "baselines",
                         f"cnn_training_stats_{algorithm}.pkl"),
            "wb",
        ) as f:
            pickle.dump(training_stats, f)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        ax[0].plot(training_stats[0], label="Train loss")
        ax[0].plot(training_stats[1], label="Valid loss")

        ax[1].plot(training_stats[2], label="Train accuracy")
        ax[1].plot(training_stats[3], label="Valid accuracy")
        plt.legend()
        plt.savefig(os.path.join(
            "results", "baselines", f"{algorithm}_train.png"))
        plt.close()

        # Get the predictions of the trained CNN on all the test sets.
        # For each sparsity level save the predictions in a dictionary structured like:
        # - dict_keys(['0.5', '0.75', '0.9'])
        #     - dict_keys(['predictions_LASSO/OMP', 'labels'])

        out_results_path = os.path.join(
            "results", "baselines", f"cnn_results_{algorithm}.pkl"
        )

        cnn_predictions = {}

        # preds50, test_labels = test_cnn_baselines(
        #     cnn_model_path=cnn_model_path, test_dataset=test_set_50
        # )

        # preds75, _ = test_cnn_baselines(
        #     cnn_model_path=cnn_model_path, test_dataset=test_set_75
        # )

        # preds90, _ = test_cnn_baselines(
        #     cnn_model_path=cnn_model_path, test_dataset=test_set_90
        # )

        cnn_predictions["0.5"] = {
            "predictions": test_preds_50,
            "labels": test_labels,
        }
        cnn_predictions["0.75"] = {
            "predictions": test_preds_75,
            "labels": test_labels,
        }
        cnn_predictions["0.9"] = {
            "predictions": test_preds_90,
            "labels": test_labels,
        }

        # save predictions as pickle
        with open(out_results_path, "wb") as f:
            pickle.dump(cnn_predictions, f)
        # Update the plots
