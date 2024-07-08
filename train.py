import torch
import hyperparams as hpm
import matplotlib.pyplot as plt
import json

import matplotlib
from matplotlib.colors import ListedColormap

import wandb
import os
import numpy as np
import pickle

from models import init_model
from sparse_dataset_gen import Sparse_MD_Dataset

from md_extraction.utils_jp import complex_to_real_matrix
from md_extraction import sparsity_based

from evaluation import get_rec_mD, compute_metrics, evaluate_run

from base_cnn import train_cnn


def rowwise_MSE(output, target):
    squared_errors = ((output - target) ** 2).mean(-1)
    mses = squared_errors.mean(-1)
    # loss = mses.mean()
    return mses


def rowwise_MAE(output, target):
    abs_errors = (torch.abs((output - target))).mean(-1)
    maes = abs_errors.mean(-1)
    return maes


def recurrent_mse(zlist, target, weights):
    if len(zlist) != len(weights):
        raise Exception("len(zlist) != len(weights)")
    loss = 0
    for i, z in enumerate(zlist):
        loss += weights[i] * rowwise_MSE(z, target)
    return loss / len(zlist)


def recurrent_mae(zlist, target, weights):
    if len(zlist) != len(weights):
        raise Exception("len(zlist) != len(weights)")
    loss = 0
    for i, z in enumerate(zlist):
        loss += weights[i] * rowwise_MAE(z, target)
    return loss / len(zlist)


def w_D_regularization_term(W_d):
    # first get the fourier matrix
    F_cpx = sparsity_based.partial_fourier(hpm.W, np.arange(hpm.W))
    F = torch.tensor(complex_to_real_matrix(F_cpx))
    penalty = torch.norm(F - W_d.detach().cpu().squeeze()) ** 2

    return penalty


def execute_run(cfg):

    torch.manual_seed(0)

    # Save config file right away as pickle
    model_dir = os.path.join("./models", "STAR_GS")

    # Create directory if not present
    os.makedirs(model_dir, exist_ok=True)

    mname = cfg["MODEL_NAME"]
    with open(os.path.join(model_dir, f"{mname}_config.pkl"), "wb") as f:
        pickle.dump(cfg, f)

    W_d_cpx = sparsity_based.partial_fourier(hpm.W, np.arange(hpm.W))
    GT_FOURIER_MATRIX = complex_to_real_matrix(W_d_cpx)
    GT_FOURIER_MATRIX = torch.tensor(GT_FOURIER_MATRIX).float().unsqueeze(0)

    model = init_model(cfg)
    model = model.to(hpm.DEVICE)

    wandb.watch(model, log="all", log_graph=False, log_freq=100)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["ADAM_LR"])

    train_filenames, valid_filenames, test_filenames = Sparse_MD_Dataset.make_splits(
        cfg["SUBJECTS"],
        cfg["ACTIVITIES"],
        subsample_factor=cfg["DATASET_SUBSAMPLE_FACTOR"],
        seed=cfg["DATASET_SPLIT_SEED"],
        train=cfg["SPLIT_PTGS"][0],
        valid=cfg["SPLIT_PTGS"][1],
        test=cfg["SPLIT_PTGS"][2],
    )

    train_set = Sparse_MD_Dataset(train_filenames, p_burst=hpm.P_BURST)
    valid_set = Sparse_MD_Dataset(valid_filenames, p_burst=hpm.P_BURST)

    for ep in range(cfg["EPOCHS"]):
        print(f"-- Epoch : {ep} ---")

        epoch_train_losses = []
        epoch_train_IHT_losses = []
        epoch_train_mD_losses = []
        model.train()

        # shuffle the dataset
        train_set.filenames = train_set.rng.permutation(
            train_set.raw_filenames)
        for i, (X, IHT_output, mD_columns) in enumerate(train_set):
            X = X.to(hpm.DEVICE).float()
            IHT_output = IHT_output.to(hpm.DEVICE).float()
            mD_columns = mD_columns.to(hpm.DEVICE).float()

            sequence_losses = []
            sequence_IHT_losses = []
            sequence_mD_losses = []

            past_wins_IHT = []

            for j in range(X.shape[0]):
                chunk = X[j]
                IHT_out_gt = IHT_output[j]
                mD_column_gt = mD_columns[j]

                p_remove = train_set.rng.uniform(
                    hpm.P_REMOVE_BOUNDS[0], hpm.P_REMOVE_BOUNDS[1]
                )

                # generate random mask where element is equal to zero with probability
                # equal to p_remove
                chunk_mask = train_set.generate_mask(
                    chunk.shape[1] // 2, p_remove)

                # apply mask on chunk
                masked_chunk = chunk * chunk_mask.unsqueeze(0)

                # update past windows
                if not cfg["TEACHER_FORCING"]:
                    if j > 0:
                        past_wins_IHT.append(IHT_out_pred.detach())

                        if len(past_wins_IHT) > cfg["N_PAST_WINDOWS"]:
                            past_wins_IHT.pop(0)
                        past_wins = torch.stack(past_wins_IHT, dim=0)
                    else:
                        past_wins = []

                if cfg["TEACHER_FORCING"] and j == 0:
                    past_wins = []

                mD_column_pred, IHT_out_pred = model(masked_chunk, past_wins)

                if cfg["TEACHER_FORCING"]:
                    if j > 0:
                        past_wins_IHT.append(IHT_out_gt.detach())

                        if len(past_wins_IHT) > cfg["N_PAST_WINDOWS"]:
                            past_wins_IHT.pop(0)
                        past_wins = torch.stack(past_wins_IHT, dim=0)
                    else:
                        past_wins = []

                # Compute the mD column from IHT output
                # mD_column_pred = IHT_to_mD(IHT_out_pred)

                # ===== L_mD and L_IHT ======
                # Here I have to shift because the ground truth is already shifted by 32
                # so to bring it back I have to shift again

                shifted_mD_column_gt = torch.roll(mD_column_gt, 32, dims=0)
                if cfg["L1_LOSS"]:
                    L_mD = rowwise_MAE(
                        mD_column_pred,
                        shifted_mD_column_gt,
                    ).float()
                    L_IHT = rowwise_MAE(
                        IHT_out_pred,
                        IHT_out_gt,
                    ).float()
                else:
                    L_mD = rowwise_MSE(
                        mD_column_pred,
                        shifted_mD_column_gt,
                    ).float()
                    L_IHT = rowwise_MSE(
                        IHT_out_pred,
                        IHT_out_gt,
                    ).float()

                # ===== Regularization Term ======
                w_D_penalty = w_D_regularization_term(model.W_d)

                # ===== TOTAL LOSS =====
                loss = (
                    L_IHT * cfg["L_IHT_WEIGHT"]
                    + L_mD * cfg["L_MD_WEIGHT"]
                    + w_D_penalty * cfg["W_D_REG_WEIGHT"]
                )

                epoch_train_losses.append(loss.item())
                sequence_losses.append(loss.item())
                epoch_train_IHT_losses.append(L_IHT.item())
                sequence_IHT_losses.append(L_IHT.item())
                epoch_train_mD_losses.append(L_mD.item())
                sequence_mD_losses.append(L_mD.item())

                # Compute gradient
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=5.0, norm_type=2.0
                )

                # Optimization Step
                optimizer.step()

                # Reset gradients
                optimizer.zero_grad()

            if i % 10 == 0:
                print(f"Sequence {i}/{len(train_set)}: ")
                print(
                    f"total loss = {loss:.3f} ; L_IHT = {L_IHT.item():.3f}, L_mD = {L_mD.item():.3f}, w_D_penalty = {w_D_penalty.item():.3f}"
                )
        # Evaluation at the end of the epoch
        model.eval()
        epoch_valid_losses = []
        epoch_valid_IHT_losses = []
        epoch_valid_mD_losses = []

        with torch.no_grad():
            for i, (X_v, IHT_output_v, mD_columns_v) in enumerate(valid_set):
                X_v = X_v.to(hpm.DEVICE).float()
                IHT_output_v = IHT_output_v.to(hpm.DEVICE).float()
                mD_columns_v = mD_columns_v.to(hpm.DEVICE).float()

                past_wins_IHT_v = []
                for j in range(X_v.shape[0]):
                    chunk_v = X_v[j]
                    IHT_out_gt_v = IHT_output_v[j]
                    mD_column_gt_v = mD_columns_v[j]

                    p_remove = valid_set.rng.uniform(
                        hpm.P_REMOVE_BOUNDS[0], hpm.P_REMOVE_BOUNDS[1]
                    )

                    chunk_mask_v = valid_set.generate_mask(
                        chunk.shape[1] // 2, p_remove
                    )

                    # apply mask on chunk
                    masked_chunk_v = chunk_v * chunk_mask_v.unsqueeze(0)

                    # update past windows
                    if j > 0:
                        past_wins_IHT_v.append(IHT_out_pred_v.detach())
                        if len(past_wins_IHT_v) > cfg["N_PAST_WINDOWS"]:
                            past_wins_IHT_v.pop(0)
                        past_wins_v = torch.stack(past_wins_IHT_v, dim=0)
                    else:
                        past_wins_v = []

                    mD_column_pred_v, IHT_out_pred_v = model(
                        masked_chunk_v, past_wins_v
                    )
                    # mD_column_pred_v = IHT_to_mD(IHT_out_pred_v)

                    # Compute loss
                    shifted_mD_column_gt_v = torch.roll(
                        mD_column_gt_v, 32, dims=0)
                    if cfg["L1_LOSS"]:
                        L_mD_v = rowwise_MAE(
                            mD_column_pred_v,
                            shifted_mD_column_gt_v,
                        ).float()
                        L_IHT_v = rowwise_MAE(
                            IHT_out_pred_v, IHT_out_gt_v).float()
                    else:
                        L_mD_v = rowwise_MSE(
                            mD_column_pred_v,
                            shifted_mD_column_gt_v,
                        ).float()
                        L_IHT_v = rowwise_MSE(
                            IHT_out_pred_v, IHT_out_gt_v).float()

                    w_D_penalty_v = w_D_regularization_term(model.W_d)
                    loss_v = (
                        L_IHT_v * cfg["L_IHT_WEIGHT"]
                        + L_mD_v * cfg["L_MD_WEIGHT"]
                        + w_D_penalty_v * cfg["W_D_REG_WEIGHT"]
                    )
                    epoch_valid_losses.append(loss_v.item())
                    epoch_valid_IHT_losses.append(L_IHT_v.item())
                    epoch_valid_mD_losses.append(L_mD_v.item())

        mean_train_loss = torch.mean(torch.tensor(epoch_train_losses))
        mean_valid_loss = torch.mean(torch.tensor(epoch_valid_losses))

        mean_train_IHT_loss = torch.mean(torch.tensor(epoch_train_IHT_losses))
        mean_valid_IHT_loss = torch.mean(torch.tensor(epoch_valid_IHT_losses))

        mean_train_mD_loss = torch.mean(torch.tensor(epoch_train_mD_losses))
        mean_valid_mD_loss = torch.mean(torch.tensor(epoch_valid_mD_losses))

        print(
            f"Epoch: {ep} -- Train Loss: {mean_train_loss:.3f}; Train IHT Loss: {mean_train_IHT_loss:.3f}; Train mD Loss: {mean_train_mD_loss:.3f};\nValid Loss: {mean_valid_loss:.3f}; Valid IHT Loss: {mean_valid_IHT_loss:.3f}; Valid mD Loss: {mean_valid_mD_loss:.3f}"
        )
        if ep % 2 == 0:
            # ==========================================================

            wandb.log(
                {
                    "Training Loss": mean_train_loss,
                    "Validation Loss": mean_valid_loss,
                    "W_d": wandb.Image(model.W_d.detach().cpu().numpy()),
                }
            )

        else:
            wandb.log(
                {
                    "Training Loss": mean_train_loss,
                    "Validation Loss": mean_valid_loss,
                }
            )

    # SAVE THE MODEL
    print("Saving model...")
    model_dir = os.path.join(".", "models", "STAR_GS")
    os.makedirs(model_dir, exist_ok=True)
    model_save_path = os.path.join(model_dir, f"{cfg['MODEL_NAME']}.pt")
    torch.save(model.state_dict(), model_save_path)

    # Save model weights online on wandb
    wandb.save(model_save_path)
    wandb.save(os.path.join(model_dir, f"{cfg['MODEL_NAME']}_cfg.pkl"))

    ps_test = [0.1, 0.3, 0.5, 0.75, 0.9]

    print("Evaluating model...")
    results_dictionary = evaluate_run(
        model, cfg, "./results/Sweep", ps_test, plot_results=False
    )

    # save results in json
    with open(os.path.join(model_dir, f"{cfg['MODEL_NAME']}_results.json"), "w") as fp:
        json.dump(results_dictionary, fp)

    wandb.save(os.path.join(model_dir, f"{cfg['MODEL_NAME']}_results.json"))

    # results_dict[p_remove][key_model][key_metric].setdefault("values", [])
    mD_SSIM = results_dictionary[0.9]["LIHT"]["SSIM"]["mean"]
    mD_PSNR = results_dictionary[0.9]["LIHT"]["PSNR"]["mean"]
    MSE_rec_shape = results_dictionary[0.9]["LIHT"]["MSE_shape"]["mean"]
    MSE_rec_noise = results_dictionary[0.9]["LIHT"]["MSE_noise"]["mean"]
    MSE = results_dictionary[0.9]["LIHT"]["MSE"]["mean"]
    MAE = results_dictionary[0.9]["LIHT"]["MAE"]["mean"]

    for p in ps_test:

        cnn_results = train_cnn(model, cfg, p_remove=p, generate_data=True)

        # save results in json
        cnn_results_dir = os.path.join("./results", "cnn_ablation")
        os.makedirs(cnn_results_dir, exist_ok=True)
        cnn_save_path = os.path.join(
            cnn_results_dir, f"{cfg['MODEL_NAME']}_results_{p}.json"
        )
        with open(cnn_save_path, "w") as fp:
            json.dump(cnn_results, fp)

        wandb.save(cnn_save_path)

    preds_IHT = cnn_results["preds_IHT"]
    preds_LIHT = cnn_results["preds_LIHT"]
    labels = cnn_results["labels"]

    # Accuracy for IHT
    accuracy_IHT = (np.array(preds_IHT) ==
                    np.array(labels)).sum() / len(labels)
    # Accuracy for LIHT
    accuracy_LIHT = (np.array(preds_LIHT) ==
                     np.array(labels)).sum() / len(labels)

    return (
        mD_SSIM,
        mD_PSNR,
        MSE_rec_shape,
        MSE_rec_noise,
        MSE,
        MAE,
        accuracy_IHT,
        accuracy_LIHT,
    )


if __name__ == "__main__":
    from utils import parse_grid_search_csv

    # Sparse_MD_Dataset.generate_dataset()

    # Ablation runs dictionary
    grid_search_dict = parse_grid_search_csv("./ablation_runs.csv")

    # make sure all the lists are of the same length
    assert (
        len(set([len(grid_search_dict[key])
            for key in grid_search_dict.keys()])) == 1
    )

    n_runs = len(grid_search_dict[list(grid_search_dict.keys())[0]])

    for run_idx in list(range(n_runs)[:1]):
        current_config = hpm.CONFIG

        # Get the config for this run
        for key in grid_search_dict.keys():
            current_config[key] = grid_search_dict[key][run_idx]

        # Execute the run
        wandb.login()
        run = wandb.init(
            project="jstsp_snn_unfolding",
            config=current_config,
            name=current_config["MODEL_NAME"],
            notes=current_config["NOTES"],
            reinit=True,
            tags=current_config["WANDB_TAG"],
            mode=current_config["WANDB_MODE"],
        )
        (
            mD_SSIM,
            mD_PSNR,
            MSE_rec_shape,
            MSE_rec_noise,
            MSE,
            MAE,
            CNN_IHT,
            CNN_LIHT,
        ) = execute_run(current_config)
        print(f"mD_SSIM: {mD_SSIM}")
        print(f"mD_PSNR: {mD_PSNR}")
        print(f"MSE_rec_shape: {MSE_rec_shape}")
        print(f"MSE_rec_noise: {MSE_rec_noise}")
        print(f"MSE: {MSE}")
        print(f"MAE: {MAE}")
        print(f"CNN_IHT: {CNN_IHT}")
        print(f"CNN_LIHT: {CNN_LIHT}")
        print(f"CNN delta: {CNN_LIHT - CNN_IHT}")
        run.finish()
        print()
