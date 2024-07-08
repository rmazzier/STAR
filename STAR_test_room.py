import os
import pickle
import torch

from models import init_model
from sparse_dataset_gen import Dataset_Rev1
import hyperparams as hpm

from train import rowwise_MAE, rowwise_MSE, w_D_regularization_term


if __name__ == "__main__":

    config_whole_path = os.path.join(
        "models", "STAR_GS", "STAR_final_config.pkl"
    )

    model_weights_path = os.path.join(
        "models", "STAR_GS", "STAR_final.pt"
    )

    # This is the path for the finetuned version
    new_model_weights_path = os.path.join(
        "models", "STAR_GS", "STAR_ft_rev1.pt")

    # This is for the trained from scratch version
    new_model_weights_path = os.path.join(
        "models", "STAR_GS", "STAR_rt_rev1.pt")

    model_weights = torch.load(model_weights_path)

    with open(config_whole_path, "rb") as f:
        cfg = pickle.load(f)

    # Set to just 1 epoch for fine-tuning
    cfg["EPOCHS"] = 1

    train_set = Dataset_Rev1(split="train", regenerate=False)
    valid_set = Dataset_Rev1(split="valid", regenerate=False)
    test_set = Dataset_Rev1(split="test", regenerate=False)

    model = init_model(cfg)
    model.load_state_dict(model_weights)
    model = model.to(hpm.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["ADAM_LR"])

    for ep in range(cfg["EPOCHS"]):
        print(f"-- Epoch : {ep} ---")

        epoch_train_losses = []
        epoch_train_IHT_losses = []
        epoch_train_mD_losses = []
        model.train()

        # shuffle the dataset
        train_set.filenames = train_set.rng.permutation(train_set.filenames)
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
                torch.save(model.state_dict(), new_model_weights_path)
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

    # SAVE THE MODEL
    # model_dir = os.path.join("./models", "STAR_GS")
    torch.save(model.state_dict(), new_model_weights_path)
