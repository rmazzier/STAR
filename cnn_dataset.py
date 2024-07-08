import os
import numpy as np
import torch
import pickle
from pathlib import Path
from matplotlib import pyplot as plt

import hyperparams as hpm
import hyperparams_cnn as hpm_cnn
from sparse_dataset_gen import Sparse_MD_Dataset
from md_extraction.sparsity_based import partial_fourier, iht
from utils import real_to_complex_vector, complex_to_real_vector
from utils import (
    real_to_complex_vector,
    process_cpx_crop,
)
from md_extraction.utils_jp import min_max_freq
from evaluation import load_model_and_config
from models import STAR, init_model
from utils import get_act_from_filename, IHT_to_mD


class CNN_Dataset(torch.utils.data.Dataset):
    """
    Dataset used for CNN evaluation. Composed as follows:
        - Training set is composed of ground truth full window IHT spectrums.
        - Test set can be of two kinds:
                - Sparse reconstructions from IHT.
                - Sparse reconstructions from LIHT."""

    def __init__(self, data_dir):
        super(CNN_Dataset, self).__init__()
        self.filenames = os.listdir(data_dir)
        self.data_dir = data_dir
        self.class_dict = {
            "WALKING": 0,
            "RUNNING": 1,
            "SITTING": 2,
            "HANDS": 3,
        }

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        with open(
            os.path.join(self.data_dir, self.filenames[idx]),
            "rb",
        ) as file:
            mD = pickle.load(file)

        label = self.class_dict[get_act_from_filename(self.filenames[idx])]
        return (mD.unsqueeze(0).float(), label)

    @staticmethod
    def crop_mD(mD, length, step):
        i = 0
        out_mDs = []
        while i + length < mD.shape[0]:
            out_mDs.append(mD[i: i + length, :])
            i += step
        return out_mDs

    @staticmethod
    def generate_train_set(filenames, out_folder):
        os.makedirs(out_folder, exist_ok=True)
        # remove old files
        for file in os.listdir(out_folder):
            os.remove(os.path.join(out_folder, file))

        ds = Sparse_MD_Dataset(filenames)
        for i in range(len(ds)):
            _, _, mD_whole = ds[i]
            cropped_mDs = CNN_Dataset.crop_mD(
                mD_whole, hpm_cnn.MD_LEN, hpm_cnn.STEP)

            for j, mD in enumerate(cropped_mDs):
                with open(
                    os.path.join(out_folder, filenames[i].replace(
                        ".obj", f"_{j}.pkl")),
                    "wb",
                ) as out:
                    pickle.dump(mD, out)

    @staticmethod
    def generate_dataset_baseline(filenames, in_folder, out_folder, sparsity_level):
        os.makedirs(out_folder, exist_ok=True)

        # remove old files
        for file in os.listdir(out_folder):
            os.remove(os.path.join(out_folder, file))

        whole_mDs = []
        mds_fnames = os.listdir(in_folder)

        # select only the ones of the correct sparsity level
        mds_fnames = [fname for fname in mds_fnames if sparsity_level in fname]

        # sort wrt index
        saved_mds = sorted(
            mds_fnames, key=lambda x: int(x.split("_")[-1].split(".")[0])
        )
        for md_fname in saved_mds:
            md = np.load(os.path.join(in_folder, md_fname))
            whole_mDs.append(md)

        # plot a sample of 9 mds with their respective label and sparsity level as titles
        _, axs = plt.subplots(3, 3)
        for i, ax in enumerate(axs.flat):
            if i >= len(whole_mDs):
                break
            ax.imshow(whole_mDs[i].T, aspect="auto")
            ax.set_title(f"{filenames[i]}")

        set_type = in_folder.split("/")[-2]

        plt.savefig(
            os.path.join(
                "results", "baselines", f"sample_{set_type}_{sparsity_level}.png"
            )
        )
        plt.close()

        for i, mD_whole in enumerate(whole_mDs):
            cropped_mDs = CNN_Dataset.crop_mD(
                mD_whole, hpm_cnn.MD_LEN, hpm_cnn.STEP)
            cropped_mDs = torch.tensor(cropped_mDs)

            for j, mD in enumerate(cropped_mDs):
                with open(
                    os.path.join(out_folder, filenames[i].replace(
                        ".obj", f"_{j}.pkl")),
                    "wb",
                ) as out:
                    pickle.dump(mD, out)

    @staticmethod
    def generate_dataset_rev1_train(dataset_rev1_instance, out_folder):
        # delete files in out_folder
        for file in os.listdir(out_folder):
            os.remove(os.path.join(out_folder, file))

        for i, (_, _, mD_whole) in enumerate(dataset_rev1_instance):
            act_name = dataset_rev1_instance.filenames[i].split("_")[2]
            cropped_mDs = CNN_Dataset.crop_mD(
                mD_whole, hpm_cnn.MD_LEN, hpm_cnn.STEP)

            for j, mD in enumerate(cropped_mDs):
                out_name = f"0_0_{act_name}_{i}_{j}.pkl"
                with open(
                    os.path.join(
                        out_folder,
                        out_name,
                    ),
                    "wb",
                ) as out:
                    pickle.dump(mD, out)
        pass

    @staticmethod
    def generate_dataset_rev1_test(in_folder, out_folder):
        # delete files in out_folder
        for file in os.listdir(out_folder):
            os.remove(os.path.join(out_folder, file))

        files = os.listdir(in_folder)

        for file in files:
            i = int(file.split("_")[-1].split(".")[0])
            act_name = file.split("_")[1]
            mD_whole = np.load(os.path.join(in_folder, file))

            cropped_mDs = CNN_Dataset.crop_mD(
                mD_whole, hpm_cnn.MD_LEN, hpm_cnn.STEP)
            cropped_mDs = torch.tensor(cropped_mDs)

            for j, mD in enumerate(cropped_mDs):
                out_name = f"0_0_{act_name}_{i}_{j}.pkl"
                with open(
                    os.path.join(
                        out_folder,
                        out_name,
                    ),
                    "wb",
                ) as out:
                    pickle.dump(mD, out)

        pass

    @staticmethod
    def generate_test_set_IHT(
        filenames, out_folder, p_remove, fix_iht_iters, n_IHT_iters
    ):
        # remove old files
        os.makedirs(out_folder, exist_ok=True)
        for file in os.listdir(out_folder):
            os.remove(os.path.join(out_folder, file))
        ds = Sparse_MD_Dataset(filenames)
        for i in range(len(ds)):
            X_test, _, _ = ds[i]
            IHT_reconstructions = []
            for j in range(X_test.shape[0]):
                chunk = X_test[j].to(hpm.DEVICE)
                chunk_mask = ds.generate_mask(
                    chunk.shape[1] // 2, p_remove=p_remove)

                # apply mask on chunk
                masked_chunk = chunk * chunk_mask.unsqueeze(0)

                # now run IHT on the masked chunk
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
                spectrum = iht(
                    rep_psi,
                    partial_chunk * partial_win,
                    fix_iht_iters,
                    n_iters=n_IHT_iters,
                )
                IHT_reconstructions.append(spectrum.squeeze())

            # process iht spectrum
            processed_IHT_mD = []
            for rec_crop in IHT_reconstructions:
                mD_shift = process_cpx_crop(rec_crop)
                mD = min_max_freq(mD_shift[np.newaxis, :])
                processed_IHT_mD.append(mD.squeeze())

            mD_whole = torch.tensor(np.stack(processed_IHT_mD, 0))
            cropped_mDs = CNN_Dataset.crop_mD(
                mD_whole, hpm_cnn.MD_LEN, hpm_cnn.STEP)
            for j, mD in enumerate(cropped_mDs):
                # Save IHT spectrum
                with open(
                    os.path.join(out_folder, filenames[i].replace(
                        ".obj", f"_{j}.pkl")),
                    "wb",
                ) as out:
                    pickle.dump(mD, out)

                # sfx = out_folder.split("/")[-1]
                # out_debug = os.path.join(hpm.CNN_DATA_PATH, "debug", sfx)
                # os.makedirs(out_debug, exist_ok=True)
                # plt.imshow(mD.T)
                # plt.savefig(os.path.join(out_debug, f"{filenames[i]}_{j}.png"))
                # plt.close()

        pass

    @staticmethod
    def generate_test_set_LIHT(filenames, out_folder, p_remove, LIHT_model, cfg):

        # remove old files
        os.makedirs(out_folder, exist_ok=True)
        for file in os.listdir(out_folder):
            os.remove(os.path.join(out_folder, file))

        LIHT_model.eval()
        ds = Sparse_MD_Dataset(filenames)
        for i in range(len(ds)):
            X_test, _, _ = ds[i]

            LIHT_reconstructions = []
            past_wins_IHT_t = []
            for j in range(X_test.shape[0]):
                chunk = X_test[j].to(hpm.DEVICE).float()
                chunk_mask = ds.generate_mask(
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

                mD_column_pred_test, IHT_pred_test = LIHT_model(
                    masked_chunk, past_wins_t
                )
                # mD_column_pred_test = IHT_to_mD(IHT_pred_test)
                LIHT_reconstructions.append(mD_column_pred_test.detach().cpu())

            LIHT_mD = np.stack(LIHT_reconstructions, 0)
            # shift LIHT_mD
            mD_whole = np.roll(LIHT_mD, 32, axis=-1)
            cropped_mDs = CNN_Dataset.crop_mD(
                mD_whole, hpm_cnn.MD_LEN, hpm_cnn.STEP)
            for j, mD in enumerate(cropped_mDs):

                # Save LIHT spectrum
                with open(
                    os.path.join(out_folder, filenames[i].replace(
                        ".obj", f"_{j}.pkl")),
                    "wb",
                ) as out:
                    pickle.dump(torch.tensor(mD), out)

                sfx = out_folder.split("/")[-1]
                out_debug = os.path.join(hpm.CNN_DATA_PATH, "debug", sfx)
                os.makedirs(out_debug, exist_ok=True)
                plt.imshow(mD.T)
                plt.savefig(os.path.join(out_debug, f"{filenames[i]}_{j}.png"))
                plt.close()

    @staticmethod
    def generate_splits(LIHT_model, cfg, p_remove):
        print("Generating CNN Dataset")
        # cfg, model_weights = load_model_and_config(model_path)

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

        print("Generating training set...")

        CNN_Dataset.generate_train_set(
            train_filenames, os.path.join(hpm.CNN_DATA_PATH, "train")
        )

        print("Generating validation set...")

        CNN_Dataset.generate_train_set(
            valid_filenames, os.path.join(hpm.CNN_DATA_PATH, "valid")
        )

        print("Generating test set IHT...")
        CNN_Dataset.generate_test_set_IHT(
            test_filenames,
            os.path.join(hpm.CNN_DATA_PATH, "test_IHT"),
            p_remove=p_remove,
            fix_iht_iters=False,
            n_IHT_iters=cfg["N_LIHT_ITERS"],
        )

        print("Generating test set LIHT...")
        CNN_Dataset.generate_test_set_LIHT(
            test_filenames,
            os.path.join(hpm.CNN_DATA_PATH, "test_LIHT"),
            p_remove=p_remove,
            LIHT_model=LIHT_model,
            cfg=cfg,
        )


if __name__ == "__main__":

    cfg = hpm.CONFIG

    model_name = "deft-sweep-15"
    model_path = os.path.join("wandb_downloads", "LIHT_Sweep_4.0", model_name)

    cfg, model_weights = load_model_and_config(model_path)
    model = init_model(cfg)

    model.load_state_dict(model_weights)

    CNN_Dataset.generate_splits(model, cfg, p_remove=0.9)
    cnn_train_set = CNN_Dataset(os.path.join(hpm.CNN_DATA_PATH, "train"))
    cnn_test_set_IHT = CNN_Dataset(os.path.join(hpm.CNN_DATA_PATH, "test_IHT"))
    cnn_test_set_LIHT = CNN_Dataset(
        os.path.join(hpm.CNN_DATA_PATH, "test_LIHT"))
    for i in range(10):
        print(cnn_train_set[i].shape)
        pass

    for i in range(10):
        print(cnn_test_set_IHT[i].shape)
        pass

    for i in range(10):
        print(cnn_test_set_LIHT[i].shape)
        pass
