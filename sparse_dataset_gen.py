import torch
import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from utils import (
    complex_to_real_vector,
    get_subj_from_filename,
    get_cir,
)
from md_extraction import sparsity_based
import hyperparams as hpm


class Dataset_Rev1(torch.utils.data.Dataset):
    """
    New dataset for revision. Used to assess generalization capabilities of the model.
    Sequences with indexes in [1,23] and [39, 45] are obtained in the same room as the training set of STAR.
    Sequences with indexes in [24,38] are obtained in a different room.

    Name convention of raw files is:
    TEST{idx1}_{idx2}_US_F2_track_{idx3}

    Where:
    idx1 =  the activity index (1=Walk, 2=Run, 3=Hands, 4=Sit)
    idx2 =  sequence index
    idx3 =  track index (there might be more than one track for each sequence,
            due to tracking algorithm errors)
    """

    activity_dict = {
        1: "WALKING",
        2: "RUNNING",
        3: "HANDS",
        4: "SITTING",
    }

    def __init__(self, split, regenerate=False, seed=123):
        super(Dataset_Rev1, self).__init__()
        self.split = split
        self.rng = np.random.default_rng(seed=seed)

        def is_in_train_split(filename):
            # da 1 a 23 e da 39 a 45
            idx = int(filename.split("_")[1])
            return (idx >= 1 and idx <= 23) or (idx >= 39 and idx <= 45)

        def is_in_valid_split(filename):
            # da 42 a 45
            # idx = int(filename.split("_")[1])
            # return idx >= 42 and idx <= 45
            return False

        def is_in_test_split(filename):
            # da 24 a 38
            idx = int(filename.split("_")[1])
            return idx >= 24 and idx <= 38

        if self.split == "train":
            self.raw_filenames = [
                f for f in os.listdir(hpm.RAW_REV1_DATA_PATH) if is_in_train_split(f)
            ]
            self.directory = os.path.join(hpm.REV1_DATA_PATH, "train")
        elif self.split == "valid":
            self.raw_filenames = [
                f for f in os.listdir(hpm.RAW_REV1_DATA_PATH) if is_in_valid_split(f)
            ]
            self.directory = os.path.join(hpm.REV1_DATA_PATH, "valid")
        elif self.split == "test":
            self.raw_filenames = [
                f for f in os.listdir(hpm.RAW_REV1_DATA_PATH) if is_in_test_split(f)
            ]
            self.directory = os.path.join(hpm.REV1_DATA_PATH, "test")

        else:
            raise Exception(f"Invalid split {split}")

        # generate empty directory
        os.makedirs(self.directory, exist_ok=True)

        # Check if directoriy is empty, if it is, generates dataset
        if len(os.listdir(self.directory)) == 0:
            self.generate_dataset()
        else:
            if regenerate:
                # delete files in directory and regenerate
                for file in os.listdir(self.directory):
                    os.remove(os.path.join(self.directory, file))

                self.generate_dataset()

        self.filenames = os.listdir(self.directory)

    def generate_dataset(self):
        """
        Takes all raw data and generates the tuples for the __get_item__ method reads to the self.directory.
        """

        print(f"Generating {self.split} set...")
        for fname in self.raw_filenames:
            fpath = os.path.join(hpm.RAW_REV1_DATA_PATH, fname)
            act_id = int(fname.split("_")[0][4:])
            act = Dataset_Rev1.activity_dict[act_id]
            seq_id = fname.split("_")[1] + "-" + \
                fname.split("_")[-1].split(".")[0]

            with open(fpath, "rb") as file:
                raw_seq = pickle.load(file)

            # pad with zeros all elements of raw_seq to make all sequences have the shape (64, 8)
            for i in range(len(raw_seq)):
                rbins = raw_seq[i].shape[1]
                raw_seq[i] = np.pad(
                    raw_seq[i], ((0, 0), (0, 8 - rbins)), "constant")

            raw_seq = np.concatenate(raw_seq, axis=0)
            complex_cir = np.transpose(raw_seq, (1, 0))

            # normalize cir
            complex_cir -= complex_cir.mean(1, keepdims=True)

            (chunks, IHT_output, full_IHT_mD,) = sparsity_based.mD_spectrum_(
                complex_cir,
                hpm.DATAGEN_PARAMS["NWIN"],
                hpm.DATAGEN_PARAMS["TREP"],
                n_kept_bins=8,
            )

            data_tuple = (chunks, IHT_output, full_IHT_mD)

            # save data tuple
            out_fname = f"0_0_{act}_{seq_id}.obj"
            out_fpath = os.path.join(self.directory, out_fname)
            with open(out_fpath, "wb") as out:
                pickle.dump(data_tuple, out)

    def generate_mask(
        self,
        size,
        p_remove,
        min_allowed_samples=3,
    ):
        if p_remove == 0:
            return torch.ones(size * 2).to(hpm.DEVICE)

        # Apply uniform random sampling pattern
        chunk_mask = (torch.rand(size) > p_remove).int().to(hpm.DEVICE)

        # ensure mask has at least 3 non zero elements
        nonzeros = torch.sum(chunk_mask)
        if nonzeros < min_allowed_samples:
            # choose 3 random indices to not mask
            chunk_mask = torch.zeros(size).int().to(hpm.DEVICE)
            idxs = self.rng.choice(
                np.arange(size),
                min_allowed_samples,
                replace=False,
            )
            chunk_mask[idxs] = 1

        # repeat mask two times
        chunk_mask = chunk_mask.repeat(2)
        return chunk_mask

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Dataset must return (X, IHT_output, mD_columns)
        """
        with open(os.path.join(self.directory, self.filenames[idx]), "rb") as file:
            Xyz = pickle.load(file)

        # Convert to real numbers
        X = complex_to_real_vector(Xyz[0])
        IHT_output = torch.tensor(complex_to_real_vector(Xyz[1]))
        mD_columns = torch.tensor(Xyz[2])

        X = torch.clamp(torch.tensor(X), min=-150, max=150)

        return X, IHT_output, mD_columns


class Sparse_MD_Dataset(torch.utils.data.Dataset):

    def __init__(self, filenames, p_burst=0, seed=0):
        super(Sparse_MD_Dataset, self).__init__()
        self.raw_filenames = filenames
        self.rng = np.random.default_rng(seed=seed)
        self.p_burst = p_burst

    @staticmethod
    def generate_dataset(out_folder=hpm.DATA_PATH_V2):
        _ = input(
            "You are about to regenerate the whole dataset! \n If you are sure, press enter to continue."
        )

        os.makedirs(out_folder, exist_ok=True)

        for pass_idx in range(hpm.DATAGEN_PARAMS["N_PASSES"]):
            for n in hpm.SUBJECTS:
                for activity in hpm.ACTIVITIES:
                    for f in os.listdir(f"{hpm.RAW_DATA_PATH}/PERSON{n}"):
                        activity_idx = f.split(".")[0].split("_")[-1]
                        if activity in f:

                            complex_cir = get_cir(
                                f"{hpm.RAW_DATA_PATH}/PERSON{n}/{f}",
                                hpm.DATAGEN_PARAMS["DIST_BOUNDS"],
                            )

                            complex_cir = complex_cir[
                                :, :, hpm.DATAGEN_PARAMS["BP_SEL"]
                            ]
                            complex_cir -= complex_cir.mean(1, keepdims=True)

                            (
                                chunks,
                                IHT_output,
                                full_IHT_mD,
                            ) = sparsity_based.mD_spectrum_(
                                complex_cir,
                                hpm.DATAGEN_PARAMS["NWIN"],
                                hpm.DATAGEN_PARAMS["TREP"],
                                n_kept_bins=hpm.N_KEPT_BINS,
                            )

                            data_tuple = (chunks, IHT_output, full_IHT_mD)

                            # Now I have my couples (X, y) = (chunks, mD) inside a big list
                            # Save them one by one as a tuple (chunk, window_mD)

                            # Naming convention: {PASS_INDEX}_{SUBJECT}_{ACTIVITY}_{ACTIVITY_INDEX}
                            with open(
                                f"{out_folder}/{pass_idx}_{n}_{activity}_{activity_idx}.obj",
                                "wb",
                            ) as out:
                                pickle.dump(data_tuple, out)

                            print(f"Saved {f}")

    @staticmethod
    def make_splits(
        subjects,
        activities,
        sparse_dataset_path=hpm.DATA_PATH_V2,
        subsample_factor=1.0,
        seed=123,
        train=0.8,
        test=0.1,
        valid=0.1,
    ):
        if not train + valid + test == 1.0:
            raise Exception(
                f"train ({train})+valid ({valid})+test ({test}) != 1")

        if subsample_factor < 0 or subsample_factor > 1:
            raise Exception(
                f"Please input a valid subsampling factor (Current is {subsample_factor})"
            )

        rng = np.random.default_rng(seed=seed)
        all_filenames = os.listdir(sparse_dataset_path)

        # Select only wanted subjects
        all_filenames = [
            f for f in all_filenames if int(get_subj_from_filename(f)) in subjects
        ]

        # Select only wanted activities
        all_filenames = [f for f in all_filenames if any(
            [a in f for a in activities])]

        nsamples = int(len(all_filenames) * subsample_factor)
        chosen_filenames = rng.choice(all_filenames, nsamples, replace=False)

        train_set, valid_test_set = train_test_split(
            chosen_filenames, train_size=train, random_state=seed
        )

        valid_set, test_set = train_test_split(
            valid_test_set,
            train_size=(valid / (valid + test)),
            random_state=seed,
        )

        # Repeat all the filenames in the training set containing the "RUNNING" activity
        # until they reach the same amount of samples as the "WALKING" activity
        # This is done to balance the dataset
        train_set = train_set.tolist()
        running_filenames = [f for f in train_set if "RUNNING" in f]
        walking_filenames = [f for f in train_set if "WALKING" in f]

        for _ in range(len(walking_filenames) - len(running_filenames)):
            train_set.append(rng.choice(running_filenames))

        train_set = np.array(train_set)
        return train_set, valid_set, test_set

    def generate_mask(
        self,
        size,
        p_remove,
        min_allowed_samples=3,
    ):
        if p_remove == 0:
            return torch.ones(size * 2).to(hpm.DEVICE)

        if self.rng.random() < self.p_burst:
            # Apply "bursting" sampling pattern
            chunk_mask = torch.zeros(size).to(hpm.DEVICE)

            burst_len1 = (torch.rand(size) > p_remove).int().sum()
            if burst_len1 < min_allowed_samples:
                burst_len1 = min_allowed_samples
            if burst_len1 == size:
                return torch.ones(size * 2).to(hpm.DEVICE)

            # select first start index for the burst
            start_idx1 = self.rng.integers(0, 64 - burst_len1)
            # select second start index for the burst
            chunk_mask[start_idx1: start_idx1 + burst_len1] = 1

            # if self.rng.random() < 0.2:
            #     # apply second burst with 20% probability

            #     candidates = [
            #         c
            #         for c in range(64 - burst_len2)
            #         if c < start_idx1 - burst_len1 - 1
            #         or c > start_idx1 + burst_len1 + 1
            #     ]
            #     start_idx2 = np.random.choice(candidates)
            #     chunk_mask[start_idx2 : start_idx2 + burst_len2] = 1
            assert torch.sum(chunk_mask) >= min_allowed_samples
        else:
            # Apply uniform random sampling pattern
            chunk_mask = (torch.rand(size) > p_remove).int().to(hpm.DEVICE)

            # ensure mask has at least 3 non zero elements
            nonzeros = torch.sum(chunk_mask)
            if nonzeros < min_allowed_samples:
                # choose 3 random indices to not mask
                chunk_mask = torch.zeros(size).int().to(hpm.DEVICE)
                idxs = self.rng.choice(
                    np.arange(size),
                    min_allowed_samples,
                    replace=False,
                )
                chunk_mask[idxs] = 1

        # repeat mask two times
        chunk_mask = chunk_mask.repeat(2)
        return chunk_mask

    def __len__(self):
        return self.raw_filenames.shape[0]

    def __getitem__(self, idx):
        with open(
            os.path.join(hpm.DATA_PATH_V2, self.raw_filenames[idx]),
            "rb",
        ) as file:
            Xy = pickle.load(file)

        # Convert to real numbers
        X = complex_to_real_vector(Xy[0])
        IHT_output = torch.tensor(complex_to_real_vector(Xy[1]))
        mD_columns = torch.tensor(Xy[2])

        X = torch.clamp(torch.tensor(X), min=-150, max=150)

        return X, IHT_output, mD_columns


if __name__ == "__main__":
    # Example of usage
    # 1) Generate all samples from raw data
    # NB: It's a long process! Do it only once :)
    cfg = hpm.CONFIG

    Sparse_MD_Dataset.generate_dataset()
    Dataset_Rev1.generate_dataset()
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
    test_set = Sparse_MD_Dataset(test_filenames, p_burst=hpm.P_BURST)

    # iterate over each set and save length of samples in a list
    train_set_lengths = []
    for i in range(len(train_set)):
        train_set_lengths.append(len(train_set[i][0]))

    valid_set_lengths = []
    for i in range(len(valid_set)):
        valid_set_lengths.append(len(valid_set[i][0]))

    random_test_set_lengths = []
    for i in range(len(test_set)):
        random_test_set_lengths.append(len(test_set[i][0]))

    # aggregate all lengths in a single list
    all_lengths = train_set_lengths + valid_set_lengths + random_test_set_lengths
    # print min and max length

    def get_len_in_seconds(n_windows, window_len, step_size):
        sample_len = 0.27  # milliseconds
        tot_samples = (n_windows - 1) * step_size + window_len
        return (sample_len * tot_samples) / 1000

    # print min and max length in seconds
    # print("Min length: ", get_len_in_seconds(min(all_lengths), 64, 32))
    # print("Max length: ", get_len_in_seconds(max(all_lengths), 64, 32))
    print(np.sum(valid_set_lengths))

    pass
