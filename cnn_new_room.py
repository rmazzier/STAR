import os
import pickle
import matplotlib.pyplot as plt

from cnn_dataset import CNN_Dataset
from base_cnn import train_cnn_testRoom
from sparse_dataset_gen import Dataset_Rev1
import hyperparams as hpm


if __name__ == "__main__":

    train_set = Dataset_Rev1(split="train", regenerate=False)
    valid_set = Dataset_Rev1(split="valid", regenerate=False)
    test_set = Dataset_Rev1(split="test", regenerate=False)

    star_preds_path = os.path.join(
        "results", "Dataset_Rev1", "STAR_testroom_preds_finetune"
    )

    cnn_train_set_path = os.path.join("data", "cnn_dataset", "newDataset_rev1", "train")
    # cnn_train_set_path = os.path.join(hpm.CNN_DATA_PATH, "train")
    cnn_valid_set_path = os.path.join("data", "cnn_dataset", "newDataset_rev1", "valid")
    cnn_test_set_path = os.path.join("data", "cnn_dataset", "newDataset_rev1", "test")
    cnn_test_set_path_90 = os.path.join(
        "data", "cnn_dataset", "newDataset_rev1", "test90"
    )

    # ONLY TO GENERATE CNN DATASET
    CNN_Dataset.generate_dataset_rev1_train(train_set, cnn_train_set_path)
    CNN_Dataset.generate_dataset_rev1_train(valid_set, cnn_valid_set_path)
    CNN_Dataset.generate_dataset_rev1_train(test_set, cnn_test_set_path)
    CNN_Dataset.generate_dataset_rev1_test(star_preds_path, cnn_test_set_path_90)

    cnn_train_set = CNN_Dataset(cnn_train_set_path)
    cnn_valid_set = CNN_Dataset(cnn_valid_set_path)
    cnn_test_set = CNN_Dataset(cnn_test_set_path)
    cnn_test_set90 = CNN_Dataset(cnn_test_set_path_90)

    cnn_model_path = os.path.join("results", "Dataset_Rev1", "cnn_model.pt")

    (
        training_stats,
        test_preds,
        test_preds_90,
        test_labels,
        test_labels90,
    ) = train_cnn_testRoom(
        cnn_train_set,
        cnn_valid_set,
        cnn_test_set,
        cnn_test_set90,
        out_model_path=cnn_model_path,
    )

    # save training stats as pickle
    with open(
        os.path.join("results", "Dataset_Rev1", f"cnn_training_stats.pkl"),
        "wb",
    ) as f:
        pickle.dump(training_stats, f)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].plot(training_stats[0], label="Train loss")
    ax[0].plot(training_stats[1], label="Valid loss")

    ax[1].plot(training_stats[2], label="Train accuracy")
    ax[1].plot(training_stats[3], label="Valid accuracy")
    plt.legend()
    plt.savefig(os.path.join("results", "Dataset_Rev1", f"cnn_train.png"))
    plt.close()

    # out_results_path = os.path.join("results", "Dataset_Rev1", f"cnn_results.pkl")
    out_results_path = os.path.join(
        "results", "Dataset_Rev1", f"cnn_results_finetuned.pkl"
    )

    cnn_predictions = {}

    cnn_predictions["0.9"] = {
        "predictions": test_preds,
        "predictions90": test_preds_90,
        "labels": test_labels,
        "labels90": test_labels90,
    }

    # save predictions as pickle
    with open(out_results_path, "wb") as f:
        pickle.dump(cnn_predictions, f)
    ## Update the plots
