import torch
import numpy as np
from torch import nn
import os
import hyperparams_cnn as hpm_cnn
import hyperparams as hpm
from matplotlib import pyplot as plt

from cnn_dataset import CNN_Dataset
from evaluation import load_model_and_config
from models import init_model


def conv_layer(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )


class CNN_Baseline(nn.Module):
    def __init__(self, blocks_list, num_classes=4):
        super(CNN_Baseline, self).__init__()

        self.blocks_list = blocks_list
        self.activation = nn.LeakyReLU(inplace=True)

        # # BN layers
        # self.bn_layers = [
        #     nn.BatchNorm2d(blocks_list[k][1]).to(hyperparams.DEVICE)
        #     for k in range(len(self.blocks_list))
        # ]

        # conv_layers
        self.conv_layers = [
            conv_layer(blocks_list[k][0], blocks_list[k][1], blocks_list[k][2]).to(
                hpm_cnn.DEVICE
            )
            for k in range(len(self.blocks_list))
        ]

        self.drop1 = nn.Dropout(p=0.2)

        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        for k in range(len(self.blocks_list)):
            x = self.conv_layers[k](x)
            x = self.activation(x)
            # x = self.bn_layers[k](x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.drop1(x)
        out = self.fc2(x)
        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_cnn(LIHT_model, cfg, p_remove, generate_data=True):
    print("Training CNN...")
    blocks = [
        (1, 8, 2),
        (8, 16, 2),
        (16, 32, 2),
        (32, 64, 2),
        (64, 128, 2),
        (128, 128, 2),
    ]
    # dummy_data = torch.zeros(32, 1, 64, 200).to(hyperparams.DEVICE)
    model = CNN_Baseline(blocks).to(hpm_cnn.DEVICE)

    print(count_parameters(model))

    # print(y.shape)

    # 3 - Define Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hpm_cnn.ADAM_LR, weight_decay=1e-6
    )
    cross_entropy = nn.CrossEntropyLoss()

    # 4 - Create PyTorch Dataset and Dataloader
    if generate_data:
        CNN_Dataset.generate_splits(LIHT_model, cfg, p_remove=p_remove)

    cnn_train_set = CNN_Dataset(os.path.join(hpm.CNN_DATA_PATH, "train"))
    cnn_valid_set = CNN_Dataset(os.path.join(hpm.CNN_DATA_PATH, "valid"))
    cnn_test_set_IHT = CNN_Dataset(os.path.join(hpm.CNN_DATA_PATH, "test_IHT"))
    cnn_test_set_LIHT = CNN_Dataset(
        os.path.join(hpm.CNN_DATA_PATH, "test_LIHT"))

    # 4) Create Dataloaders for batching and shuffling
    train_dataloader = torch.utils.data.DataLoader(
        cnn_train_set, batch_size=hpm_cnn.BATCH_SIZE, shuffle=True, drop_last=False
    )
    valid_dataloader = torch.utils.data.DataLoader(
        cnn_valid_set, batch_size=hpm_cnn.BATCH_SIZE, shuffle=True, drop_last=False
    )
    test_dataloader_IHT = torch.utils.data.DataLoader(
        cnn_test_set_IHT, batch_size=hpm_cnn.BATCH_SIZE, shuffle=False, drop_last=False
    )

    test_dataloader_LIHT = torch.utils.data.DataLoader(
        cnn_test_set_LIHT, batch_size=hpm_cnn.BATCH_SIZE, shuffle=False, drop_last=False
    )

    cnn_config = hpm_cnn.CONFIG
    train_loss, valid_loss, train_acc, valid_acc = [], [], [], []
    for ep in range(cnn_config["EPOCHS"]):
        if ep % 20 == 0:
            print(f"-- Epoch : {ep} ---")
        epoch_train_losses, epoch_train_accs = [], []
        model.train()
        for i, (X, y_gt) in enumerate(train_dataloader):

            X = X.to(cnn_config["DEVICE"])
            y_gt = y_gt.to(cnn_config["DEVICE"])

            # Forward pass
            y_pred = model(X)

            # Compute loss
            loss = cross_entropy(y_pred, y_gt).float()
            epoch_train_losses.append(loss.item())

            # Compute gradient
            loss.backward()

            # Optimization Step
            optimizer.step()

            # Reset gradients
            optimizer.zero_grad()

            _, pred_class = torch.max(y_pred, axis=1)
            acc = torch.sum(pred_class == y_gt) / len(pred_class)
            epoch_train_accs.append(acc)

            # if i % 10 == 0:
            #     print(f"Batch {i}: Train Loss = {loss.item():.3f}")

        # Evaluation at the end of the epoch
        epoch_valid_losses, epoch_valid_accs = [], []
        with torch.no_grad():
            model.eval()
            for i, (X_valid, y_gt_valid) in enumerate(valid_dataloader):
                X_valid = X_valid.to(cnn_config["DEVICE"])
                y_gt_valid = y_gt_valid.to(cnn_config["DEVICE"])

                # Forward pass
                y_pred_valid = model(X_valid)

                # Compute loss
                loss_val = cross_entropy(y_pred_valid, y_gt_valid).float()
                epoch_valid_losses.append(loss_val.item())

                _, pred_val_class = torch.max(y_pred_valid, axis=1)
                val_acc = torch.sum(
                    pred_val_class == y_gt_valid) / len(pred_val_class)
                epoch_valid_accs.append(val_acc)

        mean_train_loss = torch.mean(torch.tensor(epoch_train_losses))
        mean_valid_loss = torch.mean(torch.tensor(epoch_valid_losses))
        mean_train_acc = torch.mean(torch.tensor(epoch_train_accs))
        mean_valid_acc = torch.mean(torch.tensor(epoch_valid_accs))

        train_loss.append(mean_train_loss)
        valid_loss.append(mean_valid_loss)
        train_acc.append(mean_train_acc)
        valid_acc.append(mean_valid_acc)

    test_accs_IHT = []
    predictions_IHT = []
    labels = []
    with torch.no_grad():
        for i, (X_test, y_gt_test) in enumerate(test_dataloader_IHT):
            X_test = X_test.to(cnn_config["DEVICE"])
            y_gt_test = y_gt_test.to(cnn_config["DEVICE"])
            labels.append(y_gt_test)

            # Forward pass
            y_pred_test = model(X_test)
            predictions_IHT.append(y_pred_test)

            _, pred_test_class = torch.max(y_pred_test, axis=1)
            test_acc = torch.sum(
                pred_test_class == y_gt_test) / len(pred_test_class)
            test_accs_IHT.append(test_acc.cpu().numpy())

    print(f"Test accuracy IHT = {np.mean(test_accs_IHT)}")

    test_accs_LIHT = []
    predictions_LIHT = []
    with torch.no_grad():
        for i, (X_test, y_gt_test) in enumerate(test_dataloader_LIHT):
            X_test = X_test.to(cnn_config["DEVICE"])
            y_gt_test = y_gt_test.to(cnn_config["DEVICE"])

            # Forward pass
            y_pred_test = model(X_test)
            predictions_LIHT.append(y_pred_test)

            _, pred_test_class = torch.max(y_pred_test, axis=1)
            test_acc = torch.sum(
                pred_test_class == y_gt_test) / len(pred_test_class)
            test_accs_LIHT.append(test_acc.cpu().numpy())

    print(f"Test accuracy LIHT = {np.mean(test_accs_LIHT)}")

    preds_IHT = torch.concat(predictions_IHT, axis=0)
    _, preds_IHT = torch.max(preds_IHT, axis=1)
    preds_LIHT = torch.concat(predictions_LIHT, axis=0)
    _, preds_LIHT = torch.max(preds_LIHT, axis=1)
    labels = torch.concat(labels, axis=0)

    out_dict = {}
    out_dict["preds_IHT"] = preds_IHT.cpu().tolist()
    out_dict["preds_LIHT"] = preds_LIHT.cpu().tolist()
    out_dict["labels"] = labels.cpu().tolist()

    return out_dict


def train_cnn_baselines(
    train_dataset,
    valid_dataset,
    test_dataset_50,
    test_dataset_75,
    test_dataset_90,
    out_model_path,
):
    """
    Trains the CNN given a train and validation dataset.
    """
    print("Training CNN...")
    blocks = [
        (1, 8, 2),
        (8, 16, 2),
        (16, 32, 2),
        (32, 64, 2),
        (64, 128, 2),
        (128, 128, 2),
    ]
    model = CNN_Baseline(blocks).to(hpm_cnn.DEVICE)

    print(count_parameters(model))

    # 3 - Define Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hpm_cnn.ADAM_LR, weight_decay=1e-6
    )
    cross_entropy = nn.CrossEntropyLoss()

    # 4) Create Dataloaders for batching and shuffling
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=hpm_cnn.BATCH_SIZE, shuffle=True, drop_last=False
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=hpm_cnn.BATCH_SIZE, shuffle=True, drop_last=False
    )

    test_dataloader_50 = torch.utils.data.DataLoader(
        test_dataset_50, batch_size=hpm_cnn.BATCH_SIZE, shuffle=False, drop_last=False
    )
    test_dataloader_75 = torch.utils.data.DataLoader(
        test_dataset_75, batch_size=hpm_cnn.BATCH_SIZE, shuffle=False, drop_last=False
    )
    test_dataloader_90 = torch.utils.data.DataLoader(
        test_dataset_90, batch_size=hpm_cnn.BATCH_SIZE, shuffle=False, drop_last=False
    )

    cnn_config = hpm_cnn.CONFIG
    train_loss, valid_loss, train_acc, valid_acc = [], [], [], []
    for ep in range(cnn_config["EPOCHS"]):
        if ep % 20 == 0:
            print(f"-- Epoch : {ep} ---")
        epoch_train_losses, epoch_train_accs = [], []
        model.train()
        for i, (X, y_gt) in enumerate(train_dataloader):

            X = X.to(cnn_config["DEVICE"])
            y_gt = y_gt.to(cnn_config["DEVICE"])

            # Forward pass
            y_pred = model(X)

            # Compute loss
            loss = cross_entropy(y_pred, y_gt).float()
            epoch_train_losses.append(loss.item())

            # Compute gradient
            loss.backward()

            # Optimization Step
            optimizer.step()

            # Reset gradients
            optimizer.zero_grad()

            _, pred_class = torch.max(y_pred, axis=1)
            acc = torch.sum(pred_class == y_gt) / len(pred_class)
            epoch_train_accs.append(acc)

        # Evaluation at the end of the epoch
        epoch_valid_losses, epoch_valid_accs = [], []
        with torch.no_grad():
            model.eval()
            for i, (X_valid, y_gt_valid) in enumerate(valid_dataloader):
                X_valid = X_valid.to(cnn_config["DEVICE"])
                y_gt_valid = y_gt_valid.to(cnn_config["DEVICE"])

                # Forward pass
                y_pred_valid = model(X_valid)

                # Compute loss
                loss_val = cross_entropy(y_pred_valid, y_gt_valid).float()
                epoch_valid_losses.append(loss_val.item())

                _, pred_val_class = torch.max(y_pred_valid, axis=1)
                val_acc = torch.sum(
                    pred_val_class == y_gt_valid) / len(pred_val_class)
                epoch_valid_accs.append(val_acc)

        mean_train_loss = torch.mean(torch.tensor(epoch_train_losses))
        mean_valid_loss = torch.mean(torch.tensor(epoch_valid_losses))
        mean_train_acc = torch.mean(torch.tensor(epoch_train_accs))
        mean_valid_acc = torch.mean(torch.tensor(epoch_valid_accs))

        train_loss.append(mean_train_loss)
        valid_loss.append(mean_valid_loss)
        train_acc.append(mean_train_acc)
        valid_acc.append(mean_valid_acc)

    # Save model weights
    torch.save(model.state_dict(), out_model_path)

    # Test sets evaluation
    predictions_50 = []
    predictions_75 = []
    predictions_90 = []
    labels = []
    model.eval()

    with torch.no_grad():
        for i, (X, y_gt) in enumerate(test_dataloader_50):
            X = X.to(cnn_config["DEVICE"])
            y_gt = y_gt.to(cnn_config["DEVICE"])

            # Forward pass
            y_pred = model(X)

            _, pred_class = torch.max(y_pred, axis=1)
            predictions_50.append(pred_class)
            labels.append(y_gt)

        for i, (X, y_gt) in enumerate(test_dataloader_75):
            X = X.to(cnn_config["DEVICE"])
            y_gt = y_gt.to(cnn_config["DEVICE"])

            # Forward pass
            y_pred = model(X)

            _, pred_class = torch.max(y_pred, axis=1)
            predictions_75.append(pred_class)

        for i, (X, y_gt) in enumerate(test_dataloader_90):
            X = X.to(cnn_config["DEVICE"])
            y_gt = y_gt.to(cnn_config["DEVICE"])

            # Forward pass
            y_pred = model(X)
            _, pred_class = torch.max(y_pred, axis=1)
            predictions_90.append(pred_class)

    predictions_50 = torch.concat(predictions_50, axis=0)
    predictions_75 = torch.concat(predictions_75, axis=0)
    predictions_90 = torch.concat(predictions_90, axis=0)
    labels = torch.concat(labels, axis=0)

    # Save model weights

    return (
        (train_loss, valid_loss, train_acc, valid_acc),
        predictions_50,
        predictions_75,
        predictions_90,
        labels,
    )


def train_cnn_testRoom(
    train_dataset,
    valid_dataset,
    test_dataset,
    test_dataset_90,
    out_model_path,
):

    print("Training CNN...")
    blocks = [
        (1, 8, 2),
        (8, 16, 2),
        (16, 32, 2),
        (32, 64, 2),
        (64, 128, 2),
        (128, 128, 2),
    ]
    model = CNN_Baseline(blocks).to(hpm_cnn.DEVICE)

    print(count_parameters(model))

    # 3 - Define Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hpm_cnn.ADAM_LR, weight_decay=1e-6
    )
    cross_entropy = nn.CrossEntropyLoss()

    # 4) Create Dataloaders for batching and shuffling
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=hpm_cnn.BATCH_SIZE, shuffle=True, drop_last=False
    )
    if len(valid_dataset) > 0:

        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=hpm_cnn.BATCH_SIZE, shuffle=True, drop_last=False
        )

    test_dataloader_90 = torch.utils.data.DataLoader(
        test_dataset_90, batch_size=hpm_cnn.BATCH_SIZE, shuffle=False, drop_last=False
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=hpm_cnn.BATCH_SIZE, shuffle=False, drop_last=False
    )

    cnn_config = hpm_cnn.CONFIG
    train_loss, valid_loss, train_acc, valid_acc = [], [], [], []
    for ep in range(cnn_config["EPOCHS"]):
        if ep % 20 == 0:
            print(f"-- Epoch : {ep} ---")
        epoch_train_losses, epoch_train_accs = [], []
        model.train()
        for i, (X, y_gt) in enumerate(train_dataloader):

            X = X.to(cnn_config["DEVICE"])
            y_gt = y_gt.to(cnn_config["DEVICE"])

            # Forward pass
            y_pred = model(X)

            # Compute loss
            loss = cross_entropy(y_pred, y_gt).float()
            epoch_train_losses.append(loss.item())

            # Compute gradient
            loss.backward()

            # Optimization Step
            optimizer.step()

            # Reset gradients
            optimizer.zero_grad()

            _, pred_class = torch.max(y_pred, axis=1)
            acc = torch.sum(pred_class == y_gt) / len(pred_class)
            epoch_train_accs.append(acc)

        # Evaluation at the end of the epoch
        if len(valid_dataset) > 0:
            epoch_valid_losses, epoch_valid_accs = [], []
            with torch.no_grad():
                model.eval()
                for i, (X_valid, y_gt_valid) in enumerate(valid_dataloader):
                    X_valid = X_valid.to(cnn_config["DEVICE"])
                    y_gt_valid = y_gt_valid.to(cnn_config["DEVICE"])

                    # Forward pass
                    y_pred_valid = model(X_valid)

                    # Compute loss
                    loss_val = cross_entropy(y_pred_valid, y_gt_valid).float()
                    epoch_valid_losses.append(loss_val.item())

                    _, pred_val_class = torch.max(y_pred_valid, axis=1)
                    val_acc = torch.sum(pred_val_class == y_gt_valid) / len(
                        pred_val_class
                    )
                    epoch_valid_accs.append(val_acc)

            mean_valid_loss = torch.mean(torch.tensor(epoch_valid_losses))
            mean_valid_acc = torch.mean(torch.tensor(epoch_valid_accs))
            valid_loss.append(mean_valid_loss)
            valid_acc.append(mean_valid_acc)

        mean_train_loss = torch.mean(torch.tensor(epoch_train_losses))
        mean_train_acc = torch.mean(torch.tensor(epoch_train_accs))
        train_acc.append(mean_train_acc)
        train_loss.append(mean_train_loss)

    # Save model weights
    torch.save(model.state_dict(), out_model_path)

    # Test set evaluation
    predictions_90 = []
    predictions = []
    labels90 = []
    labels = []
    model.eval()

    with torch.no_grad():

        for _, (X, y_gt) in enumerate(test_dataloader_90):
            X = X.to(cnn_config["DEVICE"])
            y_gt = y_gt.to(cnn_config["DEVICE"])

            # Forward pass
            y_pred = model(X)
            labels90.append(y_gt)
            _, pred_class = torch.max(y_pred, axis=1)
            predictions_90.append(pred_class)

        for _, (X, y_gt) in enumerate(test_dataloader):
            X = X.to(cnn_config["DEVICE"])
            y_gt = y_gt.to(cnn_config["DEVICE"])

            # Forward pass
            y_pred = model(X)
            labels.append(y_gt)
            _, pred_class = torch.max(y_pred, axis=1)
            predictions.append(pred_class)

    predictions_90 = torch.concat(predictions_90, axis=0)
    predictions = torch.concat(predictions, axis=0)
    labels = torch.concat(labels, axis=0)
    labels90 = torch.concat(labels90, axis=0)

    # Save model weights

    return (
        (train_loss, valid_loss, train_acc, valid_acc),
        predictions,
        predictions_90,
        labels,
        labels90,
    )


def test_cnn_baselines(cnn_model_path, test_dataset):

    print("Testing CNN...")
    blocks = [
        (1, 8, 2),
        (8, 16, 2),
        (16, 32, 2),
        (32, 64, 2),
        (64, 128, 2),
        (128, 128, 2),
    ]
    model = CNN_Baseline(blocks).to(hpm_cnn.DEVICE)

    # load weights
    model.load_state_dict(torch.load(cnn_model_path))

    print(count_parameters(model))

    # 4) Create Dataloaders for batching and shuffling
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=hpm_cnn.BATCH_SIZE, shuffle=True, drop_last=False
    )

    cnn_config = hpm_cnn.CONFIG
    predictions = []
    labels = []
    with torch.no_grad():
        model.eval()
        for i, (X, y_gt) in enumerate(test_dataloader):

            X = X.to(cnn_config["DEVICE"])
            y_gt = y_gt.to(cnn_config["DEVICE"])

            # Forward pass
            y_pred = model(X)

            _, pred_class = torch.max(y_pred, axis=1)
            predictions.append(pred_class)
            labels.append(y_gt)

    preds = torch.concat(predictions, axis=0)
    labels = torch.concat(labels, axis=0)

    return preds, labels


if __name__ == "__main__":
    from utils import parse_grid_search_csv
    import json
    import pickle

    run_names = [
        "STAR_final",
        # "DUST_V2",
        # "Ablation_NoAtt",
        # "Ablation_3Windows",
        # "Ablation_1Window",
        # "Ablation_LearnS",
        # "Ablation_OnlyAdd",
    ]
    path = "./models/STAR_GS/"

    out_dir = "./results/cnn_ablation/"

    for run_idx in range(len(run_names)):
        out_dict = {}
        run_name = run_names[run_idx]
        config_path = os.path.join(path, f"{run_name}_config.pkl")

        with open(config_path, "rb") as f:
            config = pickle.load(f)

        model = init_model(config)
        model_weights = torch.load(os.path.join(path, run_name + ".pt"))

        model.load_state_dict(model_weights)

        ps_remove = [0.9]
        for p_remove in ps_remove:

            cnn_results = train_cnn(
                model, config, p_remove=p_remove, generate_data=True
            )

            # Save cnn_results as json
            out_path = os.path.join(
                out_dir, f"{run_name}_results_{p_remove:.2f}.json")
            with open(out_path, "w") as f:
                json.dump(cnn_results, f)
