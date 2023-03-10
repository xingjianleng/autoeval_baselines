import argparse
import os
import sys

sys.path.append(".")

import numpy as np
import scipy.stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import torch
import torch.utils.data
import torchvision.datasets
from tqdm import tqdm

from utils import predict_multiple, CIFAR10NP, TRANSFORM


parser = argparse.ArgumentParser(description="AutoEval baselines - ATC")
parser.add_argument(
    "--model", required=True, type=str, help="the model used to run this script"
)
parser.add_argument(
    "--dataset_path",
    required=True,
    type=str,
    help="path containing all datasets (training and validation)",
)


def calculate_threshold(acc, atcs):
    # This function is used to determine the threshold used
    # for the ATC method.
    sorted_atcs = np.sort(atcs)
    lower_tail_num = int(np.ceil(acc * len(atcs)))
    return sorted_atcs[lower_tail_num]


def calculate_atcs(dataloader, model, device):
    correct, atcs = [], []
    for imgs, labels in iter(dataloader):
        imgs, labels = imgs.to(device), labels.to(device)
        pred, prob = predict_multiple(model, imgs)
        correct.append(pred.squeeze(1).eq(labels).cpu())
        atcs.extend((-scipy.stats.entropy(prob, axis=1)).tolist())
    correct = torch.cat(correct).numpy()
    return np.mean(correct), np.array(atcs)


def calculate_atc_score(atcs, threshold):
    return np.mean(atcs < threshold)


if __name__ == "__main__":
    # paths
    args = parser.parse_args()
    dataset_path = args.dataset_path
    if not dataset_path or dataset_path[-1] != "/":
        dataset_path += "/"
    model_name = args.model
    train_set = "train_data"
    val_sets = sorted(["cifar10-f-32", "cifar-10.1-c", "cifar-10.1"])
    temp_file_path = f"../temp/{model_name}/atc/"

    batch_size = 500
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # load the model
    if model_name == "resnet":
        model = torch.hub.load(
            "chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True
        )
    elif model_name == "repvgg":
        model = torch.hub.load(
            "chenyaofo/pytorch-cifar-models", "cifar10_repvgg_a0", pretrained=True
        )
    else:
        raise ValueError("Unexpected model_name")
    model.to(device)
    model.eval()

    # Use original CIFAR10 test set to determine the threshold
    print("===> Calculating the threshold for ATC method")
    cifar_testloader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.CIFAR10(
            root=dataset_path,
            train=False,
            transform=TRANSFORM,
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    threshold = calculate_threshold(*calculate_atcs(cifar_testloader, model, device))

    # need to do atc calculation
    if not os.path.exists(temp_file_path) or not os.path.exists(
        f"{temp_file_path}{train_set}.npy"
    ):
        if not os.path.exists(temp_file_path):
            os.makedirs(temp_file_path)

        # training set calculation
        train_path = f"{dataset_path}{train_set}"
        train_candidates = []
        for file in sorted(os.listdir(train_path)):
            if file.endswith(".npy") and file.startswith("new_data"):
                train_candidates.append(file)

        atc_scores = np.zeros(len(train_candidates))
        print(f"===> Calculating ATC for {train_set}")

        for i, candidate in enumerate(tqdm(train_candidates)):
            data_path = f"{train_path}/{candidate}"
            label_path = f"{train_path}/labels.npy"

            dataloader = torch.utils.data.DataLoader(
                dataset=CIFAR10NP(
                    data_path=data_path,
                    label_path=label_path,
                    transform=TRANSFORM,
                ),
                batch_size=batch_size,
                shuffle=False,
            )
            _, atcs = calculate_atcs(dataloader, model, device)
            atc_scores[i] = calculate_atc_score(atcs, threshold)

        np.save(f"{temp_file_path}{train_set}.npy", atc_scores)

    if not os.path.exists(f"{temp_file_path}val_sets.npy"):
        # validation set calculation
        val_candidates = []
        val_paths = [f"{dataset_path}{set_name}" for set_name in val_sets]
        for val_path in val_paths:
            for file in sorted(os.listdir(val_path)):
                val_candidates.append(f"{val_path}/{file}")

        atc_scores = np.zeros(len(val_candidates))
        print(f"===> Calculating ATC for validation sets")

        for i, candidate in enumerate(tqdm(val_candidates)):
            data_path = f"{candidate}/data.npy"
            label_path = f"{candidate}/labels.npy"

            dataloader = torch.utils.data.DataLoader(
                dataset=CIFAR10NP(
                    data_path=data_path,
                    label_path=label_path,
                    transform=TRANSFORM,
                ),
                batch_size=batch_size,
                shuffle=False,
            )
            _, atcs = calculate_atcs(dataloader, model, device)
            atc_scores[i] = calculate_atc_score(atcs, threshold)

        np.save(f"{temp_file_path}val_sets.npy", atc_scores)

    # if the calculation of ATC is finished
    # calculate the linear regression model (accuracy in %)
    print(f"===> Linear Regression model for ATC method with model: {model_name}")
    train_x = np.load(f"{temp_file_path}{train_set}.npy") * 100
    train_y = np.load(f"../temp/{model_name}/acc/{train_set}.npy") * 100
    val_x = np.load(f"{temp_file_path}val_sets.npy") * 100
    val_y = np.load(f"../temp/{model_name}/acc/val_sets.npy") * 100

    lr = LinearRegression()
    lr.fit(train_x.reshape(-1, 1), train_y)
    # predictions will have 6 decimals
    val_y_hat = np.round(lr.predict(val_x.reshape(-1, 1)), decimals=6)
    rmse_loss = mean_squared_error(y_true=val_y, y_pred=val_y_hat, squared=False)
    print(f"The RMSE on validation set is: {rmse_loss}")
