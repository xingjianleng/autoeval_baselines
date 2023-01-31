import os
import sys
sys.path.append(".")

import numpy as np
import torch
from tqdm import tqdm

from utils import predict_multiple, CIFAR10NP, TRANSFORM


def calculate_acc(dataloader, model, device):
    # function for calculating the accuracy on a given dataset
    correct = []
    for imgs, labels in iter(dataloader):
        imgs, labels = imgs.to(device), labels.to(device)
        pred, _ = predict_multiple(model, imgs)
        correct.append(pred.squeeze(1).eq(labels).cpu())
    correct = torch.cat(correct).numpy()
    return np.mean(correct)


if __name__ == "__main__":
    # paths
    dataset_path = "/data/lengx/cifar/"
    train_set = "train_data"
    val_sets = sorted(["cifar10-f-32", "cifar-10.1-c", "cifar-10.1"])
    model_name = sys.argv[1]
    temp_file_path = f"../temp/{model_name}/acc/"

    batch_size = 500
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # load the model
    if model_name == "resnet":
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True)
    elif model_name == "repvgg":
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_repvgg_a0", pretrained=True)
    else:
        raise ValueError("Unexpected model_name")
    model.to(device)
    model.eval()

    # need to do accuracy calculation
    if not os.path.exists(temp_file_path) or not os.path.exists(f"{temp_file_path}{train_set}.npy"):
        if not os.path.exists(temp_file_path):
            os.mkdir(temp_file_path)

        # training set calculation
        train_path = f"{dataset_path}{train_set}"
        train_candidates = []
        for file in sorted(os.listdir(train_path)):
            if file.endswith(".npy") and file.startswith("new_data"):
                train_candidates.append(file)
        
        accuracies = np.zeros(len(train_candidates))
        print(f"===> Calculating accuracy for {train_set}")        

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
            accuracies[i] = calculate_acc(dataloader, model, device)

        accuracies = np.round(accuracies, decimals=6)
        np.save(f"{temp_file_path}{train_set}.npy", accuracies)

    if not os.path.exists(f"{temp_file_path}val_sets.npy"):
        # validation set calculation
        val_candidates = []
        val_paths = [f"{dataset_path}{set_name}" for set_name in val_sets]
        for val_path in val_paths:
            for file in sorted(os.listdir(val_path)):
                val_candidates.append(f"{val_path}/{file}")
        
        accuracies = np.zeros(len(val_candidates))
        print(f"===> Calculating accuracy for validation sets")        

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
            accuracies[i] = calculate_acc(dataloader, model, device)

        accuracies = np.round(accuracies, decimals=6)
        np.save(f"{temp_file_path}val_sets.npy", accuracies)
