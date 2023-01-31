import os
import sys
sys.path.append(".")

import numpy as np
import scipy.stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import torch
import torch.utils.data
from tqdm import tqdm

from utils import predict_multiple, CIFAR10NP, TRANSFORM


def calculate_entscore(dataloader, model, device):
    # return the entropy score
    ent_scores = []
    for imgs, labels in iter(dataloader):
        imgs, labels = imgs.to(device), labels.to(device)
        _, prob = predict_multiple(model, imgs)
        ent_scores.extend(scipy.stats.entropy(prob, axis=1).tolist()) 
    return np.mean(ent_scores)


if __name__ == "__main__":
    # paths
    dataset_path = "/data/lengx/cifar/"
    train_set = "train_data"
    val_sets = sorted(["cifar10-f-32", "cifar-10.1-c", "cifar-10.1"])
    model_name = sys.argv[1]
    temp_file_path = f"../temp/{model_name}/entscore/"

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

    # need to do entropy score calculation
    if not os.path.exists(temp_file_path) or not os.path.exists(f"{temp_file_path}{train_set}.npy"):
        if not os.path.exists(temp_file_path):
            os.mkdir(temp_file_path)

        # training set calculation
        train_path = f"{dataset_path}{train_set}"
        train_candidates = []
        for file in sorted(os.listdir(train_path)):
            if file.endswith(".npy") and file.startswith("new_data"):
                train_candidates.append(file)
        
        entscores = np.zeros(len(train_candidates))
        print(f"===> Calculating average entropy score for {train_set}")        

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
            entscores[i] = calculate_entscore(dataloader, model, device)

        np.save(f"{temp_file_path}{train_set}.npy", entscores)

    if not os.path.exists(f"{temp_file_path}val_sets.npy"):
        # validation set calculation
        val_candidates = []
        val_paths = [f"{dataset_path}{set_name}" for set_name in val_sets]
        for val_path in val_paths:
            for file in sorted(os.listdir(val_path)):
                val_candidates.append(f"{val_path}/{file}")
        
        entscores = np.zeros(len(val_candidates))
        print(f"===> Calculating average entropy score for validation sets")        

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
            entscores[i] = calculate_entscore(dataloader, model, device)

        np.save(f"{temp_file_path}val_sets.npy", entscores)
    
    # if the calculation of average entropy score is finished
    # calculate the linear regression model (accuracy in %)
    print(f"===> Linear Regression model for entropy method with model: {model_name}")
    train_x = np.load(f"{temp_file_path}{train_set}.npy")
    train_y = np.load(f"../temp/{model_name}/acc/{train_set}.npy") * 100
    val_x = np.load(f"{temp_file_path}val_sets.npy")
    val_y = np.load(f"../temp/{model_name}/acc/val_sets.npy") * 100

    lr = LinearRegression()
    lr.fit(train_x.reshape(-1, 1), train_y)
    # predictions will have 6 decimals
    val_y_hat = np.round(lr.predict(val_x.reshape(-1, 1)), decimals=6)
    rmse_loss = mean_squared_error(y_true=val_y, y_pred=val_y_hat, squared=False)
    print(f"The RMSE on validation set is: {rmse_loss}")
