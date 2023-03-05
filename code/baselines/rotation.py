import argparse
import os
import sys

sys.path.append(".")

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import torch
from tqdm import tqdm

from models.resnet_rotation import ResNetRotation
from models.repvgg_rotation import RepVGGRotation
from utils import CIFAR10NP, TRANSFORM


parser = argparse.ArgumentParser(description="AutoEval baselines - Rotation Prediction")
parser.add_argument(
    "--model", required=True, type=str, help="the model used to run this script"
)
parser.add_argument(
    "--dataset_path",
    required=True,
    type=str,
    help="path containing all datasets (training and validation)",
)


# Assumes that tensor is (nchannels, height, width)
def tensor_rot_90(x):
    return x.flip(2).transpose(1, 2)


def tensor_rot_180(x):
    return x.flip(2).flip(1)


def tensor_rot_270(x):
    return x.transpose(1, 2).flip(2)


def rotate_batch_with_labels(batch, labels):
    images = []
    for img, label in zip(batch, labels):
        if label == 1:
            img = tensor_rot_90(img)
        elif label == 2:
            img = tensor_rot_180(img)
        elif label == 3:
            img = tensor_rot_270(img)
        images.append(img.unsqueeze(0))
    return torch.cat(images)


def rotate_batch(batch, label):
    if label == "rand":
        labels = torch.randint(4, (len(batch),), dtype=torch.long)
    elif label == "expand":
        labels = torch.cat(
            [
                torch.zeros(len(batch), dtype=torch.long),
                torch.zeros(len(batch), dtype=torch.long) + 1,
                torch.zeros(len(batch), dtype=torch.long) + 2,
                torch.zeros(len(batch), dtype=torch.long) + 3,
            ]
        )
        batch = batch.repeat((4, 1, 1, 1))
    else:
        assert isinstance(label, int)
        labels = torch.zeros((len(batch),), dtype=torch.long) + label
    return rotate_batch_with_labels(batch, labels), labels


def rotation_pred(dataloader, model, device):
    # return a tuple of (classification accuracy, rotation prediction accuracy)
    correct_rot = []
    for imgs, _ in iter(dataloader):
        imgs_rot, labels_rot = rotate_batch(imgs, "expand")
        imgs_rot, labels_rot = imgs_rot.to(device), labels_rot.to(device)
        with torch.no_grad():
            _, out_rot = model(imgs_rot)
            pred_rot = torch.argmax(out_rot, dim=1, keepdim=True)
            correct_rot.append(pred_rot.squeeze(1).eq(labels_rot).cpu())
    correct_rot = torch.cat(correct_rot).numpy()
    return np.mean(correct_rot)


if __name__ == "__main__":
    # paths
    args = parser.parse_args()
    dataset_path = args.dataset_path
    if not dataset_path or dataset_path[-1] != "/":
        dataset_path += "/"
    model_name = args.model
    train_set = "train_data"
    val_sets = sorted(["cifar10-f-32", "cifar-10.1-c", "cifar-10.1"])
    temp_file_path = f"../temp/{model_name}/rotation/"

    batch_size = 500
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # load the model
    if model_name == "resnet":
        model = ResNetRotation()
        model_state = model.state_dict()
        fc_rot_weights = torch.load(
            "../model_weights/resnet-rotation-fc.pt", map_location=torch.device("cpu")
        )
    elif model_name == "repvgg":
        model = RepVGGRotation()
        model_state = model.state_dict()
        fc_rot_weights = torch.load(
            "../model_weights/repvgg-rotation-fc.pt", map_location=torch.device("cpu")
        )
    else:
        raise ValueError("Unexpected model_name")
    # load the rotation FC layer weights
    for key, value in fc_rot_weights.items():
        model_state[key] = value
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    # need to do rotation accuracy calculation
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

        rotation_acc = np.zeros(len(train_candidates))
        print(f"===> Calculating rotation accuracy for {train_set}")

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
            rotation_acc[i] = rotation_pred(dataloader, model, device)

        np.save(f"{temp_file_path}{train_set}.npy", rotation_acc)

    if not os.path.exists(f"{temp_file_path}val_sets.npy"):
        # validation set calculation
        val_candidates = []
        val_paths = [f"{dataset_path}{set_name}" for set_name in val_sets]
        for val_path in val_paths:
            for file in sorted(os.listdir(val_path)):
                val_candidates.append(f"{val_path}/{file}")

        rotation_acc = np.zeros(len(val_candidates))
        print(f"===> Calculating rotation accuracy for validation sets")

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
            rotation_acc[i] = rotation_pred(dataloader, model, device)

        np.save(f"{temp_file_path}val_sets.npy", rotation_acc)

    # if the calculation of rotation accuracy is finished
    # calculate the linear regression model (accuracy in %)
    print(
        f"===> Linear Regression model for rotation accuracy method with model: {model_name}"
    )
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
