"""
The Frechet Distance calculation functions utilized in this analysis were sourced from a publicly available repository
https://github.com/Simon4Yan/Meta-set
"""
import os
import sys

import numpy as np
from scipy import linalg
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import torch
import torch.nn
import torch.utils.data
import torchvision.datasets
from tqdm import tqdm

from utils import CIFAR10NP, TRANSFORM


def get_activations(dataloader, model, dims, device):
    # Calculates the activations of final feature vector for all images
    batch_size = dataloader.batch_size
    n_used_imgs = len(dataloader.dataset)

    pred_arr = np.empty((n_used_imgs, dims))

    with torch.no_grad():
        for i, (imgs, _) in enumerate(dataloader):
            start = i * batch_size
            end = start + batch_size
            imgs = imgs.to(device)
            pred = model(imgs)
            pred_arr[start:end] = pred.cpu().data.numpy().reshape(imgs.shape[0], -1)
    return pred_arr


def calculate_activation_statistics(dataloader, model, dims, device):
    # Calculation of the statistics used by the FD.
    act = get_activations(dataloader, model, dims, device)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma, act


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    # Numpy implementation of the Frechet Distance.
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, \
        "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = 'fid calculation produces singular product; ''adding %s to diagonal of cov estimates' % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


if __name__ == "__main__":
    # paths
    dataset_path = "/data/lengx/cifar/"
    train_set = "cifar10-test-transformed"
    val_sets = sorted(["cifar10-f-32", "cifar-10.1-c", "cifar-10.1"])
    model_name = sys.argv[1]
    temp_file_path = f"temp/{model_name}/fid/"

    batch_size = 500
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # load the model
    if model_name == "resnet":
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True)
        model_feat = torch.nn.Sequential(*list(model.children())[:-1], torch.nn.Flatten())
        dims = 64
    elif model_name == "repvgg":
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_repvgg_a0", pretrained=True)
        model_feat = torch.nn.Sequential(*list(model.children())[:-1], torch.nn.Flatten())
        dims = 1280
    else:
        raise ValueError("Unexpected model_name")
    model_feat.to(device)
    model_feat.eval()

    # Use original CIFAR10 training data to calculate reference FID
    print("===> Calculating the FID of original dataset")
    cifar_testloader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.CIFAR10(
            root=dataset_path,
            train=True,
            transform=TRANSFORM,
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    m1, s1, act1 = calculate_activation_statistics(cifar_testloader, model_feat, dims, device)

    # need to do fid calculation
    if not os.path.exists(temp_file_path) or not os.path.exists(f"{temp_file_path}{train_set}.npy"):
        if not os.path.exists(temp_file_path):
            os.mkdir(temp_file_path)

        # training set calculation
        train_path = f"{dataset_path}{train_set}"
        train_candidates = []
        for file in sorted(os.listdir(train_path)):
            if file.endswith(".npy") and file.startswith("new_data"):
                train_candidates.append(file)
        
        fids = np.zeros(len(train_candidates))
        print(f"===> Calculating FID for {train_set}")        

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
            m2, s2, act2 = calculate_activation_statistics(dataloader, model_feat, dims, device)
            fids[i] = calculate_frechet_distance(m1, s1, m2, s2)

        np.save(f"{temp_file_path}{train_set}.npy", fids)

    if not os.path.exists(f"{temp_file_path}val_sets.npy"):
        # validation set calculation
        val_candidates = []
        val_paths = [f"{dataset_path}{set_name}" for set_name in val_sets]
        for val_path in val_paths:
            for file in sorted(os.listdir(val_path)):
                val_candidates.append(f"{val_path}/{file}")
        
        fids = np.zeros(len(val_candidates))
        print(f"===> Calculating FID for validation sets")        

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
            m2, s2, act2 = calculate_activation_statistics(dataloader, model_feat, dims, device)
            fids[i] = calculate_frechet_distance(m1, s1, m2, s2)

        np.save(f"{temp_file_path}val_sets.npy", fids)
    
    # if the calculation of rotation accuracy is finished
    # calculate the linear regression model (accuracy in %)
    print(f"===> Linear Regression model for FID method with model: {model_name}")
    train_x = np.load(f"{temp_file_path}{train_set}.npy")
    train_y = np.load(f"temp/{model_name}/acc/{train_set}.npy") * 100
    val_x = np.load(f"{temp_file_path}val_sets.npy")
    val_y = np.load(f"temp/{model_name}/acc/val_sets.npy") * 100

    lr = LinearRegression()
    lr.fit(train_x.reshape(-1, 1), train_y)
    val_y_hat = lr.predict(val_x.reshape(-1, 1))
    rmse_loss = mean_squared_error(y_true=val_y, y_pred=val_y_hat, squared=False)
    print(f"The RMSE on validation set is: {rmse_loss}")
