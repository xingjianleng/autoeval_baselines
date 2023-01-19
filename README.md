# Baselines
This repository includes various baseline techniques for label-free model evaluation task in the VDU2023 competition.

## Methods
The following succinctly outlines the methodology of each method, as detailed in the appendix of "Predicting Out-of-Distribution Error with the Projection Norm" paper [(Yu et al., 2022)](https://arxiv.org/abs/2202.05834).

### Rotation
The *Rotation Prediction* (Rotation) [(Deng et al., 2021)](https://arxiv.org/abs/2106.05961) metric is defined as
```math
\text{Rotation} = \frac{1}{m}\sum_{j=1}^m\Big\{\frac{1}{4}\sum_{r \in \{0^\circ, 90^\circ, 180^\circ, 270^\circ\}}\mathbf{1}\{C^r(\tilde{x}_j; \hat{\theta}) \neq y_r\}\Big\},
```
where $y_r$ is the label for $r \in \lbrace 0^{\circ}, 90^{\circ}, 180^{\circ}, 270^{\circ} \rbrace$, and $C^r(\tilde{x}_j; \hat{\theta})$ predicts the rotation degree of an image $\tilde{x}_j$.

### ConfScore
The *Averaged Confidence* (ConfScore) [(Hendrycks & Gimpel, 2016)](https://arxiv.org/abs/1610.02136) is defined as
$$\text{ConfScore} = \frac{1}{m}\sum_{j=1}^m \max_{k} \text{Softmax}(f(\tilde{x}_j; \hat{\theta}))_k,$$
where $\text{Softmax}(\cdot)$ is the softmax function.

### Entropy
The *Entropy* [(Guillory et al., 2021)](https://arxiv.org/abs/2107.03315) metric is defined as
```math
\text{Entropy} = \frac{1}{m}\sum_{j=1}^m \text{Ent}(\text{Softmax}(f(\tilde{x}_j; \hat{\theta}))),$$
where $\text{Ent}(p)=-\sum^K_{k=1}p_k\cdot\log(p_k)$.
```

### ATC
The *Averaged Threshold Confidence* (ATC) [(Garg et al., 2022)](https://arxiv.org/abs/2201.04234) is defined as
```math
\text{ATC} = \frac{1}{m}\sum_{j=1}^m\mathbf{1}\{s(\text{Softmax}(f(\tilde{x}_j; \hat{\theta}))) < t\},$$
where $s(p)=\sum^K_{j=1}p_k\log(p_k)$, and $t$ is defined as the solution to the following equation,
$$\frac{1}{m^{\text{val}}} \sum_{\ell=1}^{m^\text{val}}\mathbf{1}\{s(\text{Softmax}(f(x_{\ell}^{\text{val}}; \hat{\theta}))) < t\} = \frac{1}{m^{\text{val}}}\sum_{\ell=1}^{m^\text{val}}\mathbf{1}\{C(x_\ell^{\text{val}}; \hat{\theta}) \neq y_\ell^{\text{val}}\}
```
where $(x_\ell^{\text{val}}, y_\ell^{\text{val}}), \ell=1,\dots, m^{\text{val}}$, are in-distribution validation samples.

### FID
The *Frechet Distance* (FD) between datasets [(Deng et al., 2020)](https://arxiv.org/abs/2007.02915) is defined as
```math
\text{FD}(\mathcal{D}_{ori}, \mathcal{D}) = \lvert \lvert \mu_{ori} - \mu \rvert \rvert_2^2 + Tr(\Sigma_{ori} + \Sigma - 2(\Sigma_{ori}\Sigma)^\frac{1}{2}),
```
where $\mu\_{ori}$ and $\mu$ are the mean feature vectors of $\mathcal{D}\_{ori}$ and $\mathcal{D}$, respectively. $\Sigma\_{ori}$ and $\Sigma$ are the covariance matrices of $\mathcal{D}\_{ori}$ and $\mathcal{D}$, respectively. They are calculated from the image features in $\mathcal{D}\_{ori}$ and $\mathcal{D}$, which are extracted using the classifier $f\_{\theta}$ trained on $\mathcal{D}\_{ori}$.

## Results
The necessary Python dependencies are specified in the `requirements.txt` file and the experiments were executed using Python version 3.10.8.

The table presented below displays the results of the foundational measurements using [root-mean-square error](https://en.wikipedia.org/wiki/Root-mean-square_deviation) (RMSE). The experiment was conducted utilizing a single Geforce RTX 2080 Ti GPU, with a training dataset comprised of an augmented 1,000 [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) test set. The validation set was composed of [CIFAR-10.1](https://github.com/modestyachts/CIFAR-10.1), [CIFAR-10.1-C](https://github.com/hendrycks/robustness) (add corruptions to [CIFAR-10.1](https://github.com/modestyachts/CIFAR-10.1) dataset), and CIFAR-10-F (real-world images collected from [Flickr](https://www.flickr.com)).

The CIFAR-10.1 dataset is a single dataset. In contrast, CIFAR-10-C and CIFAR-10-F each contain 20 datasets. Therefore, the total number of datasets is 40.

### ResNet-56 (from [public repository](https://github.com/chenyaofo/pytorch-cifar-models))

| Method    | CIFAR-10.1 |  CIFAR-10.1-C | CIFAR-10-F  |  Overall   |
| --------  | ---- | ---- | ---- | ------ |
| Rotation  | 7.285  | 6.386  | 5.159  | 5.831  |
| ConfScore | 2.190  | 9.743  | 4.852  | 7.549  |
| Entropy   | 2.424  | 10.300 | 4.340  | 7.743  |
| ATC       | 11.428 | 5.964  | 4.112  | 5.350  |
| FID       | 7.517  | 5.145  | 3.930  | 4.659  |

### RepVGG (from [public repository](https://github.com/chenyaofo/pytorch-cifar-models))

| Method    | CIFAR-10.1 |  CIFAR-10.1-C | CIFAR-10-F  |  Overall   |
| --------  | ---- | ---- | ---- | ------ |
| Rotation  | 16.726 | 17.137 | 8.272 | 13.442 |
| ConfScore | 5.470  | 12.004 | 3.320 | 8.643  |
| Entropy   | 5.997  | 12.645 | 3.630 | 9.134  |
| ATC       | 15.168 | 8.050  | 8.256 | 8.403  |
| FID       | 10.718 | 6.318  | 5.886 | 6.257  |

## Code execution
The above results can be replicated by executing the code provided below in the terminal.
```bash
pip3 install -r requirements.txt
chmod u+x run.sh && ./run.sh
```
