# DataCV Challenge @ CVPR 2023

Welcome to DataCV Challenge 2023!

This is the development kit repository for [the 1st DataCV Challenge](https://sites.google.com/view/vdu-cvpr23/competition?authuser=0). This repository includes details on how to download datasets, run baseline models, and organize your result as `answer.zip`. The final evaluation will occur on the [CodeLab evaluation server](https://codalab.lisn.upsaclay.fr/competitions/10221), where all competition information, rules, and dates can be found.

## Overview
Label-free model evaluation is the competition task. It is different from the standard evaluation that calculates model accuracy based on model outputs and corresponding test labels. Label-free model evaluation (AutoEval), on the other hand, has no access to test labels. In this competition, participants need to **design a method that can estimate the model accuracies on test sets without ground truths**.

## Table of Contents

- [DataCV Challenge @ CVPR 2023](#datacv-challenge--cvpr-2023)
	- [Overview](#overview)
	- [Table of Contents](#table-of-contents)
	- [Competition Submission](#competition-submission)
	- [Challenge Data](#challenge-data)
	- [Classifiers being Evaluated](#classifiers-being-evaluated)
	- [Organize Results for Submission](#organize-results-for-submission)
	- [Several Baselines](#several-baselines)
		- [Baseline Results](#baseline-results)
			- [ResNet-56](#resnet-56)
			- [RepVGG-A0](#repvgg-a0)
		- [Code-Execution](#code-execution)
		- [Baseline Description](#baseline-description)

## Competition Submission
In total, the test set comprises 100 datasets. As a result, each model's accuracy should be predicted 100 times. Given that there are two models to be evaluated, the expected number of lines in the "answer.txt" file is 200.

The first 100 lines represent the accuracy predictions of the ResNet model, while the second 100 lines represent those of the RepVGG model. Each of the 100-line predictions is the accuracy prediction for the model on *xxx.npy* dataset, where *xxx* goes from *001* to *100*.

To prepare your submission, you need to write your predicted accuracies into a plain text file named "**answer.txt**", with one prediction (e.g., 0.876543) per line. For example, 

```
0.100000
0.100000
.
.
.
0.100000
0.100000
0.100000
0.100000
```
Then, **zip** the text file and it submit to the competition website. 

**How to organize an answer.txt file for validation evaluation?**

Please refer to [Organize Results for Submission](#organize-results-for-submission). 

In the competition, you are only required to submit the zipped prediction results named the "answer.txt". We give an example for this txt file at [answer.txt](https://drive.google.com/file/d/1WJg2_RCJ1liFCAKfSSthOoFsj5TCMZCK/view) demo.


## Challenge Data
The training dataset consists of 1,000 transformed datasets from the original [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) test set, using the transformation strategy proposed by [Deng et al. (2021)](https://arxiv.org/abs/2007.02915). The validation set was composed of [CIFAR-10.1](https://github.com/modestyachts/CIFAR-10.1), [CIFAR-10.1-C](https://github.com/hendrycks/robustness) (add corruptions [(Hendrycks et al., 2019)](https://arxiv.org/abs/1903.12261) to [CIFAR-10.1](https://github.com/modestyachts/CIFAR-10.1) dataset), and CIFAR-10-F (real-world images collected from [Flickr](https://www.flickr.com))

The CIFAR-10.1 dataset is a single dataset. In contrast, CIFAR-10.1-C and CIFAR-10-F contain 19 and 20 datasets, respectively. Therefore, the total number of datasets in the validation set is 40.

The training datasets share a common label file named `labels.npy`, and images files are named `new_data_xxx.npy`, where `xxx` is a number from 000 to 999. For every dataset in the validation set, the image file and their labels are stored as two separate `Numpy` array files named "data.npy" and "labels.npy". The PyTorch implementation of the Dataset class for loading the data can be found in `utils.py`.

Download the training datasets: [link](https://anu365-my.sharepoint.com/:u:/g/personal/u7136359_anu_edu_au/Eb9yO_Qg41lOkoRS7P6gmqMBk5Q6A2gCV8YbRbuLpB8NwQ?e=WO3Gqi)

Download the validation datasets: [link](https://anu365-my.sharepoint.com/:u:/g/personal/u7136359_anu_edu_au/Edg83yRxM9BPonPP22suB_IBrHlKYV5bOn4VK-c5RZ8dtQ?e=kExXEm)

Download the training datasets' accuracies on the ResNet-56 model: [link](https://anu365-my.sharepoint.com/:t:/g/personal/u7136359_anu_edu_au/EQ4XcZLeVPNAg45JdB0mZ4ABO6nsIDDD3z2_frx0rnbRpg?e=5wA3Xi)

Download the training datasets' accuracies on the RepVGG-A0 model: [link](https://anu365-my.sharepoint.com/:t:/g/personal/u7136359_anu_edu_au/EWUVPpAqYcNJq8iB4AfYD7oBodhHMI1B_1Mijd7x8V8xlA?e=oPDaL3)

**NOTE: To access the test datasets and participate in the competition, please fill in the [Datasets Request Form](https://anu365-my.sharepoint.com/:b:/g/personal/u7136359_anu_edu_au/ERz4ANQ1A31PvJKgd3mNxr8B1F4e0zfaZL3P_NLOvKrivg?e=lG7mkL) and send the signed form to [the competition organiser](mailto:datacvchallenge2023@gmail.com;VDU2023@gmail.com). Failing to provide the form will lead to the revocation of the CodaLab account in the competition.**

## Classifiers being Evaluated
In this competition, the classifiers being evaluated are ResNet-56 and RepVGG-A0. Both implementations can be accessed in the public repository at https://github.com/chenyaofo/pytorch-cifar-models. To utilize the models and load their pretrained weights, use the code provided.
```python
import torch

model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56"， pretrained=True)
model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_repvgg_a0", pretrained=True)
```

## Organize Results for Submission
As we use automated evaluation scripts for submissions, the format of submitted files is very important. So, there is a function to store the accuracy predictions into the required format, named `store_ans` in `code/utils.py` and `results_format/get_answer_txt.py`.

Read the `results_format/get_answer_txt.py` file to comprehend the function usage. Execute the code below to see the results.

```bash
python3 results_format/get_answer_txt.py
```

## Several Baselines
The necessary dependencies are specified in the `requirements.txt` file and the experiments were conducted using Python version 3.10.8, with a single GeForce RTX 2080 Ti GPU.

The table presented below displays the results of the foundational measurements using [root-mean-square error](https://en.wikipedia.org/wiki/Root-mean-square_deviation) (RMSE). Accuracies are converted into percentages prior to calculation.

### Baseline Results
#### ResNet-56

| Method    | CIFAR-10.1 |  CIFAR-10.1-C | CIFAR-10-F  |  Overall   |
| --------  | ---- | ---- | ---- | ------ |
| Rotation  | 7.285  | 6.386  | 7.763  | 7.129  |
| ConfScore | 2.190  | 9.743  | 2.676  | 6.985  |
| Entropy   | 2.424  | 10.300 | 2.913  | 7.402  |
| ATC       | 11.428 | 5.964  | 8.960  | 7.766  |
| FID       | 7.517  | 5.145  | 4.662  | 4.985  |

#### RepVGG-A0

| Method    | CIFAR-10.1 |  CIFAR-10.1-C | CIFAR-10-F  |  Overall   |
| --------  | ---- | ---- | ---- | ------ |
| Rotation  | 16.726 | 17.137 | 8.105 | 13.391 |
| ConfScore | 5.470  | 12.004 | 3.709 | 8.722  |
| Entropy   | 5.997  | 12.645 | 3.419 | 9.093  |
| ATC       | 15.168 | 8.050  | 7.694 | 8.132  |
| FID       | 10.718 | 6.318  | 5.245 | 5.966  |

### Code-Execution
To install required Python libraries, execute the code below.
```bash
pip3 install -r requirements.txt
```
The above results can be replicated by executing the code provided below in the terminal.
```bash
cd code/
chmod u+x run_baselines.sh && ./run_baselines.sh
```
To run one specific baseline, use the code below.
```bash
cd code/
python3 get_accuracy.py --model <resnet/repvgg> --dataset_path DATASET_PATH
python3 baselines/BASELINE.py --model <resnet/repvgg> --dataset_path DATASET_PATH
```

###  Baseline Description
The following succinctly outlines the methodology of each method, as detailed in the appendix of "Predicting Out-of-Distribution Error with the Projection Norm" paper [(Yu et al., 2022)](https://arxiv.org/abs/2202.05834).

**Rotation.** The *Rotation Prediction* (Rotation) [(Deng et al., 2021)](https://arxiv.org/abs/2106.05961) metric is defined as
```math
\text{Rotation} = \frac{1}{m}\sum_{j=1}^m\Big\{\frac{1}{4}\sum_{r \in \{0^\circ, 90^\circ, 180^\circ, 270^\circ\}}\mathbf{1}\{C^r(\tilde{x}_j; \hat{\theta}) \neq y_r\}\Big\},
```
where $y_r$ is the label for $r \in \lbrace 0^{\circ}, 90^{\circ}, 180^{\circ}, 270^{\circ} \rbrace$, and $C^r(\tilde{x}_j; \hat{\theta})$ predicts the rotation degree of an image $\tilde{x}_j$.

**ConfScore.** The *Averaged Confidence* (ConfScore) [(Hendrycks & Gimpel, 2016)](https://arxiv.org/abs/1610.02136) is defined as
$$\text{ConfScore} = \frac{1}{m}\sum_{j=1}^m \max_{k} \text{Softmax}(f(\tilde{x}_j; \hat{\theta}))_k,$$
where $\text{Softmax}(\cdot)$ is the softmax function.

**Entropy.** The *Entropy* [(Guillory et al., 2021)](https://arxiv.org/abs/2107.03315) metric is defined as
```math
\text{Entropy} = \frac{1}{m}\sum_{j=1}^m \text{Ent}(\text{Softmax}(f(\tilde{x}_j; \hat{\theta}))),$$
where $\text{Ent}(p)=-\sum^K_{k=1}p_k\cdot\log(p_k)$.
```

**ATC.** The *Averaged Threshold Confidence* (ATC) [(Garg et al., 2022)](https://arxiv.org/abs/2201.04234) is defined as
```math
\text{ATC} = \frac{1}{m}\sum_{j=1}^m\mathbf{1}\{s(\text{Softmax}(f(\tilde{x}_j; \hat{\theta}))) < t\},$$
where $s(p)=\sum^K_{j=1}p_k\log(p_k)$, and $t$ is defined as the solution to the following equation,
$$\frac{1}{m^{\text{val}}} \sum_{\ell=1}^{m^\text{val}}\mathbf{1}\{s(\text{Softmax}(f(x_{\ell}^{\text{val}}; \hat{\theta}))) < t\} = \frac{1}{m^{\text{val}}}\sum_{\ell=1}^{m^\text{val}}\mathbf{1}\{C(x_\ell^{\text{val}}; \hat{\theta}) \neq y_\ell^{\text{val}}\}
```
where $(x_\ell^{\text{val}}, y_\ell^{\text{val}}), \ell=1,\dots, m^{\text{val}}$, are in-distribution validation samples.

**FID.** The *Frechet Distance* (FD) between datasets [(Deng et al., 2020)](https://arxiv.org/abs/2007.02915) is defined as
```math
\text{FD}(\mathcal{D}_{ori}, \mathcal{D}) = \lvert \lvert \mu_{ori} - \mu \rvert \rvert_2^2 + Tr(\Sigma_{ori} + \Sigma - 2(\Sigma_{ori}\Sigma)^\frac{1}{2}),
```
where $\mu\_{ori}$ and $\mu$ are the mean feature vectors of $\mathcal{D}\_{ori}$ and $\mathcal{D}$, respectively. $\Sigma\_{ori}$ and $\Sigma$ are the covariance matrices of $\mathcal{D}\_{ori}$ and $\mathcal{D}$, respectively. They are calculated from the image features in $\mathcal{D}\_{ori}$ and $\mathcal{D}$, which are extracted using the classifier $f\_{\theta}$ trained on $\mathcal{D}\_{ori}$.

The Frechet Distance calculation functions utilized in this analysis were sourced from a [publicly available repository](https://github.com/Simon4Yan/Meta-set) by [Weijian Deng](https://github.com/Simon4Yan).
