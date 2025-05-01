<p align="center">
<img src="https://github.com/thespcrewroy/gradcon-anomaly/blob/master/figs/banner.jpg" alt="Logo" width="500" height="350" />
<h1 align="center">CSCI 4900 Final Project: Backpropagated Gradient Representations for Anomaly Detection Implementation</h1>
<p align="center">
<a href="https://github.com/thespcrewroy/gradcon-anomaly/tool"><img alt="Tool" src="https://img.shields.io/badge/Python-v3.6.0-turquoise.svg" height="20"/></a>
<a href="https://github.com/thespcrewroy/gradcon-anomaly/graphs/contributors"><img alt="Contributors" src="https://img.shields.io/github/contributors/thespcrewroy/gradcon-anomaly.svg" height="20"/></a>
<a href="https://github.com/thespcrewroy/gradcon-anomaly/pulls"><img alt="Pull Requests" src="https://img.shields.io/github/issues-pr/thespcrewroy/gradcon-anomaly?color=1039ac"/></a>
<a href="https://github.com/thespcrewroy/gradcon-anomaly/graphs/issues"><img alt="Issues" src="https://img.shields.io/github/issues/thespcrewroy/gradcon-anomaly.svg" height="20"/></a>
<a href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="License: CC BY-NC-ND 4.0" src="https://img.shields.io/badge/Attributions-OLIVES @ Georgia Tech-lightgrey.svg" height="20"/></a>

<p align="center">
<a href="https://github.com/thespcrewroy/gradcon-anomaly/graphs/forks"><img alt="Forks" src="https://img.shields.io/github/forks/thespcrewroy/gradcon-anomaly.svg" height="20"/></a>
<a href="https://github.com/thespcrewroy/gradcon-anomaly/graphs/stars"><img alt="Stars" src="https://img.shields.io/github/stars/thespcrewroy/gradcon-anomaly.svg" height="20"/></a>

</p>

<p align="center">
  <b>Abstract</b></br>
  <sub>Learning representations that clearly distinguish between normal and abnormal data is key to the success of anomaly detection. Most of existing anomaly detection algorithms use activation representations from forward propagation while not exploiting gradients from backpropagation to characterize data. Gradients capture model updates required to represent data. Anomalies require more drastic updates to models to fully represent them compared to normal data. Hence, we propose the utilization of backpropagated gradients as representations to characterize model behavior on anomalies and, consequently, detect such anomalies. We show that the proposed method using gradient-based representations achieves state-of-the-art anomaly detection performance in benchmarking image recognition datasets. Also, we highlight the computational efficiency and the simplicity of the proposed method by comparing with other state-of-the-art methods relying on adversarial networks or autoregressive models, which require at least 27 times more model parameters than the proposed method. <sub>
</p>

<br />

<p align="center">
  <img src="./figs/abstract.jpg", alt="Demo" width="800"">
</p>

- **[Adjective]**: [Description]
- **[Adjective]**: [Description]
- **[Adjective]**: [Description]

<details>
<summary>📖 Table of Contents</summary>
<br />

## Table of Contents

- [Getting Started](#getting-started)
  - [Setting Up The Environment](#set-up-the-environment)
  - [Prepare Datasets](#prepare-datasets)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Questions](#questions)
- [Thanks to all our Contributors!](#thanks-to-all-our-contributors)
</details>

[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)](#getting-started)
## Getting Started (Not Finished)

### Set up the environment
Clone this repository and run following commands to create a conda environment and install all dependencies.
```
conda create -n gradcon python=3.6
conda activate gradcon
cd gradcon-anomaly
conda install pytorch torchvision -c pytorch
pip install -r requirments.txt
```
If using a MAC M chips with ARM-Based Processor

1. Right click terminal application → Get Info → Check “Open using Rosetta”
2. Close all terminal windows
3. Reopen the terminal application
4. Download the x86_64 [Miniconda installer from](https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh)
```
bash Miniconda3-latest-MacOSX-x86_64.sh
  - install into a seperate path file from main 'minimiconda': /Users/myname/miniconda3_x86
/Users/myname/miniconda3_x86/bin/conda init
conda info
  - you should see 'platform : osx-64'
conda create -n gradcon python=3.6
conda activate gradcon
cd gradcon-anomaly
conda install pytorch torchvision -c pytorch
pip install -r requirments.txt
```

### Prepare datasets
Works with: CIFAR-10; FMNIST; MNIST; CURE_TSR

Following datasets must be downloaded seperately: <br>
* To download CURE-TSR dataset, please visit [this repository](https://github.com/olivesgatech/CURE-TSR). <br>
* To download Maldeb dataset, please visit [this kaggle repository](https://www.kaggle.com/datasets/saquib7hussain/maldeb-dataset)

```
usage: prep_datasets.py [-h] [--dataset DATASET] [--save_dir SAVE_DIR]

Download datasets and create splits

optional arguments:
  -h, --help           show this help message and exit
  --dataset DATASET    Dataset to be downloaded (e.g. cifar-10, mnist, fmnist)
  --save_dir SAVE_DIR  Path to save the data
```

Run prep_dataset.py to download datasets and create train/val/test splits as follows:
```
python prep_datasets.py --dataset 'maldeb' --save_dir ./datasets
``` 

Run following commands:
### Training
```
python train.py --dataset 'cifar-10' --dataset_dir './datasets' --save_dir './save'  --save_name 'GradConCAE'
``` 

### Evaluation
```
python eval.py --dataset 'cifar-10' --dataset_dir './datasets'  --ckpt_dir './save' --ckpt_name 'GradConCAE' --output_dir './results'
```
[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)](#questions)
## Questions?

If you have any questions, regarding the dataset or the code, contact the authors: (gukyeong.kwon@gatech.edu or mohit.p@gatech.edu). <br>
Even better, open an issue in the [here](https://github.com/gukyeongkwon/gradcon-anomaly/issues) and the author's will do their best to help.

[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)](#contributors)

## Thanks to all our Contributors!

<a href="https://github.com/thespcrewroy/gradcon-anomaly/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=thespcrewroy/gradcon-anomaly" />

[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)](#attributions)

<a href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Attribution-NonCommercial-NoDerivatives 4.0 International" src="https://img.shields.io/badge/Attributions-OLIVES @ Georgia Tech-lightgrey.svg" height="20"/></a>

This work was conducted in the [OLIVES @ Georgia Institute of Technology](http://www.ghassanalregib.info) under researchers [Gukyeong Kwon](https://https://gukyeongkwon.github.io/), [Mohit Prabhushankar](https://www.linkedin.com/in/mohitps/), [Dogancan Temel](http://cantemel.com/), and [Ghassan AlRegib](http://www.ghassanalregib.info)

The official code repository for the paper: [***"Backpropagated Gradient Representations for Anomaly Detection,"*** **In Proceedings of the European Conference on Computer Vision (ECCV), 2020.**](https://github.com/gukyeongkwon/gradcon-anomaly)

<p align="right"><a href="#top">🔼 Back to top</a></p>
</small>
