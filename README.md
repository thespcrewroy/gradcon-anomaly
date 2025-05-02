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

- **Lightweight**: low parameter count (e.g., ~230K parameters) makes it efficient for deployment in resource-constrained environments
- **Plug-and-Play Architecture**: a simple autoencoder that can be used with various datasets without major architectural change
- **Instantaneous**: the small architecture supports quick inference, making it viable for real-time or near-real-time anomaly detection.

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
## Getting Started

### Set up the environment
Clone this repository and run following commands to create a conda environment and install all dependencies.
```
pip install crewai (requires Python >= 3.10 and < 3.13)
conda create -n gradcon python=3.6
conda activate gradcon
cd gradcon-anomaly
conda install pytorch torchvision -c pytorch
pip install -r requirements.txt
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
```
usage: prep_datasets.py [-h] [--dataset DATASET] [--save_dir SAVE_DIR]

Download datasets and create splits

optional arguments:
  -h, --help           show this help message and exit
  --dataset DATASET    Dataset to be downloaded (e.g. cifar-10, mnist, fmnist)
  --save_dir SAVE_DIR  Path to save the data
```

Works with: CIFAR-10; FMNIST; MNIST; CURE_TSR; Maldeb

CURE-TSR:
* To download CURE-TSR dataset, please visit [this repository](https://github.com/olivesgatech/CURE-TSR).
* Place the downloaded zip archive within the 'datasets' directory

Maldeb:
* To download Maldeb dataset, please visit [this kaggle repository](https://www.kaggle.com/datasets/saquib7hussain/maldeb-dataset) <br>
* Place the downloaded dataset within the 'datasets' directory
* Unzip the archive
* Rename the parent folder to 'Maldeb'
* Make sure it has the following file structure folderwise:
```
Maldeb
├── Benign
└── Malicious
```

Run prep_dataset.py to download datasets and create train/val/test splits on a lightweight model for debugging:
```
python prep_datasets.py --dataset maldeb --save_dir ./datasets --benign_cap 300 --malicious_cap 100
```
Run prep_dataset.py to download datasets and create train/val/test splits for a production ready model as follows:
```
python prep_datasets_orig.py --dataset 'maldeb' --save_dir ./datasets
```
### Training
```
usage: train.py [-h] [-e N] [--start-epoch N] [-pf N] [-wf N] [-r PATH]
                [--dataset DATASET] [--dataset_dir DIR] [--save_dir DIR]
                [--save_name NAME] [-gw WEIGHT]

Training GradCon

optional arguments:
  -h, --help            show this help message and exit
  -e N, --epochs N      number of total epochs to run (default: 5)
  --start-epoch N       manual epoch number (useful on restarts)
  -pf N, --print-freq N print frequency (default: 10)
  -wf N, --write-freq N write frequency (default: 5)
  -r PATH, --resume PATH
                        Resume training from a checkpoint
  --dataset DATASET     Dataset to be used for training (e.g. cifar-10, mnist, fmnist, maldeb)
  --dataset_dir DIR     Path to the dataset (default: ./datasets)
  --save_dir DIR        Path to save the model and logs (default: ./save)
  --save_name NAME      Save name for the run (default: GradConCAE)
  -gw WEIGHT, --grad-loss-weight WEIGHT
                        Gradient loss weight (default: 0.03)
```

Run train.py to train the autoencoder on the inliner classes to obtain a lightweight model for debugging:
```
python train.py --dataset maldeb --dataset_dir ./datasets --save_dir ./save --save_name GradConCAE_test --epochs 2 --print-freq 1 --write-freq 1
```
Run train.py to train the autoencoder on the inliner classes to obtain a production ready model:
```
python train.py --dataset 'maldeb' --dataset_dir './datasets' --save_dir './save' --save_name 'GradConCAE' --epochs 30 --grad-loss-weight 0.05 --write-freq 2
```

See the visualizations on a lightweight model for debugging:
```
tensorboard --logdir=./save/maldeb/GradConCAE_test/logs --port=6006
```
See the visualizations on a production ready model:
```
tensorboard --logdir=./save/maldeb/GradConCAE/logs --port=6006
```

### Evaluation
```
usage: eval.py [-h] [--print-freq N] [--dataset DATASET]
               [--dataset_dir DATASET_DIR] [--ckpt_dir CKPT_DIR]
               [--ckpt_name CKPT_NAME] [--output_dir OUTPUT_DIR]
               [--grad-loss-weight N]

Evaluation of GradCon (Fixed Inlier Class)

optional arguments:
  -h, --help            show this help message and exit
  --print-freq N, -pf N
                        print frequency (default: 10)
  --dataset DATASET     Dataset to be used for training (e.g. cifar-10, mnist, fmnist)
  --dataset_dir DATASET_DIR
                        Path for the dataset
  --ckpt_dir CKPT_DIR   Path to the folder that contains saved models
  --ckpt_name CKPT_NAME
                        Checkpoint name
  --output_dir OUTPUT_DIR
                        Path to save the result file
  --grad-loss-weight N, -gw N
                        gradient loss weight for the anomaly score
```

Run eval.py to evaluate the decoder on the eval classes to test a lightweight model for debugging:
```
python eval.py --dataset maldeb --dataset_dir ./datasets --ckpt_dir ./save --ckpt_name GradConCAE_test --output_dir ./results
```

Run eval.py to evaluate the decoder on the eval classes to test a production ready model:
```
python eval.py --dataset 'maldeb' --dataset_dir './datasets'  --ckpt_dir './save' --ckpt_name 'GradConCAE' --output_dir './results'
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
