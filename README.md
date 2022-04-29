# LaplaceAE

Code for Laplace Auto-Encoding Bayes and the associate baseline methods.

## Setup

```bash 
# Clone repo and enter
git clone https://github.com/FrederikWarburg/LaplaceAE;
cd LaplaceAE;

# Install Pytorch, see https://pytorch.org/get-started/locally/ for more instuctions
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch

# Install package requirements
pip install -r requirements.txt

# Clone specialized version of stochman
git clone https://github.com/FrederikWarburg/stochman;
cd stochman;
python setup.py install;
```

This should give you a folder structure like this:

    LAPLACEAE
    ├── stochman          # updated stochman version to support different hessian computations
    ├── configs           # config files, organised by experiments
    ├── figures           # generated figures
    ├── src               # source code
    │   ├── hessian       # Code for computing hessians
    │   ├── models        # Model architechtures
    │   ├── tests         # Testing code
    │   └── trainer_*.py  # files for training the actual models
    └── requirements.txt  # file containing python packages that are required to run code

## Train & Test

To train and test a LAE:

```bash
cd src; 
CUDA_VISIBLE_DEVICES=0 python trainer_lae_elbo.py --config PATH_TO_CONFIG
```
