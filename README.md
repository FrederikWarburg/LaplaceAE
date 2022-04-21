# LaplaceAE

Code for Laplace Auto-Encoding Bayes and the associate baseline methods.

## Setup

```
git clone https://github.com/FrederikWarburg/LaplaceAE;
cd LaplaceAE;
git clone https://github.com/FrederikWarburg/Laplace;
git clone https://github.com/FrederikWarburg/stochman;
cd stochman;
python setup.py install;
```
This should give you a folder structure like this:

```
* Laplace
* stochman
* src
   * hessian
   * models
   * tests
   * ...
```

## Train & Test

To train and test a LAE:

```
cd src; 
CUDA_VISIBLE_DEVICES=0 python trainer_lae_elbo.py --config PATH_TO_CONFIG
```
