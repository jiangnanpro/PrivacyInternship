# Privacy in Synthetic generated data

## WGAN-GP
Based on the original implementation (available on this [repository](https://github.com/igul222/improved_wgan_training)) of WGAN with gradient penalty proposed by Gulrajani et al. in [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028).

We have updated the code to work in python 3, refactorized it and added support to train with the original NIST dataset after applying some preprocessing.

Example of execution:
```
cd wgan-gp
python gan_nist.py --datapath /path/to/nist/data/folder
```
The NIST data folder expected is the one produced by the script `linkNist.py` in the root directory.
