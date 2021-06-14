# Privacy in Synthetic generated data

## WGAN-GP
Based on the original implementation (available on this [repository](https://github.com/igul222/improved_wgan_training)) of WGAN with gradient penalty proposed by Gulrajani et al. in [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028).

We have updated the code to work in python 3, refactorized it and added support to train with the original NIST dataset after applying some preprocessing.

Example of execution:
```bash
cd wgan-gp
python gan_nist.py --datapath /path/to/nist/data/folder
```
The NIST data folder expected is the one produced by the script `linkNist.py` in the root directory.

### Attack
Based on the original implementation (available on this [repository](https://github.com/DingfanChen/GAN-Leaks)) of GAN-Leaks proposed by Chen et al. in [GAN-Leaks: A Taxonomy of Membership Inference Attacks against Generative Models](https://arxiv.org/abs/1909.03935).


WARNING: to use the scripts, tensorflow v1.14.0 is needed. To create a new environment and install all dependencies:

```bash
conda create --name ganleaks-tf python=3.6
conda activate ganleaks-tf
conda install tensorflow-gpu=1.14.0
pip install six tqdm pillow matplotlib scikit-learn
```

Example with full black-box:
```bash
cd wgan-gp
python sample.py --model_path "models/model_dir" # Create synthetic samples from the generative model
python attacks/fbb.py -name test1 -gdir "models/model_dir" -data "/path/to/nist/data/folder"
python attacks/eval_roc.py -ldir "attacks/results/fbb/test1" --attack_type fbb
```

## NIST Data

Source: https://www.nist.gov/srd/nist-special-database-19

Summary table

| partition | writers | forms | digits | uppers | lowers | const | writer origin |
| --- | --- | --- | --- |--- |--- |--- |--- |
| hsf_0 | 0000-0499 | SD1 | SD3 | SD3 | SD3 | SD19 | Census Field |
| hsf_1 | 0500-0999 | SD1 | SD3 | SD3 | SD3 | SD19 | Census Field |
| hsf_2 | 1000-1499 | SD1 | SD3 | SD3 | SD3 | SD19 | Census Field |
| hsf_3 | 1500-2099 | SD1 | SD3 | SD3 | SD3 | † | Census Field |
| hsf_4 | 2100-2599 | SD19 | SD7 | SD7 | SD7 | ø | High School |
| hsf_5 | 2600-3099 | - | - | - | - | ø | High School |
| hsf_6 | 3100-3599 | SD19 | SD19 | SD19 | SD19 | ø | Census MD |
| hsf_7 | 3600-4099 | SD19 | SD19 | SD19 | SD19 | ø | Census MD |
| hsf_8 | 4100-4169 | SD19 | † | † | † | † | Census MD |

SD3, SD7 are Special Databases 3 and 7 released as as the training and testing materials for the First Census OCR Systems Conference. A † indicates that those fields were completed but not processed at the time of the CD release. A ø indicates that the field was never filled out.

## QMNIST Data

Reconstruction of the original MNIST dataset using NIST data. [Original paper](https://arxiv.org/pdf/1905.10498.pdf) by Chhavi Yadav and Léon Bottou.

The images are numpy arrays with shape (n_images,28,28), with values from 0 to 255 (dtype=uint8).
The targets are arrays with shape (n_images,8). Each of the 8 values contain the following information:

| Position | Description  | Range |
|:------:|--------------|-------|
| 0 | Character class | 0 to 9
| 1 | NIST HSF series | 0, 1, 2, 3, 4, 6, or 7
| 2 | NIST writer ID | 0-2599 and 3100-4099  
| 3 | Digit index for this writer | 0 to 146
| 4 | NIST class code | 30-39
| 5 | Global NIST digit index | 0 to 402952
| 6 | Duplicate | 0
| 7 | Unused | 0

Source: [Original QMNIST repository](https://github.com/facebookresearch/qmnist)