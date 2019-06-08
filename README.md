# Noise2Self: Blind Denoising by Self-Supervision

This repo demonstrates a framework for blind denoising high-dimensional measurements,
as described in the [paper](https://arxiv.org/abs/1901.11365). It can be used to calibrate 
classical image denoisers and train deep neural nets; 
the same principle works on matrices of single-cell gene expression.

<img src="https://github.com/czbiohub/noise2self/blob/master/figs/hanzi_movie.gif" width="512" height="256" title="Hanzi Noise2Self">

*The result of training a U-Net to denoise a stack of noisy Chinese characters. Note that the only input is the noisy data; no ground truth is necessary.*

## Images

The notebook [Intro to Calibration](notebooks/Intro%20to%20Calibration.ipynb) shows how to calibrate any traditional image denoising model, such as median filtering, wavelet thresholding, or non-local means. We use the excellent [scikit-image](www.scikit-image.org) implementations of these methods, and have submitted a PR to incorporate self-supervised calibration directly into the package. (Comments welcome on the [PR](https://github.com/scikit-image/scikit-image/pull/3824)!)

The notebook [Intro to Neural Nets](notebooks/Intro%20to%20Neural%20Nets.ipynb) shows how to train a denoising neural net using a self-supervised loss, on the simple example of MNIST digits. The notebook runs in less than a minute, on CPU, on a MacBook Pro. We implement this in [pytorch](www.pytorch.org).

The notebook [Single Shot Denoising](notebooks/Single-Shot%20Denoising.ipynb) demonstrates that there is enough information in a single 512x512 noisy image for a deep neural net to learn to denoise it, with performance better than classical blind image denoisers.

Because the self-supervised loss is much easier to implement than the data loading, GPU management, logging, and architecture design required for handling any particular dataset, we recommend that you take any existing pipeline for your data and simply modify the training loop.

### Traditional Supervised Learning

```
for i, batch in enumerate(data_loader):
    noisy_images, clean_images = batch
    output = model(noisy_images)
    loss = loss_function(output, clean_images)
```

### Self-Supervised Learning

```
from mask import Masker
masker = Masker()
for i, batch in enumerate(data_loader):
    noisy_images, _ = batch
    input, mask = masker.mask(noisy_images, i)
    output = model(input)
    loss = loss_function(output*mask, noisy_images*mask)
```

Dependencies are in the `environment.yml` file.

The remaining notebooks generate figures from the [paper](https://arxiv.org/abs/1901.11365).




