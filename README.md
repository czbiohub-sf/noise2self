# Noise2Self: Blind Denoising by Self-Supervision

This repo demonstrates a framework for blind denoising based on self-supervision, 
as described in the [paper](https://arxiv.org/abs/1901.11365).

> We propose a general framework for denoising high-dimensional measurements which requires no prior on the signal, no estimate of the noise, and no clean training data. The only assumption is that the noise exhibits statistical independence across different dimensions of the measurement. Moreover, our framework is not restricted to a particular denoising model. We show how it can be used to calibrate any parameterised denoising algorithm, from the single hyperparameter of a median filter to the millions of weights of a deep neural network. We demonstrate this on natural image and microscopy data, where we exploit noise independence between pixels, and on single-cell gene expression data, where we exploit independence between detections of individual molecules. Finally, we prove a theoretical lower bound on the performance of an optimal denoiser. This framework generalizes recent work on training neural nets from noisy images and on cross-validation for matrix factorization.

## Images

The notebook [Intro to Calibration](notebooks/Intro%20to%20Calibration.ipynb) shows how to calibrate any traditional image denoising model, such as median filtering, wavelet thresholding, or non-local means. We use the excellent [scikit-image](www.scikit-image.org) implementations of these methods, and have submitted a PR to incorporate self-supervised calibration directly into the package. (Comments welcome on the [PR](https://github.com/scikit-image/scikit-image/pull/3824)!)

The notebook [Intro to Neural Nets](notebooks/Intro%20to%20Neural%20Nets.ipynb) shows how to train a denoising neural net using a self-supervised loss, on the simple example of MNIST digits. The notebook runs in less than a minute, on CPU, on a MacBook Pro. We implement this in [pytorch](www.pytorch.org).

Because the self-supervised loss is much easier to implement than the data loading, GPU management, logging, and architecture design required for handling any particular dataset, we recommend that you take any existing pipeline for your data and simply modify the training loop.

### Traditional Supervised Learning

```
for i, batch in enumerate(data_loader):
    x, y = batch
    output = model(x)
    loss = loss_function(output, y)
```

### Self-Supervised Learning

```
from mask import Masker
masker = Masker()
for i, batch in enumerate(data_loader):
    x, _ = batch
    
    input, mask = masker.mask(noisy_images, i)
    output = model(input)
    
    loss = loss_function(output*mask, x*mask)
```

Dependencies are in the `environment.yml` file.

The remaining notebooks generate figures from the [paper](https://arxiv.org/abs/1901.11365).




