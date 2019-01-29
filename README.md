# Noise2Self: Blind Denoising by Self-Supervision

In addition to the supplementary text, `supplement.pdf`, this folder contains:

Notebooks for:

1. Calibrating median filter, wavelet denoisers, and NL-means.
2. Simulating Gaussian Processes
3. Generating a GP from a Template dataset and comparing optimal inference.
4. Fitting a linear model to single-cell sequencing data

These are in the `notebooks` folder.

There are also two files, `mask.py` and `train.py`, which demonstrate
the procedure for training masked neural networks used in Section 5 of the paper.
To run they require additional infrastructure, but the core choices are
implemented here in simple `pytorch` code, and we hope they clarify the
method. Models used are in the `models` folder.
