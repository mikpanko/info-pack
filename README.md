info-pack
=========

MATLAB package to estimate mutual information (MI) between a multidimensional signal and a discrete signal. Originally developed to estimate MI between multi-channel neural signals and discrete targets in the context of [brain-computer interfaces](http://en.wikipedia.org/wiki/Brain%E2%80%93computer_interface).

Main functions:

- `computeInfoGaussian()` - estimates MI using various methods modelling the multidimensional signal as distinct Gaussians for each value of the discrete signal (the full signal with be a mixute of Gaussians)

- `computeInfoGaussianNoise()` - estimates MI using the Gaussian mixture model with bias reduction via incremental noisy dimensions

- `computeInfoDirect()` - estimates MI using the direct method with an optional Panzeri-Treves bias correction. The direct method approximates the signal PDF with a sample histogram and requires the signal to be pre-discretized. The direct method usually works only on signals with small number of dimensions (<5-10, then runs out of memory) and produces significantly biased estimates

- `computeInfoDecoder()` - estimates MI using the confusion matrix produced by decoding the multidimensional signal (mapping onto the discrete signal space)

- `computeInfoKnn()` - estimates MI using the K-nearest neighbor method

- `computeInfoKDE()` - estimates MI using the kernel density estimator method

- `computeInfoIBTB()` - estimates MI using [Information Breakdown ToolBox](http://www.infotoolbox.org/), which is described in detail in [this article](http://www.biomedcentral.com/1471-2202/10/81). The toolbox needs to be downloaded separately.

Supporting functions:

- `generateData()` - generates multidimensional and discrete signals

- `discretizeData()` - discretizes multidimensional signals

- `decodeData()` - decodes multidimensional signals (maps them onto the discrete signal space)

- `computeEntropyAnalytic()` - computes analytical entropy of several multivariate distributions

- `shrinkCov()` - estimates population covariance matrices from a sample using several shrinkage methods, which improve MI estimation using Gaussian methods.

If you have any questions about the package, shoot me an email at mikpanko@gmail.com.
