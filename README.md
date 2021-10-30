# Sparse-SIM-python
I modified the SIM project from @Tlambert03 to get sparse hessian denoise\\
Sparse-SIM has two steps, the first is Sparse denoising and the second is RL deconvolution
## Tlambert03's SIM project
https://github.com/tlambert03/pycudasirecon
## Sparse-SIM official project address:
https://github.com/WeisongZhao/Sparse-SIM

## Usage
```
from sparse_deconv import sparse_hessian_denoise
from tifffile import imread

img = imread('raw.tif')
g = sparse_hessian_denoise(img,fidelity=100,sparsity=30,iteration_num=70,contiz=0.8)
```
## RL deconvolution can be done by gputools.deconv
## Requirements
```
numpy or cupy(CUDA needs to be installed)
tifffile
matplotlib
```
