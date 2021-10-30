# Sparse-SIM-python
I modified the SIM project from @Tlambert03 to get a simple Python version of sparse hessian denoise

Sparse-SIM has two steps, the first is sparse denoising and the second is RL deconvolution
## Tlambert03's SIM project
https://github.com/tlambert03/pycudasirecon
## Sparse-SIM official project address:
https://github.com/WeisongZhao/Sparse-SIM

## Usage
```
from sparse_deconv import sparse_hessian_denoise
from tifffile import imread

img = imread('raw.tif')
g = sparse_hessian_denoise(img,mu=100,sparsity=5,iters=70,sigma=0.8)
```
## RL deconvolution can be done by gputools.deconv
## Requirements

numpy or cupy(CUDA needs to be installed)

tifffile

matplotlib

##Tips
The code under Matlabstyleimplement folder is my own implementation according to the Sparse-SIM author's Matlab version
