## Compressive Feature Pyramid Network
This is a model designed to compress and reconstruct images. Currently only reconstruction is implemented.


## todo
* write an evaluator which calculates bits per pixel
    * current one reports ssim and ms-ssim on kodak dataset (needs verifying)
* Create a new registry for quantization modules. These will be used in the CFPN 
meta architecture to reduce the entropy of the encodings by adding another loss function. 
* implement a GSM as a first pass for quantization