# FluoGAN

## Help 

      ./main.py -h
      
## Procedure

### Set the microscope parameters

You have to provide to the program :
	- either the FWHM of the PSF or the wavelength and the numerical aperture
	- the sensor's pixels size in nanometers
	
The corresponding options are respectively `--FWHM` or `--NA --wavelength`
	
###  Find the best regularisation 

Try to apply the software on a small crop of your sample and optimize manually the regularization parameter `-g_x_l1`. It is related to the proportion of signal compared to the backgroud. 

To find the best regularization it is better not to use the discriminator, which means `-g_D 0` and `-n_d 0`. First it will be faster and in any case you have to achieve first convergence with L2 distance before using the discriminator distance to improve results.

### Example to find best regularization

	./main.py images.tif --FWHM 413 -g_x_l1 0.2 -g_D 0 -n_d 0 -n_e 1000 -x 0 0 100 100


## Full Example

Once you found the best parameter `-g_x_l1` for regularization then you can start the program with discriminator on the full sample with more epochs.

	./main.py images.tif --FWHM 413 -g_x_l1 0.2 -g_D 1 -n_d 1 -n_e 4000

You can save the results and different curves using the `-o` option and `-l` `-m` `-v` or `-r`

## Installation

Requires python 3 and pytorch >= 1.10 

Install required python modules (Linux) : 

      sudo pip3 install torch torchvision matplotlib tifffile read_lif tqdm czifile
      
See https://pytorch.org to install PyTorch on different platforms

## References

fft_conv from https://github.com/fkodom/fft-conv-pytorch
