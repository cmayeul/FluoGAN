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

Try to apply the software on a small crop of your sample and optimize manually the regularization parameters `-g_x_l1` and `-g_b_grad`. There are related to the proportion of signal compared to the backgroud and the smoothness of the background.

To find the best regularization it is better not to use the discriminator, which means `-g_D 0` and `-n_d 0`. First it will be faster and in any case you have to achieve first convergence with L2 distance before using the discriminator distance to improve results.

### Example to find best regularization

	./main.py images.tif --FWHM 318 -g_x_l1 0.8 -g_D 0 -n_d 0 -n_e 1000 -x 0 0 50 50

### Example to run full algorithm

Once you found the best parameters `-g_x_l1` and `g_b_grad` for regularization then you can start the program with discriminator on the full sample with more epochs.

	./main.py images.tif --FWHM 318 -g_x_l1 0.8 -g_D 1 -n_d 1 -n_e 5000

You can save the results and different curves using the `-o` option and `-l` `-m` `-v` or `-r`

## Full exemple with phantom data 

Two different sets of parameters are provided in the `tests/` folder. Their only difference is the use of discriminator or not. You can run them using the following commands : 

	./main.py tests/with_discriminator.json
	./main.py tests/without_discriminator.json
	
The results are automatically saved in the corresponding folder.

## Installation

Requires python 3 and pytorch >= 1.10 

Install required python modules (Linux) : 

      sudo pip3 install torch torchvision matplotlib tifffile read_lif tqdm czifile
      
See https://pytorch.org to install PyTorch on different platforms

## References

fft_conv from https://github.com/fkodom/fft-conv-pytorch
