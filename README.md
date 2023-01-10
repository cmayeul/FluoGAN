# FluoGAN

## Help 

      ./main.py -h

## Examples

To compute a super-resolution image from a tif file, with 1200 epochs, saving the parameters and the outputs in test directory, ploting results and learning curves :

      ./main.py test/test_data.tif -n_e 1200 -o test -r -l

If a parameter json file already exists, then you can directely use it as an input, and eventually overwrite some parameters (here the number of epochs) :

      ./main.py test/ -n_e 800

The test data is a portion of a "phantom" sample with calibrated filaments. Here the two nearest filaments are separated by 300nm

## Installation

Requires python 3 and pytorch >= 1.10 

Install required python modules (Linux) : 

      sudo pip3 install torch torchvision matplotlib tifffile read_lif tqdm czifile
      
See https://pytorch.org to install PyTorch on different platforms

## References

fft_conv from https://github.com/fkodom/fft-conv-pytorch
