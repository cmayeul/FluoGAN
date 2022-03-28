# SRGAN

## Help 

      ./main.py -h

## Examples

To compute a super-resolution image from a tif file, with 500 epochs, taking only the 500 first images from the tif file, croping them to extract a 50 × 50 region, saving the parameters and the outputs in test directory, ploting results and learning curves :

      ./main.py ../donnees_ostreopsis/Export-TempFluc.tif -n_e 500 -n_img 500 -x 200 200 250 250 -o test -r -l

## References

fft_conv from https://github.com/fkodom/fft-conv-pytorch
