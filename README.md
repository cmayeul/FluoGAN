# SRGAN

## Help

usage: main.py [-h] [-o OUTPUT] [-p PARAMS] [-i INDEX] [-n_img N_IMAGES] [-x CROP CROP CROP CROP] [-a ALPHA] [-w KWIDTH] [-s KSIGMA] [-u UNDERSAMPLING] [-e ESIGMA] [--c C] [--n_conv N_CONV]
               [--conv_k_size CONV_K_SIZE] [--max_pool_k_size MAX_POOL_K_SIZE] [-d_gp D_GP] [-g_D G_D] [-g_l2 G_L2] [-g_x_l1 G_X_L1] [-g_b_grad G_B_GRAD] [-n_d N_DISCRIMINATOR] [-n_g N_GENERATOR]
               [-n_e N_EPOCHS] [-B BATCH_SIZE] [-C] [-d_lr D_LR] [-g_x_lr G_X_LR] [-g_x_min_lr G_X_MIN_LR] [-g_b_lr G_B_LR] [-g_x_eta G_X_ETA] [-l] [-m] [-v] [-r]
               input

positional arguments:
  input                 main input, may be : 1. a video or lif file or tiff image serie 2. a json parameter file containing path to input data 3. a directory containing a file called 'parameter.json'
                        Remark 1 : the image serie should be well aligned to yield good results Remark 2 : this software creates or overwrite the output directory to save parameters and results (default
                        : current directory)

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        output directory, default : ./
  -p PARAMS, --params PARAMS
                        load parameters from a json file parameters are loaded in this order : 1. CLI, 2. JSON file, 3. default
  -i INDEX, --index INDEX
                        in case of lif input give the serie index in case of a video input give the channel index
  -n_img N_IMAGES, --n_images N_IMAGES
                        take only the n_img first image from the input serie
  -x CROP CROP CROP CROP, --crop CROP CROP CROP CROP
                        select only a region of the input data give the crop window coordinates xmin ymin xmax ymax
  -a ALPHA, --alpha ALPHA
                        average number of photons per emitter
  -w KWIDTH, --kwidth KWIDTH
                        width of the convolution kernel
  -s KSIGMA, --ksigma KSIGMA
                        width of the gaussian in conv kernel
  -u UNDERSAMPLING, --undersampling UNDERSAMPLING
                        downsampling factor
  -e ESIGMA, --esigma ESIGMA
                        standard deviation of the gaussian noise
  --c C                 number of channels in the first conv layer (doubles after each next layer
  --n_conv N_CONV       number of convolutional layers for Discriminator
  --conv_k_size CONV_K_SIZE
                        kernel size for conv layers
  --max_pool_k_size MAX_POOL_K_SIZE
                        kernel size for max pool layer
  -d_gp D_GP            regularization param for grad penalty (WGAN-GP)
  -g_D G_D              weight for discriminator distance
  -g_l2 G_L2            weight for L2 distance
  -g_x_l1 G_X_L1        param for L1 reg on x
  -g_b_grad G_B_GRAD    param for smooth reg on b
  -n_d N_DISCRIMINATOR, --n_discriminator N_DISCRIMINATOR
                        number of discriminator update during 1 epoch
  -n_g N_GENERATOR, --n_generator N_GENERATOR
                        number of generator update during 1 epoch
  -n_e N_EPOCHS, --n_epochs N_EPOCHS
                        number of epochs
  -B BATCH_SIZE, --batch_size BATCH_SIZE
                        batch size for D and G training
  -C, --cpu             use CPU instead of GPU
  -d_lr D_LR            learning rate for the discriminator update
  -g_x_lr G_X_LR        learning rate for x (inverse of lipschitz constant L)
  -g_x_min_lr G_X_MIN_LR
                        min learning rate for x (inverse of lipschitz constant L)
  -g_b_lr G_B_LR        learning rate for b
  -g_x_eta G_X_ETA      eta param to increase Lipschitz constant during FISTA step
  -l, --save_losses     save G loss (requires extra computation)
  -m, --save_models     save G and D pytorch modules after training
  -v, --save_video      ave a video to show x convergence
  -r, --plot_results    plot and save x and b as plt with their scale


## Examples

To compute super resolution image from a tif file, with 500 epochs, taking only the 500 first images from the tif file, croping them to extract a 50 × 50 region, saving the parameters and the outputs in test directory, ploting results and learning curves :

./main.py ../donnees_ostreopsis/Export-TempFluc.tif -n_e 500 -n_img 500 -x 200 200 250 250 -o test -r -l

## References

fft_conv from https://github.com/fkodom/fft-conv-pytorch
