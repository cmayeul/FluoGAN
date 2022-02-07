# SRGAN

## Help
usage: parser.py [-h] [-i INDEX] [-n_img N_IMAGES] [-p PARAMS_IN] [-o PARAMS_OUT] [-O OUT] [-a ALPHA] [-w KWIDTH] [-s KSIGMA] [-u UNDERSAMPLING] [-e ESIGMA] [-d_gp D_GP] [-g_D G_D] [-g_l2 G_L2]
                 [-g_x_l1 G_X_L1] [-g_b_grad G_B_GRAD] [-n_d N_DISCRIMINATOR] [-n_g N_GENERATOR] [-n_e N_EPOCHS] [-B BATCH_SIZE] [-c] [-d_lr D_LR] [-g_x_lr G_X_LR] [-g_b_lr G_B_LR] [-g_x_eta G_X_ETA]
                 [-l] [-m] [-v] [-r]
                 input

positional arguments:
  input                 input file : video or lif file or tiff image the image serie should be well aligned to yield good results

optional arguments:
  -h, --help            show this help message and exit
  -i INDEX, --index INDEX
                        in case of lif input give the serie index in case of a video input give the channel index
  -n_img N_IMAGES, --n_images N_IMAGES
                        take only the n_img first image from the input serie
  -p PARAMS_IN, --params_in PARAMS_IN
                        load parameters from a json file in the output dir parameters are loaded in this order : 1. CLI, 2. JSON file, 3. default
  -o PARAMS_OUT, --params_out PARAMS_OUT
                        save current configuration in a json parameter file (creates or updates this file in the out dir)
  -O OUT, --out OUT     output directory, default : ./
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
  -c, --cpu             use CPU instead of GPU
  -d_lr D_LR            learning rate for the discriminator update
  -g_x_lr G_X_LR        learning rate for x (inverse of lipschitz constant L)
  -g_b_lr G_B_LR        learning rate for b
  -g_x_eta G_X_ETA      eta param to increase Lipschitz constant during FISTA step
  -l, --save_losses     save G loss (requires extra computation)
  -m, --save_models     save G and D pytorch modules after training
  -v, --save_video      ave a video to show x convergence
  -r, --plot_results    plot and save x and b as plt with their scale

## Examples

./main.py input.lif -p parameters.json


