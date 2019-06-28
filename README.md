# gan-variants-tensorflow2.0
Tensorflow2.0 implementation of different Generative Adversarial Networks

### Results
DCGANs can generate an image similar to the ones in the dataset through a random noise vector.
Conditional GANs can be used to generate a specific class of image out of all the classes in the dataset. It takes in an image label along with the noise as input and generates an image of the corresponding label. The visualization is shown below.

Labels for CGAN image generation: [0,2,4,6,8,1,3,5,7,9,3,6,9,1,5,7]

           CGAN              |          DCGAN

![alt text](https://github.com/SamarthGupta93/gan-variants-tf2.0/blob/master/images/cgan_mnist_resized.gif "Training visualization through gif") ![alt text](https://github.com/SamarthGupta93/gan-variants-tf2.0/blob/master/images/dcgan_mnist_resized.gif "DCGAN Training visualization through gif")

[CGAN gif images](https://github.com/SamarthGupta93/gan-variants-tf2.0/tree/master/conditional_gan/generated_images) | [DCGAN gif images](https://github.com/SamarthGupta93/gan-variants-tf2.0/tree/master/dcgan/generated_images/lr_1e-4)

### Requirements
1. Tensorflow v2
2. Numpy
3. Matplotlib for plotting (optional)
4. imagio for gif creation (optional)

### References
1. [DCGAN](https://arxiv.org/abs/1511.06434) paper
2. [CGAN] (https://arxiv.org/abs/1411.1784) paper
2. [Tensorflow tutorial](https://www.tensorflow.org/beta/tutorials/generative/dcgan)
