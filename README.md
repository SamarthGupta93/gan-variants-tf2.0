# dcgan-tensorflow2.0
Tensorflow2.0 implementation of Deep Convolutional Generative Adversarial Network

![alt text](https://github.com/SamarthGupta93/dcgan-tensorflow2.0/blob/master/generated_images/lr_1e-4/dcgan_mnist.gif "Training visualization through gif")

Individual images can be found [here](https://github.com/SamarthGupta93/dcgan-tensorflow2.0/tree/master/generated_images/lr_1e-4)

### Model Architecture
Similar to the [DCGAN paper](https://arxiv.org/abs/1511.06434) except for few details changed.
1. LeakyReLU in both the generator and discriminator. Original paper uses ReLU in generator and LeakyReLU in discriminator
2. Adam optimizer learning rate is set as 1e-4. Original paper uses 2e-4
3. Default settings for Adam hyperparameters (beta1 & beta2). In the original paper, beta1 is set as 0.5

### Requirements
1. Tensorflow v2
2. Numpy
3. Matplotlib for plotting (optional)
4. imagio for gif creation (optional)

### References
1. [DCGAN](https://arxiv.org/abs/1511.06434) 2016 paper
2. [Tensorflow tutorial](https://www.tensorflow.org/beta/tutorials/generative/dcgan)
