# gan-variants-tensorflow2.0
Tensorflow2.0 implementation of different Generative Adversarial Networks

### CGAN result
Conditional GANs can be used to generate a specific class of image out of all the classes in the dataset. It takes in an image label along with the noise as input and generates an image of the corresponding label. The visualization is shown below.
Labels for image generation: [0,2,4,6,8,1,3,5,7,9,3,6,9,1,5,7]
![alt text](https://github.com/SamarthGupta93/gan-variants-tf2.0/blob/master/cgan/generated_images/cgan_mnist.gif "Training visualization through gif")
Individual images in the gif can be found [here](https://github.com/SamarthGupta93/gan-variants-tf2.0/tree/master/cgan/generated_images)

### Requirements
1. Tensorflow v2
2. Numpy
3. Matplotlib for plotting (optional)
4. imagio for gif creation (optional)

### References
1. [DCGAN](https://arxiv.org/abs/1511.06434) 2016 paper
2. [Tensorflow tutorial](https://www.tensorflow.org/beta/tutorials/generative/dcgan)
