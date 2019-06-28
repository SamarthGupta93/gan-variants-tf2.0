import tensorflow as tf
import numpy as np
from model import CGAN
import matplotlib.pyplot as plt
import math
import imageio
import glob
from tensorflow.keras.utils import to_categorical

EPOCHS=50
BUFFER_SIZE=60000
BATCH_SIZE=256
gen_lr = 1e-3
disc_lr=2e-4
leaky_relu=0.1
num_examples_to_generate=16
NOISE_DIM=100
save_ckpt_path='checkpoints/ckpt_lr_2e-4'
model_test=False


def sigmoid(x):
	return 1/(1+math.exp(-x))


def run_gan():
	(train_images, train_labels),(_,_) = tf.keras.datasets.mnist.load_data()

	train_images = train_images.reshape(train_images.shape[0],28,28,1).astype('float32')
	train_images = (train_images-127.5)/127.5 # Normalize images to [-1,1]
	print(train_images.shape)

	train_labels = to_categorical(train_labels)
	print(train_labels.shape)

	# Batch and shuffle the data
	train_dataset = tf.data.Dataset.from_tensor_slices((train_images,train_labels)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

	gan = CGAN(gen_lr,disc_lr,noise_dim=NOISE_DIM)
	gan.create_generator()
	gan.create_discriminator()

	if model_test:
		# Test generator
		random_noise = tf.random.normal([1,NOISE_DIM])
		condition = tf.zeros(shape=(1,10))
		generated_image=gan.generator([random_noise,condition])
		plt.imshow(generated_image[0,:,:,0],cmap='gray')
		plt.show()
		# Test Discriminator
		prob = gan.discriminator([generated_image,condition])
		print("Probability of image being real: {}".format(sigmoid(prob)))

	gan.set_noise_seed(num_examples_to_generate)
	print(gan.label_seed.shape)
	gan.set_checkpoint(path=save_ckpt_path)
	gen_loss_array,disc_loss_array = gan.train(train_dataset,epochs=EPOCHS)

	# Plot Discriminator Loss
	plt.plot(range(EPOCHS),gen_loss_array)
	plt.plot(range(EPOCHS),disc_loss_array)
	plt.show()


def create_gif(image_dir,gif_name='cgan_mnist.gif'):
	with imageio.get_writer(image_dir+gif_name,mode='I') as writer:
		files = glob.glob(image_dir+'image*.png')
		files=sorted(files)
		last=-1
		for i,filepath in enumerate(files):
			frame=2*(i**0.5)
			if round(frame)>round(last):
				last=frame
			else:
				continue
			image = imageio.imread(filepath)
			writer.append_data(image)
		image = imageio.imread(filepath)
		writer.append_data(image)
	

if __name__=='__main__':
	create_gif(image_dir='generated_images/')