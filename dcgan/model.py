import numpy as np
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Activation, Reshape, Dropout, Flatten, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import time
import tensorflow as tf

class DCGAN():
	def __init__(self,gen_lr=1e-4,disc_lr=1e-4,leaky_relu=0.2,batch_size=32,noise_dim=100):
		self.leaky_relu_slope=leaky_relu
		self.gen_optimizer=Adam(gen_lr)
		self.disc_optimizer=Adam(disc_lr)
		self.cross_entropy = BinaryCrossentropy(from_logits=True)
		self.batch_size=batch_size
		self.noise_dim=noise_dim


	def set_noise_seed(self,num_examples_to_generate):
		 # We will use this noise to test the generator output at regular epochs of training
		self.noise_seed = tf.random.normal([num_examples_to_generate,self.noise_dim])


	def create_generator(self):
		input_noise = Input(shape=(100,))
		X = Dense(units=7*7*256, use_bias=False)(input_noise)
		X = BatchNormalization()(X)
		X = LeakyReLU(self.leaky_relu_slope)(X)
		X = Reshape(target_shape=(7,7,256))(X)
		
		# First layer of Conv2DTranspose (Does not do any upsampling as in tensorflow tutorial. Compare by upsampling instead)
		X = Conv2DTranspose(filters=128,kernel_size=5,strides=(1,1),padding='same', use_bias=False)(X)
		X = BatchNormalization()(X)
		X = LeakyReLU(self.leaky_relu_slope)(X)

		# Second layer of Conv2DTranspose (Upsampling) (14*14)
		X = Conv2DTranspose(filters=64, kernel_size=5, strides=(2,2),padding='same',use_bias=False)(X)
		X = BatchNormalization()(X)
		X = LeakyReLU(self.leaky_relu_slope)(X)

		# Third layer of Conv2DTranspose (Upsampling) (28*28)
		gen_output = Conv2DTranspose(filters=1, kernel_size=5, strides=(2,2),padding='same',use_bias=False, activation='tanh')(X)

		self.generator = Model(inputs=input_noise, outputs=gen_output)
		print(self.generator.summary())


	def create_discriminator(self):
		input_image = Input(shape=(28,28,1))
		# Layer 1
		X = Conv2D(filters=64,kernel_size=5,strides=(2,2),padding='same')(input_image)
		X = LeakyReLU(self.leaky_relu_slope)(X)
		X = Dropout(0.3)(X)
		# Layer 2
		X = Conv2D(filters=128,kernel_size=5,strides=(2,2),padding='same')(X)
		X = LeakyReLU(self.leaky_relu_slope)(X)
		X = Dropout(0.3)(X)
		# Output Layer
		X = Flatten()(X)
		output_prob = Dense(units=1)(X)

		self.discriminator=Model(inputs=input_image,outputs=output_prob)
		print(self.discriminator.summary())


	def generator_loss(self,fake_output):
		# Generator has to fool to discriminator into predicting generated image as real.
		# fake_output is the disc probability when generated image is passed. 
		# For generated to fool the disc, fake_output should be as close to 1 as poosible.
		return self.cross_entropy(tf.ones_like(fake_output),fake_output)


	def discriminator_loss(self, real_output, fake_output):
		# Discriminator has to maximize the probability of assigning the correct label.
		# Real/Fake output is the probability of predicting the input as real.
		# real_output is the disc probability when real image is passed. Disc should pred it as close to 1.
		# fake_output is the disc probability when generated image is passed. Disc should pred it as close to 0.
		real_image_loss = self.cross_entropy(tf.ones_like(real_output),real_output)
		fake_image_loss = self.cross_entropy(tf.zeros_like(fake_output),fake_output)
		return (real_image_loss+fake_image_loss)


	def set_checkpoint(self,path):
		self.checkpoint_prefix = path
		self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.gen_optimizer,
								discriminator_optimizer=self.disc_optimizer,
								generator=self.generator, discriminator=self.discriminator)
	
	# tf.function causes the function to be "compiled"
	@tf.function
	def train_step(self,images):
		# Generate Noise for generator 
		noise = tf.random.normal([self.batch_size,self.noise_dim])

		with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
			# Generator pass
			generated_images = self.generator(noise)
			# Discriminator pass
			fake_output = self.discriminator(generated_images)
			real_output = self.discriminator(images)
			# Calculate discriminator and generator losses
			disc_loss = self.discriminator_loss(real_output,fake_output)
			gen_loss = self.generator_loss(fake_output)
			
		# Get gradients for backpropagation
		generator_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
		discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

		self.gen_optimizer.apply_gradients(zip(generator_gradients,self.generator.trainable_variables))
		self.disc_optimizer.apply_gradients(zip(discriminator_gradients,self.discriminator.trainable_variables))

		return gen_loss, disc_loss


	def train(self,train_images,epochs):
		n_batches=0
		gen_loss_array = []
		disc_loss_array = []
		self.generate_and_save_images(epoch=0,noise=self.noise_seed)
		# Start training
		for epoch in range(epochs):
			start = time.time()
			gen_loss_total = 0
			disc_loss_total = 0
			# Train each batch
			for image_batch in train_images:
				gen_loss, disc_loss = self.train_step(image_batch)
				gen_loss_total = gen_loss_total+gen_loss
				disc_loss_total = disc_loss_total+disc_loss
				if epoch==0: n_batches+=1
			
			self.generate_and_save_images(epoch+1,self.noise_seed)

			# Save mode after every 15 epochs
			if (epoch+1)%5==0:
				self.checkpoint.save(file_prefix=self.checkpoint_prefix)

			print("-------------- Epoch {} -------------".format(epoch+1))
			print("Generator Loss {}".format(gen_loss_total/n_batches))
			print("Discriminator Loss {}".format(disc_loss_total/n_batches))
			print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

			gen_loss_array.append(gen_loss_total/n_batches)
			disc_loss_array.append(disc_loss_total/n_batches)

		return gen_loss_array,disc_loss_array


	def generate_and_save_images(self,epoch,noise):
		generated_images = self.generator(noise)

		fig = plt.figure(figsize=(4,4))
		for i in range(generated_images.shape[0]):
			plt.subplot(4,4,i+1)
			plt.imshow((generated_images[i,:,:,0]*127.5 + 127.5), cmap='gray')
			plt.axis('off')

		plt.savefig('generated_images/images_at_epoch_{}.png'.format(epoch))
		#plt.show()
		plt.close(fig)


