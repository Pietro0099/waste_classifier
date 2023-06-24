import tensorflow as tf
from keras import layers

import numpy as np


class EntryBlock(layers.Layer):
	def __init__(self, filters, kernel_size, strides, activation, drop_rate=0.1, seed=42):
		super(EntryBlock, self).__init__()
		self.conv_module = tf.keras.Sequential([
			layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="valid", use_bias=False),
			layers.BatchNormalization(),
			layers.Activation(activation),
			layers.Dropout(drop_rate, seed=seed)
		])

	def call(self, inputs, training=False):
		x = self.conv_module(inputs, training=training)
		return x



class CoreBlock(layers.Layer):
	def __init__(self, filters, strides, use_eca, use_fuse, use_skip, activation, drop_rate=0.1, seed=42):
		super(CoreBlock, self).__init__()

		self.use_eca = use_eca
		self.use_fuse = use_fuse
		self.use_skip = use_skip
		self.strides = strides
		self.filters = filters

		if self.use_fuse:
			self.conv_module1 = tf.keras.Sequential([
				layers.Conv2D(filters=self.filters, kernel_size=3, strides=strides, padding="same"),
				layers.BatchNormalization(),
				layers.Activation(activation),
				layers.Dropout(drop_rate, seed=seed)
			])
		else:
			self.conv_module1 = tf.keras.Sequential([
				layers.Conv2D(filters=self.filters, kernel_size=1, padding="same"),
				layers.DepthwiseConv2D(kernel_size=3, strides=strides, padding="same"),
				layers.BatchNormalization(),
				layers.Activation(activation),
				layers.Dropout(drop_rate, seed=seed)
			])


		if self.use_eca:
			self.eca_module = tf.keras.Sequential([
				layers.GlobalAveragePooling2D(),
				layers.Lambda(lambda x: tf.expand_dims(x, axis=1)),
				layers.Conv1D(filters=1, kernel_size=self.get_kernel_size(self.filters), padding="same"),
				layers.Activation("sigmoid")
			])

		self.conv_module2 = tf.keras.Sequential([
			layers.Conv2D(filters=self.filters, kernel_size=1, padding="same"),
			layers.BatchNormalization(),
			layers.Activation(activation),
			layers.Dropout(drop_rate, seed=seed)
		])

	def get_kernel_size(self, filters):
		gamma = 2
		b = 1
		t = int(abs((np.log2(filters) + b) / gamma))
		k = t if t % 2 else t + 1
		return k

	def call(self, inputs, training=False):
		x = self.conv_module1(inputs, training=training)

		if self.use_eca:
			x_eca = self.eca_module(x)
			x = layers.Multiply()([x, x_eca])

		x = self.conv_module2(x, training=training)

		if self.use_skip and inputs.shape[-1] == x.shape[-1]:
			if self.strides == 2:
				inputs = layers.AveragePooling2D(padding="same")(inputs)
			x = layers.Add()([inputs, x])

		return x



class Classifier(layers.Layer):
	def __init__(self, num_classes, filters, activation, drop_rate=0.2, seed=42):
		super(Classifier, self).__init__()
		self.block = tf.keras.Sequential([
			layers.Conv2D(filters=filters, kernel_size=1, use_bias=False, padding="same"),
			layers.BatchNormalization(),
			layers.Activation(activation),
			layers.GlobalAveragePooling2D(keepdims=True),
			layers.Conv2D(filters=filters, kernel_size=1, padding="same"),
			layers.Activation(activation),
			layers.Conv2D(filters=filters, kernel_size=1, padding="same"),
			layers.Dropout(drop_rate, seed=seed),
			layers.Flatten(),
			layers.Dense(units=num_classes, activation="softmax")
		])

	def call(self, inputs, training=False):
		x = self.block(inputs, training=training)
		return x

