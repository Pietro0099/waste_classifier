import tensorflow as tf

from model.architecture.blocks import EntryBlock, CoreBlock, Classifier



class CNN_MODEL(tf.keras.Model):
	def __init__(self, num_classes, arch, stride2_blocks, activation, drop_rate, use_eca, use_fuse, use_skip):
		super(CNN_MODEL, self).__init__()

		self.arch = arch
		self.num_core_blocks = len(self.arch) - 2

		self.entry_block = EntryBlock(
			filters=self.arch[0],
			kernel_size=3,
			strides=2,
			activation=activation,
		)

		self.core_blocks = [
			CoreBlock(
				filters=self.arch[i],
				strides=2 if i in stride2_blocks else 1,
				use_eca=use_eca,
				use_fuse=use_fuse and i <= self.num_core_blocks // 3,
				use_skip=use_skip,
				activation=activation,
			)
			for i in range(1, self.num_core_blocks)
		]

		self.classifier = Classifier(
			num_classes=num_classes,
			filters=self.arch[-1],
			activation=activation,
			drop_rate=drop_rate,
		)

	def call(self, inputs, training=False):
		x = self.entry_block(inputs, training=training)
		for core_block in self.core_blocks:
			x = core_block(x, training=training)
		x = self.classifier(x, training=training)
		return x