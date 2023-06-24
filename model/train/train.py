import tensorflow as tf
from keras import callbacks
from keras.callbacks import LearningRateScheduler
from keras import losses
import os, time


class WarmupLearningRateScheduler(LearningRateScheduler):
	def __init__(self, warmup_epochs=5, initial_lr=1e-5, warmup_lr=3e-4, **kwargs):
		super().__init__(self.warmup_learning_rate_scheduler, **kwargs)
		self.warmup_epochs = warmup_epochs
		self.initial_lr = initial_lr
		self.warmup_lr = warmup_lr

	def warmup_learning_rate_scheduler(self, epoch, lr):
		if epoch < self.warmup_epochs:
			lr = (self.warmup_lr - self.initial_lr) * (epoch / self.warmup_epochs) + self.initial_lr
		return lr


# Train model
def run_experiment(model, train_ds, val_ds, num_epochs, batch_size, save_model_path, tb_log_dir, model_name, optimizer_lr=3e-4, label_smoothing=None, reduce_lr_plateau_patience=3, monitor="val_accuracy", warmup_epochs=5, initial_lr=1e-5, warmup_lr=3e-4):

	# Define Tensorboard name for the model
	TB_NAME = "{}-{}".format(model_name, int(time.time()))
	TB_NAME = os.path.join(tb_log_dir, TB_NAME)
	save_model_path = os.path.join(save_model_path, model_name)

	optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate=optimizer_lr)
	loss = losses.CategoricalCrossentropy(label_smoothing=label_smoothing)

	# Compile the model
	model.compile(
		optimizer=optimizer,
		loss=loss,
		metrics=["accuracy"],
	)

	# Define the callbacks
	checkpoint = callbacks.ModelCheckpoint(save_model_path, monitor=monitor, save_best_only=True)
	reduce_lr = callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.1, patience=reduce_lr_plateau_patience)
	early_stop = callbacks.EarlyStopping(monitor="val_accuracy", patience=10)
	lr_scheduler = WarmupLearningRateScheduler(warmup_epochs, initial_lr, warmup_lr)
	tensorboard_callback = callbacks.TensorBoard(
		log_dir=TB_NAME,
		histogram_freq=1,
		write_graph=False
	)
	cbks = [checkpoint, reduce_lr, early_stop, lr_scheduler, tensorboard_callback]

	history = model.fit(
		train_ds,
		validation_data=val_ds,
		epochs=num_epochs,
		batch_size=batch_size,
		callbacks=cbks
	)

	return history