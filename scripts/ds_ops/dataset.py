import tensorflow as tf

# Loading train, val, test data
def data_loader(train_dir, test_dir, batch_size, img_shape, val_split=0.2, seed=42):

	if img_shape[-1] == 3:
		color_mode = "rgb"
	elif img_shape[-1] == 1:
		color_mode = "grayscale"
	else:
		raise Exception(f"Color Channels expected to be 1 or 3, received: {img_shape[-1]}")

	train_ds = tf.keras.preprocessing.image_dataset_from_directory(
		directory=train_dir,
		labels="inferred",
		label_mode="categorical",
		color_mode=color_mode,
		batch_size=batch_size,
		image_size=img_shape[:-1],
		shuffle=True,
		seed=seed,
		validation_split=val_split,
		subset="training"
	)

	val_ds = tf.keras.preprocessing.image_dataset_from_directory(
		directory=train_dir,
		labels="inferred",
		label_mode="categorical",
		color_mode=color_mode,
		batch_size=batch_size,
		image_size=img_shape[:-1],
		shuffle=True,
		seed=seed,
		validation_split=val_split,
		subset="validation"
	)

	test_ds = tf.keras.preprocessing.image_dataset_from_directory(
		directory=test_dir,
		labels="inferred",
		label_mode="categorical",
		color_mode=color_mode,
		batch_size=batch_size,
		image_size=img_shape[:-1],
		shuffle=False
	)

	return train_ds, val_ds, test_ds


def preprocess_augment(ds, rescale=[0, 255], use_augmentation=False, seed=42, flip_type=None, rotation=None, zoom=None, contrast=None, brightness=None, fill_mode=None):
	AUTOTUNE = tf.data.AUTOTUNE

	if use_augmentation:
		data_augmentation = tf.keras.Sequential([
			tf.keras.layers.RandomFlip(flip_type, seed=seed),
			tf.keras.layers.RandomRotation(rotation, fill_mode=fill_mode, seed=seed),
			tf.keras.layers.RandomZoom(zoom, fill_mode=fill_mode, seed=seed),
			tf.keras.layers.RandomContrast(contrast, seed=seed),
			tf.keras.layers.RandomBrightness(brightness, value_range=(0, 255), seed=seed),
		])
		ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)

	if rescale == [-1, 1]:
		rescale = tf.keras.Sequential([
			tf.keras.layers.Rescaling(scale=1./127.5, offset=-1)
		])
		ds = ds.map(lambda x, y: (rescale(x), y), num_parallel_calls=AUTOTUNE)
	elif rescale == [0, 1]:
		rescale = tf.keras.Sequential([
			tf.keras.layers.Rescaling(scale=1./255.0)
		])
		ds = ds.map(lambda x, y: (rescale(x), y), num_parallel_calls=AUTOTUNE)
	elif rescale == [0, 255]:
		pass
	else:
		raise Exception(f"Scaling expected to be [-1, 1], [0, 1], [0, 255]. Received:({rescale})")
	# Use buffered prefetching on all datasets. 
	return ds.prefetch(buffer_size=AUTOTUNE)


def get_img_shape(size):
	if size == "s":
		img_shape = [64, 64, 3]
	elif size == "m":
		img_shape = [96, 96, 3]
	elif size == "l":
		img_shape = [224, 224, 3]
	return img_shape



# Run this file to get a preview of same samples of the dataset
if __name__ == "__main__":

	import yaml
	import os

	import sys
	sys.path.append("scripts/utils")
	from utils import plot_examples


	# Set path to the config for main
	environment_path = "environment.yaml"

	# Load environment from .yaml file
	with open(environment_path, "r") as file:
		environment = yaml.safe_load(file)

	# Access the environment
	try:
		dir_path = environment["DIR_PATH"]

		ds = dir_path["DS"]
		ds_dir = ds["DS_DIR"]

		ds_train_dir = os.path.join(ds_dir, ds["TRAIN_DIR"])
		ds_test_dir = os.path.join(ds_dir, ds["TEST_DIR"])

	except Exception as e:
		print(f"Unable to load environment from: {environment_path}")
		print(f"Error: {e}")


	train_ds, val_ds, test_ds = data_loader(ds_train_dir, ds_test_dir, 64, [96, 96, 3])

	train_ds = preprocess_augment(
		train_ds,
		rescale=[0, 255],
		augment=True,
		flip_type="horizontal_and_vertical",
		rotation=0.2,
		zoom=[-0.1, 0.1],
		contrast=0.1,
		brightness=0.1,
		fill_mode="constant"
	)
	val_ds = preprocess_augment(val_ds, rescale=[0, 255], augment=False)
	test_ds = preprocess_augment(test_ds, rescale=[0, 255], augment=False)

	plot_examples(train_ds, class_names=["biological", "glass", "metal", "paper", "plastic"], scale=[0, 255])
	plot_examples(val_ds, class_names=["biological", "glass", "metal", "paper", "plastic"], scale=[0, 255])
	plot_examples(test_ds, class_names=["biological", "glass", "metal", "paper", "plastic"], scale=[0, 255])
