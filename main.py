import tensorflow as tf
from keras.models import load_model

import os
import yaml
from termcolor import colored

from model.architecture.model import CNN_MODEL
from model.train import train
from model.test import test

from scripts.ds_ops import dataset
from scripts.model_compress import lite_converter, prune
from scripts.utils import utils


# Set path to the config for main
environment_path = "environment.yaml"

# Load environment from .yaml file
with open(environment_path, "r") as file:
	environment = yaml.safe_load(file)


# Access the environment
# Load all the directories path
try:
	dir_path = environment["dir_path"]

	ds = dir_path["ds"]
	config = dir_path["config"]

	ds_train_dir = os.path.join(ds["ds_dir"], ds["train_dir"])
	ds_test_dir = os.path.join(ds["ds_dir"], ds["test_dir"])

	main_settings_path = os.path.join(config["config_dir"], config["main_settings_file"])
	dataset_settings_path = os.path.join(config["config_dir"], config["dataset_settings_file"])
	model_settings_path = os.path.join(config["config_dir"], config["model_settings_file"])

	saved_models_dir = dir_path["saved_models"]
	tensorboard_log_dir = dir_path["tensorboard_log"]
	classification_dir = dir_path["classification"]

except Exception as e:
	print(colored(f"Unable to load environment from: {environment_path}", 'red', attrs=['bold']))
	print(colored(f"Error: {e}", 'red', attrs=['bold']))


# Access main settings
try:
	# Load main settings from .yaml file
	with open(main_settings_path, "r") as file:
		main_settings = yaml.safe_load(file)

	# Global settings
	global_settings = main_settings["global_settings"]
	mode = global_settings["mode"]
	from_pretrained = global_settings["from_pretrained"]
	seed = global_settings["seed"]
	set_memory_growth = global_settings["set_memory_growth"]
	tf_stfu = global_settings["tf_stfu"]

except Exception as e:
	print(colored(f"Unable to load main settings from {main_settings_path}", 'red', attrs=['bold']))
	print(colored(f"Error: {e}", 'red', attrs=['bold']))


# Access dataset settings
try:
	# Load dataset settings from .yaml file
	with open(dataset_settings_path, "r") as file:
		dataset_settings = yaml.safe_load(file)

	# ds settings
	ds_settings = dataset_settings["ds_settings"]

	# Augmentation settings
	augmentation_settings = dataset_settings["augmentation_settings"]

	# Load common dataset settings
	classes = ds_settings["classes"]
	num_classes = ds_settings["num_classes"]
	img_shape = ds_settings["img_shape"]
	rescale = ds_settings["rescale"]
	batch_size = ds_settings["batch_size"]
	val_split = ds_settings["val_split"]

except Exception as e:
	print(colored(f"Unable to load dataset settings from {dataset_settings_path}", 'red', attrs=['bold']))
	print(colored(f"Error: {e}", 'red', attrs=['bold']))


# Access model settings
try:
	# Load model settings from .yaml file
	with open(model_settings_path, "r") as file:
		model_settings = yaml.safe_load(file)

	# Model settings
	model_architecture = model_settings["model_architecture"]
	model_training = model_settings["model_training"]

except Exception as e:
	print(colored(f"Unable to load model settings from {model_settings_path}", 'red', attrs=['bold']))
	print(colored(f"Error: {e}", 'red', attrs=['bold']))


# Set global variables
# Global seed
tf.keras.utils.set_random_seed(seed)

# Verify access to GPU
devices = tf.config.list_physical_devices("GPU")
print(devices)

# Use memory growth
if set_memory_growth:
	tf.config.experimental.set_memory_growth(devices[0], True)

# Make tf stfu
if tf_stfu:
	os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


# Select mode
if mode == "training":

	# Setup new model
	model_name = input(colored("Enter new model name: ", 'cyan', attrs=['bold']))
	model_path = os.path.join(saved_models_dir, model_name)
	os.mkdir(model_path)

	# Load data
	train_ds, val_ds, test_ds = dataset.data_loader(
		train_dir=ds_train_dir,
		test_dir=ds_test_dir,
		batch_size=batch_size,
		img_shape=img_shape,
		val_split=val_split,
		seed=seed
	)

	# Preprocess data
	train_ds = dataset.preprocess_augment(
		train_ds,
		rescale=rescale,
		**augmentation_settings
	)
	val_ds = dataset.preprocess_augment(val_ds, rescale=rescale, use_augmentation=False)
	test_ds = dataset.preprocess_augment(test_ds, rescale=rescale, use_augmentation=False)

	# Initiate model by loading a pretrained model or instantiate a new one
	if from_pretrained:
		pretrained_model_name = input(colored("Enter pretrained model name: ", 'cyan', attrs=['bold']))
		pretrained_model_path = os.path.join(saved_models_dir, pretrained_model_name)
		model = load_model(pretrained_model_path)
	else:
		model = CNN_MODEL(
			num_classes=num_classes,
			**model_architecture
		)

	# Train model
	history = train.run_experiment(
		model=model,
		train_ds=train_ds,
		val_ds=val_ds,
		save_model_path=model_path,
		tb_log_dir=tensorboard_log_dir,
		model_name=model_name,
		batch_size=batch_size,
		**model_training
	)
	# Save history plot
	utils.plot_history(history, model_path)

	# Test model
	test.test_model(model, test_ds)
	test.plot_cm(model, test_ds, classes, model_path)
	model.summary()


elif mode == "inference":

	# Load model for inference
	model_name = input(colored("Enter model name: ", 'cyan', attrs=['bold']))
	model_path = os.path.join(saved_models_dir, model_name, model_name)
	model = load_model(model_path)

	# Run inference on saved images
	utils.classification(
		classification_dir=classification_dir,
		model=model,
		class_names=classes,
		img_shape=img_shape,
		rescale=rescale
	)

elif mode == "prune":

	# Setup model to prune
	model_name = input(colored("Enter model to convert: ", 'cyan', attrs=['bold']))
	model_path = os.path.join(saved_models_dir, model_name)
	model_path_pruned = model_name + "_pruned"
	model_path_pruned = os.path.join(model_path, model_path_pruned)
	model_path = os.path.join(model_path, model_name)
	model = load_model(model_path)

	# Load the training data
	train_ds, val_ds, _ = dataset.data_loader(
		train_dir=ds_train_dir,
		test_dir=ds_test_dir,
		batch_size=batch_size,
		img_shape=img_shape,
		val_split=val_split,
		seed=seed
	)

		# Preprocess data
	train_ds = dataset.preprocess_augment(
		train_ds,
		rescale=rescale,
		**augmentation_settings
	)
	val_ds = dataset.preprocess_augment(val_ds, rescale=rescale, use_augmentation=False)

	# Apply pruning to the model
	model = prune.prune_model(
		model=model,
		save_path=model_path_pruned,
		train_ds=train_ds,
		val_ds=val_ds,
		batch_size=batch_size,
		log_dir=tensorboard_log_dir,
		num_epochs=2,
		initial_sparsity=0.5,
		final_sparsity=0.8
	)


elif mode == "convert":

	# Load and setup model for conversion to tflite
	model_name = input(colored("Enter model to convert: ", 'cyan', attrs=['bold']))
	pruned = input(colored(f"Convert pruned version of {model_name}? [y/n]: ", 'cyan', attrs=['bold']))
	model_path = os.path.join(saved_models_dir, model_name)
	if pruned == "y":
		model_name = model_name + "_pruned"
	lite_model_path = model_name + ".tflite"
	lite_model_path = os.path.join(model_path, lite_model_path)
	model_path = os.path.join(model_path, model_name)
	model = load_model(model_path)

	# Define a Representative Dataset
	_, _, test_ds = dataset.data_loader(
		train_dir=ds_train_dir,
		test_dir=ds_test_dir,
		batch_size=batch_size,
		img_shape=img_shape,
		val_split=val_split,
		seed=seed
	)
	test_ds = dataset.preprocess_augment(test_ds, rescale=rescale, use_augmentation=False)
		
	def representative_datagen():
		for img, _ in test_ds.take(1):
			yield [tf.cast(img, tf.float32)]

	# Convert model to tflite
	lite_model = lite_converter.convert(model=model, representative_datagen=representative_datagen)

	# Save tflite model
	with open(lite_model_path, "wb") as f:
		f.write(lite_model)

else:
	raise Exception(colored(f"Invalid mode selected.\n Excpected modes: (training, inference, prune, convert).\n Received: {mode}", 'red', attrs=['bold']))