import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import glob
import math
from PIL import Image
import os



# Plot a grid of 9 images
def plot_examples(ds, class_names, scale=(0, 255)):
	plt.figure(figsize=(10, 10))
	for images, labels in ds.take(1):
		for i in range(9):
			ax = plt.subplot(3, 3, i + 1)
			if scale == [-1, 1]:
				plt.imshow(((images[i].numpy()+1)*127.5).astype("uint8"))
			elif scale == [0, 1]:
				plt.imshow((images[i].numpy()*255.0).astype("uint8"))
			elif scale == [0, 255]:
				plt.imshow(images[i].numpy().astype("uint8"))
			else:
				raise Exception(f"Scaling expected to be [-1, 1], [0, 1], [0, 255]. Received:({scale})")
			plt.title(class_names[tf.argmax(labels[i])])
			plt.axis("off")
	plt.show()
	

def plot_history(history, dir_path):

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

	# Plot training & validation accuracy values
	ax1.plot(history.history['accuracy'])
	ax1.plot(history.history['val_accuracy'])
	ax1.set_title('Model accuracy')
	ax1.set_ylabel('Accuracy')
	ax1.set_xlabel('Epoch')
	ax1.legend(['Train', 'Validation'], loc='upper left')

	# Plot training & validation loss values
	ax2.plot(history.history['loss'])
	ax2.plot(history.history['val_loss'])
	ax2.set_title('Model loss')
	ax2.set_ylabel('Loss')
	ax2.set_xlabel('Epoch')
	ax2.legend(['Train', 'Validation'], loc='upper left')

	fig_path = os.path.join(dir_path, 'history.png')
	fig.savefig(fig_path)
	plt.close("all")


def classification(classification_dir, model, class_names, img_shape, rescale=[0, 255]):
	# Read in the images
	classification_dir = os.path.join(classification_dir, "*.jpg")
	imgs_path = glob.glob(classification_dir)

	# Define the number of rows and columns for the subplot grid
	num_cols = 3
	num_rows = math.ceil(len(imgs_path)/num_cols)
	subplot_num = 1

	# Create a figure and set the size
	fig = plt.figure(figsize=(10, 10))
	for img_path in imgs_path:
		# Load images
		img = Image.open(img_path)
		img = img.resize(img_shape[:-1])
		x = tf.keras.preprocessing.image.img_to_array(img)
		# Rescale the images
		if rescale == [-1, 1]:
			x = x / 127.5 - 1
		elif rescale == [0, 1]:
			x = x / 255.0
		elif rescale == [0, 255]:
			pass
		else:
			raise Exception(f"Scaling expected to be (-1, 1), (0, 1), (0, 255). Received:({rescale})")
		# Get the predicted label for the image
		prediction = model.predict(np.expand_dims(x, axis=0))
		predicted_label = class_names[np.argmax(prediction)]
		# Create a subplot for the image and set the title
		ax = fig.add_subplot(num_rows, num_cols, subplot_num)
		ax.imshow(img)
		disp_prd = round(np.max(prediction*100), 2).astype(str)
		ax.set_title(f"Prediction: {predicted_label} ~ {disp_prd}%")
		ax.axis("off")

		# Increment the subplot counter
		subplot_num += 1

	# Show the plot
	plt.tight_layout()
	plt.show()
