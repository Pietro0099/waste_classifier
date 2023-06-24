import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from termcolor import colored

# Model evaluation
def test_model(model, test_ds):

	test_loss, test_accuracy = model.evaluate(test_ds)
	print(colored(f"Test Loss: {test_loss}", 'green', attrs=['bold']))
	print(colored(f"Test Accruacy: {test_accuracy}", 'green', attrs=['bold']))

# Confusion Matrix
def plot_cm(model, ds, classes, dir_path):

	y_pred = np.array([], dtype=int)
	y_true = np.array([], dtype=int)
	labels_true = []
	labels_pred = []
	for x, y in ds:
		y_pred = np.concatenate([y_pred, np.argmax(model.predict(x), axis=-1)])
		y_true = np.concatenate([y_true, np.argmax(y.numpy(), axis=-1)])
	for label in y_true:
		labels_true.append(classes[label])
	for label in y_pred:
		labels_pred.append(classes[label])

	_, test_acc = model.evaluate(ds)
	test_acc = f"{np.round(test_acc, 4)*100}%"
	cm = confusion_matrix(labels_true, labels_pred, labels=classes)
	sns.heatmap(
		cm,
		annot=True,
		cmap='rocket',
		fmt='g',
		xticklabels=classes,
		yticklabels=classes
	)
	plt.xlabel('Predicted')
	plt.ylabel('True')

	# Add the test accuracy value to the title with red and bold font
	plt.title(f'Confusion Matrix        Test Acc: {test_acc}')


	plt.savefig(f"{dir_path}/confusion_matrix")
	plt.close("all")
