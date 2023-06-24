from keras.models import save_model
import tensorflow_model_optimization as tfmot



def prune_model(model, save_path, train_ds, val_ds, batch_size, log_dir, num_epochs=2, initial_sparsity=0.5, final_sparsity=0.8):

	prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
	pruning_params = {
		'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
			initial_sparsity=initial_sparsity,
			final_sparsity=final_sparsity,
			begin_step=0,
			end_step=147
			)
	}

	model = prune_low_magnitude(model, **pruning_params)
	model.compile(
		optimizer="Adam",
		loss="CategoricalCrossentropy",
		metrics=["accuracy"],
	)

	cbks = [
		tfmot.sparsity.keras.UpdatePruningStep(),
		tfmot.sparsity.keras.PruningSummaries(log_dir=log_dir)
	]

	model.fit(
		train_ds,
		validation_data=val_ds,
		epochs=num_epochs,
		batch_size=batch_size,
		callbacks=cbks
	)
	model = tfmot.sparsity.keras.strip_pruning(model)
	save_model(model, filepath=save_path, save_format="h5", include_optimizer=False)
