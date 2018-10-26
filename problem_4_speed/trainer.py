from argparse import ArgumentParser
import numpy as np
import pickle
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow import keras

def get_data(data_file, label_file):
	PAD = 3
	INPUT_N = PAD*2+1

	with open(data_file, 'rb') as f:
		data = pickle.load(f)
		padding_val = [[0,0] for i in range(len(data[0]))]
		data = [padding_val for _ in range(PAD)] + data + [padding_val for _ in range(PAD)]

		temp = []
		for i in range(len(data) - INPUT_N + 1):
			temp.append(data[i:i+INPUT_N])
		train_data = np.array(temp)

	with open(label_file, 'r') as f:
		labels = [float(i) for i in f.readlines()]
		labels = labels[1:]
		labels = [0 for _ in range(PAD)] + labels + [0 for _ in range(PAD)]

		train_labels = []
		for i in range(len(labels) - INPUT_N + 1):
			train_labels.append(labels[i+PAD])
		train_labels = np.array(train_labels)

	print("Training data: {}".format(train_data.shape))
	print("Training labels: {}".format(train_labels.shape))

	return (train_data, train_labels)

class PrintProgress(keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs):
		if epoch % 100 == 0: print('')
		print('.', end='', flush=True)

def plot_history(history):
	plt.figure()
	plt.xlabel('Epoch')
	plt.ylabel('Mean Squared Error')
	plt.plot(history.epoch, np.array(history.history['mean_squared_error']), label='Train Loss')
	plt.plot(history.epoch, np.array(history.history['val_mean_squared_error']), label = 'Val loss')
	plt.legend()
	plt.show()

def build_model(input_shape):
	model = keras.Sequential([
		keras.layers.Flatten(input_shape=input_shape),
		keras.layers.Dense(1164, activation=tf.nn.tanh,
			input_shape=(train_data.shape[1],)),
		keras.layers.Dense(100, activation=tf.nn.tanh),
		keras.layers.Dense(50, activation=tf.nn.tanh),
		keras.layers.Dense(10, activation=tf.nn.tanh),
		keras.layers.Dense(1)
	])

	optimizer = tf.train.RMSPropOptimizer(0.0005)

	model.compile(loss='mse',
		optimizer=optimizer,
		metrics=['mse'])

	return model

if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('data_file', type=str)
	parser.add_argument('label_file', type=str)
	args = parser.parse_args()

	(train_data, train_labels) = get_data(args.data_file, args.label_file)

	model = build_model(train_data.shape[1:])
	model.summary()

	EPOCHS = 500
	early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
	history = model.fit(train_data, train_labels, epochs=EPOCHS, validation_split=0.2, verbose=0, callbacks=[early_stop, PrintProgress()])

	model.save('model.h5')

	plot_history(history)

