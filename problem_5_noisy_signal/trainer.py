from argparse import ArgumentParser
from matplotlib import pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow import keras

def get_data():
	INPUT_N = 60
	data = []

	for i in range(1,6):
		f_real = open('Tests/Test' + str(i) + '_real')
		f_noisy = open('Tests/Test' + str(i))

		noisy_values = [0 for _ in range(INPUT_N-1)] + [float(i) for i in f_noisy.readlines()]
		real_values = [0 for _ in range(INPUT_N-1)] + [float(i) for i in f_real.readlines()]

		f_real.close()
		f_noisy.close()

		for j in range(len(noisy_values) - INPUT_N):
			data.append((noisy_values[j:j+INPUT_N], real_values[j+INPUT_N]))

	return data

def split_data(data):
	np.random.shuffle(data)

	train_data = []
	train_labels = []
	test_data = []
	test_labels = []

	TRAIN_RATIO = 1
	TRAIN_INDEX = int(len(data) * TRAIN_RATIO)

	for i in range(TRAIN_INDEX):
		train_data.append(data[i][0])
		train_labels.append(data[i][1])

	for i in range(TRAIN_INDEX, len(data)):
		test_data.append(data[i][0])
		test_labels.append(data[i][1])

	train_data = np.array(train_data)
	train_labels = np.array(train_labels)
	test_data = np.array(test_data)
	test_labels = np.array(test_labels)

	return (train_data, train_labels, test_data, test_labels)

def build_model():
	model = keras.Sequential([
		keras.layers.Dense(30, activation=tf.nn.tanh,
			input_shape=(train_data.shape[1],)),
		keras.layers.Dense(10, activation=tf.nn.tanh),
		keras.layers.Dense(5, activation=tf.nn.tanh),
		keras.layers.Dense(1)
	])

	optimizer = tf.train.RMSPropOptimizer(0.001)

	model.compile(loss='mse',
		optimizer=optimizer,
		metrics=['mae'])

	return model

class PrintProgress(keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs):
		if epoch % 100 == 0: print('')
		print('.', end='', flush=True)

def plot_history(history):
	plt.figure()
	plt.xlabel('Epoch')
	plt.ylabel('Mean Abs Error')
	plt.plot(history.epoch, np.array(history.history['mean_absolute_error']), label='Train Loss')
	plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']), label = 'Val loss')
	plt.legend()
	plt.show()

if __name__ == "__main__":
	# parser = ArgumentParser()
	# parser.add_argument('test_no', metavar='N', type=int, help='Test number.')
	# args = parser.parse_args()

	# test_no = args.test_no
	# assert 1 <= test_no <= 5, "Invalid test number."
	# f_real = open('Tests/Test' + str(test_no) + '_real')
	# f_noisy = open('Tests/Test' + str(test_no))

	# noisy_values = [float(i) for i in f_noisy.readlines()]
	# real_values = [float(i) for i in f_real.readlines()]

	# f_real.close()
	# f_noisy.close()

	# n = len(noisy_values)
	# plt.plot(np.linspace(1, n, n), noisy_values, linewidth=1.0, color='blue')
	# plt.plot(np.linspace(1,n,n), real_values, linewidth=1.0, color='red')
	# plt.show()
	data = get_data()
	(train_data, train_labels, test_data, test_labels) = split_data(data)

	print("Training set: {}".format(train_data.shape))
	print("Testing set: {}".format(test_data.shape))

	model = build_model()
	model.summary()

	EPOCHS = 500
	early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
	history = model.fit(train_data, train_labels, epochs=EPOCHS, validation_split=0.2, verbose=0, callbacks=[early_stop, PrintProgress()])

	model.save('model.h5')

	plot_history(history)



