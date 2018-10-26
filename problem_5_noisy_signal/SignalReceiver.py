import argparse
from collections import deque
from matplotlib import pyplot as plt

import numpy as np

import tensorflow as tf
from tensorflow import keras

parser = argparse.ArgumentParser(description='Test number')
parser.add_argument('test_no', metavar='N', type=int, help='Test number.')


class SignalReceiver:
	def __init__(self, test_no):
		assert 1 <= test_no <= 5, "Invalid test number."
		f_real = open('Tests/Test' + str(test_no) + '_real')
		f_noisy = open('Tests/Test' + str(test_no))

		self.__noisy_values = [float(i) for i in f_noisy.readlines()]
		self.__real_values = [float(i) for i in f_real.readlines()]
		self.__c_index = 0
		self.__total_error = 0

	def get_value(self):
		'''
		Gets next noisy value from device. This must be called before push_value.
		:return: (float) device value, None if the device is closed.
		'''
		if self.__c_index >= len(self.__noisy_values):
			return None

		val = self.__noisy_values[self.__c_index]
		self.__c_index = self.__c_index + 1
		return val

	def push_value(self, c_val):
		'''
		Computes the error between the real signal and the corrected value.
		:param c_val: corrected value.
		:return: (float) error value, None if the device is closed.
		'''

		if self.__c_index - 1 >= len(self.__real_values):
			return None

		error = abs(self.__real_values[self.__c_index - 1] - c_val)
		self.__total_error += error
		return error

	def get_error(self):
		return self.__total_error


if __name__ == "__main__":
	'''
	Dumb example of usage.
	'''
	args = parser.parse_args()
	sr = SignalReceiver(args.test_no)

	model = keras.models.load_model('model.h5')
	model.summary()

	vals = deque([0 for _ in range(model.input_shape[1])])

	noisy_values = []
	i_val = sr.get_value()

	pred_values = []
	while i_val:
		noisy_values.append(i_val)
		vals.popleft()
		vals.append(i_val)

		prediction = model.predict(np.array([vals])).flatten()
		pred_values.append(prediction[0])
		
		sr.push_value(prediction[0])
		i_val = sr.get_value()

	print('Total error: ' + str(sr.get_error()))

	n = len(noisy_values)
	plt.plot(np.linspace(1,n,n), noisy_values, color='blue')
	plt.plot(np.linspace(1,n,n), pred_values, color='red')
	plt.show()
