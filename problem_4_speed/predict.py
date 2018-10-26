from argparse import ArgumentParser
from collections import deque
from matplotlib import pyplot as plt

import numpy as np

import tensorflow as tf
from tensorflow import keras
import pickle

def get_data(data_file):
	PAD = 3
	INPUT_N = PAD*2+1

	with open(data_file, 'rb') as f:
		data = pickle.load(f)
		padding_val = [[0,0] for i in range(len(data[0]))]
		data = [padding_val for _ in range(PAD)] + data + [padding_val for _ in range(PAD)]

		temp = []
		for i in range(len(data) - INPUT_N+1):
			temp.append(data[i:i+INPUT_N])
		train_data = np.array(temp)


	print("Data: {}".format(train_data.shape))

	return train_data

if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('data_file', type=str)
	parser.add_argument('output_file', type=str)
	args = parser.parse_args()

	data = get_data(args.data_file)

	model = keras.models.load_model('model.h5')
	model.summary()

	preds = model.predict(data).flatten()

	with open(args.output_file, 'wt') as f:
		f.write('0\n')
		for v in preds:
			f.write(str(v) + '\n')
