# The speed challenge

I attempt to train a neural network to predict the speed, using the optical in a few sparse points as input, over several frames.

If the test_video is the one given initially, just run: `python3 predict.py test.pkl`, or just read the results in `train.txt`.

First, the videos need to be preprocessed with: `python3 preprocess.py <Video_file> <Data_file_out>`

Second, run: `python3 predict.py <Data_file>`