from argparse import ArgumentParser
import numpy as np
import cv2
import pickle

npoints = 7 

if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('fin', type=str);
	parser.add_argument('fout', type=str)
	args = parser.parse_args()

	cap = cv2.VideoCapture(args.fin)

	lk_params = dict(winSize = (15,15), maxLevel=2,
		criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

	ret,old_frame = cap.read()
	old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

	imin = 0
	imax = old_frame.shape[0] - 100
	jmin = 0
	jmax = old_frame.shape[1]

	di = (imax-imin) / float(npoints)
	dj = (jmax-jmin) / float(npoints)

	points = []
	for i in range(npoints):
		for j in range(npoints):
			py = di/2+di*i + imin
			px = dj/2+dj*j + jmin
			points.append([px, py])
			cv2.circle(old_gray, (int(px),int(py)), 5, (255,), -1)


	cv2.imshow('a', old_gray)
	cv2.waitKey(1)

	
	p0 = np.array(points, dtype=np.float32)

	optflows = []

	count = 1
	while(1):
		ret,frame = cap.read()
		if not ret:
			break
		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		
		p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

		dp = p1 - p0

		optflows.append(dp)

		old_gray = frame_gray.copy()
		if count % 100 == 0:
			print(count)
		count += 1

	with open(args.fout, 'wb') as f:
		pickle.dump(optflows, f)

