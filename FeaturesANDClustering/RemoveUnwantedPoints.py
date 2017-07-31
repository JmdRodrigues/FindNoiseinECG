import numpy as np
from scipy.spatial import distance

def RemoveUglyDots(Matrix):

	if(len(Matrix) < 2000):
		euclideanDist = distance.cdist(Matrix, Matrix, 'euclidean')
		euclideanDist = euclideanDist/euclideanDist.max()

		dispertiontemp = sum(euclideanDist)
		#print(dispertiontemp)
		dispertion = sum(dispertiontemp)/len(euclideanDist)**2

		#print(dispertion)

		distThreshold = dispertion/10
		PointsThreshold = 2/dispertion

		densTemp = np.array([])
		TemporDens = np.array([0])
		for i in range(0, len(euclideanDist)):
			distTemp = np.where(euclideanDist[i][:] < distThreshold)[0]
			TemporalDist = len(np.where(np.logical_and(distTemp < i + 5, distTemp > i - 5))[0])
			if(np.logical_and(TemporalDist > 5, i > TemporDens[-1] + 10)):
				TemporDens = np.append(TemporDens, i)
			NumberofPoints = len(distTemp)
			densTemp = np.append(densTemp, NumberofPoints)

		#density = np.where(np.logical_and(densTemp > PointsThreshold, TemporDens < 10))[0]
		density = np.where(densTemp > PointsThreshold)

	else:
		l = len(Matrix)
		density = 0
		TemporDens = np.array([0])
		for i in range(0, l, 100):
			TemporDens = np.append(TemporDens, i)


	return density, TemporDens