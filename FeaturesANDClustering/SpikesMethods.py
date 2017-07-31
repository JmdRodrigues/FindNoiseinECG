from FeaturesANDClustering.PeakFinder import detect_peaks
import numpy as np

def Spikes(inputSignal, mph, edge="rising"):
	pks = detect_peaks(inputSignal, mph=mph)
	numPics = len(pks)
	if(len(pks)<2):
		meanDistance=0
	else:
		meanDistance = np.mean(np.diff(pks))

	return numPics, meanDistance