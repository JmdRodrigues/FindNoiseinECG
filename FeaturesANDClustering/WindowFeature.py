import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import signaltonoise, entropy
from novainstrumentation import smooth
from FeaturesANDClustering.MF import MF_calculus, PowerSpectrum, SumPowerSpectrum
from FeaturesANDClustering.PeakFinder import detect_peaks
from FeaturesANDClustering.SpikesMethods import Spikes
from nolds import dfa, corr_dim, hurst_rs


#WindMethod
def WindowStat(inputSignal, statTool, fs, window_len=50, window='hanning'):

	output = np.zeros(len(inputSignal))
	win = eval('np.' + window + '(window_len)')

	if inputSignal.ndim != 1:
		raise ValueError("smooth only accepts 1 dimension arrays.")
	if inputSignal.size < window_len:
		raise ValueError("Input vector needs to be bigger than window size.")
	if window_len < 3:
		return inputSignal

	inputSignal = inputSignal - np.mean(inputSignal)

	WinRange = int(window_len/2)

	sig = np.r_[inputSignal[WinRange:0:-1], inputSignal, inputSignal[-1:len(inputSignal)-WinRange:-1]]

	# windowing
	if(statTool is 'stn'):
		WinSize = window_len
		numSeg = int(len(inputSignal) / WinSize)
		SigTemp = np.zeros(numSeg)
		for i in range(1, numSeg):
			signal = inputSignal[(i - 1) * WinSize:i * WinSize]
			SigTemp[i] = signaltonoise(signal)
		output = np.interp(np.linspace(0, len(SigTemp), len(output)), np.linspace(0, len(SigTemp), len(SigTemp)), SigTemp)
	elif(statTool is 'zcr'):
		# inputSignal = inputSignal - smooth(inputSignal, window_len=fs*4)
		# inputSignal = inputSignal - smooth(inputSignal, window_len=int(fs/10))
		# sig = np.r_[inputSignal[WinRange:0:-1], inputSignal, inputSignal[-1:len(inputSignal) - WinRange:-1]]

		for i in range(int(WinRange), len(sig) - int(WinRange)):
			output[i - int(WinRange)] = ZeroCrossingRate(sig[i - WinRange:WinRange + i]*win)
		output = smooth(output, window_len=1024)
	elif(statTool is 'std'):
		for i in range(int(WinRange), len(sig) - int(WinRange)):
			output[i - WinRange] = np.std(sig[i - WinRange:WinRange + i]*win)
	elif(statTool is 'subPks'):
		for i in range(int(WinRange), len(sig) - int(WinRange)):
			pks = [0]
			win_len = window_len
			while(len(pks) < 10):
				pks = detect_peaks(sig[i - int(win_len / 2):int(win_len / 2) + i], valley=False, mph=np.std(sig[i - int(win_len / 2):int(win_len / 2)+ i]))
				if(len(pks) < 10):
					win_len += int(win_len/5)
			sub_zero = pks[1] - pks[0]
			sub_end = pks[-1] - pks[-2]
			subPks = np.r_[sub_zero, (pks[1:-1] - pks[0:-2]), sub_end]
			win = eval('np.' + window + '(len(subPks))')
			output[i - int(WinRange)] = np.mean(subPks*win)
	elif (statTool is 'findPks'):
		for i in range(int(WinRange), len(sig) - int(WinRange)):
			pks = detect_peaks(sig[i - WinRange:WinRange + i], valley=False,
								   mph=np.std(sig[i - WinRange:WinRange + i]))
			LenPks = len(pks)
			output[i - int(WinRange)] = LenPks
	elif(statTool is 'sum'):
		for i in range(int(WinRange), len(sig) - int(WinRange)):
			output[i - WinRange] = np.sum(abs(sig[i - WinRange:WinRange + i] * win))
	elif(statTool is 'AmpDiff'):
		for i in range(int(WinRange), len(sig) - int(WinRange)):
			win_len = window_len
			tempSig = sig[i - int(win_len / 2):int(win_len / 2) + i]
			maxPks = detect_peaks(tempSig, valley=False,
								   mph=np.std(tempSig))
			minPks = detect_peaks(tempSig, valley=True,
								   mph=np.std(tempSig))
			AmpDiff = np.sum(tempSig[maxPks]) - np.sum(tempSig[minPks])
			output[i - WinRange] = AmpDiff
	elif(statTool is 'MF'):
		for i in range(int(WinRange), len(sig) - int(WinRange)):
			f, Pxx = PowerSpectrum(inputSignal[i - WinRange:i + WinRange], fs=fs, nperseg=WinRange/2)
			mf = MF_calculus(Pxx)
			output[i - WinRange] = mf
	elif(statTool is "SumPS"):
		for i in range(int(WinRange), len(sig) - int(WinRange)):
			f, Pxx = PowerSpectrum(inputSignal[i - WinRange:i + WinRange], fs=fs, nperseg=WinRange / 2)
			sps = SumPowerSpectrum(Pxx)
			output[i - WinRange] = sps
	elif(statTool is 'fractal'):
		for i in range(int(WinRange), len(sig) - int(WinRange)):
			output[i - WinRange] = entropy(sig[i - WinRange:WinRange + i]*win)
			output[np.where(output is "nan" or output > 1E308)[0]] = 0
	elif(statTool is "AmpMean"):
		for i in range(int(WinRange), len(sig) - int(WinRange)):
			output[i - WinRange] = np.mean(abs(sig[i - WinRange:WinRange + i]) * win)
	elif(statTool is"Spikes1"):
		ss = 0.1*max(sig)
		for i in range(int(WinRange), len(sig) - int(WinRange)):
			pkd, md = Spikes(sig[i - WinRange:WinRange + i] * win, mph=ss)
			output[i - WinRange] = pkd
	elif (statTool is "Spikes2"):
		ss = 0.1 * max(sig)
		for i in range(int(WinRange), len(sig) - int(WinRange)):
			pkd, md = Spikes(sig[i - WinRange:WinRange + i] * win, mph=ss)
			output[i - WinRange] = md
	elif (statTool is "Spikes3"):
		ss = 0.1 * max(sig)
		for i in range(int(WinRange), len(sig) - int(WinRange)):
			pkd, md = Spikes(abs(sig[i - WinRange:WinRange + i] )* win, mph=ss)
			output[i - WinRange] = md

	output = output - np.mean(output)
	output = output/max(output)
	#output = smooth(output, window_len=10)

	return output

def ZeroCrossingRate(signal):
	signal = signal - np.mean(signal)
	ZCVector = np.where(np.diff(np.sign(signal)))[0]

	return len(ZCVector)

def findPeakDistance(signal, mph, threshold):
	pks = detect_peaks(signal, mph = mph, show = False)
	vpks = detect_peaks(signal, mph= mph, valley=True)

	if(len(vpks)> len(pks)):
		pks = vpks

	signaldPks = np.zeros(np.size(signal))
	dpks = np.log10(abs(np.diff(pks) - np.mean(np.diff(pks))) + 1)

	for i in range(0, len(dpks)):
		if(i == 0):
			signaldPks[0:pks[i]] = dpks[i]
			signaldPks[pks[i]:pks[i + 1]] = dpks[i]
		elif(i == len(dpks)-1):
			signaldPks[pks[i]:pks[i+1]] = dpks[-1]
		else:
			signaldPks[pks[i]:pks[i+1]] = dpks[i]


	return signaldPks

