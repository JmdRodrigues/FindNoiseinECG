import os
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas
import itertools
import matplotlib.patheffects as pte
import matplotlib.gridspec as grid
from scipy.stats import signaltonoise, entropy
from scipy.signal import decimate
from matplotlib.font_manager import FontProperties
from scipy.spatial import distance
from novainstrumentation import smooth
from FeaturesANDClustering.RemoveUnwantedPoints import RemoveUglyDots
from FeaturesANDClustering.WindowFeature import WindowStat, findPeakDistance
from FeaturesANDClustering.FrequencyFeature import SpectralComponents
from FeaturesANDClustering.MultiDClustering import MultiDimensionalClusteringKmeans, MultiDimensionalClusteringAGG
from PerformanceMetric.SensEspAcc import GetResults
from GenerateThings.PlotSaver import plotClusters, plotClustersBW, plotDistanceMetric, plotLinearData
from GenerateThings.PDFATextGen import get_file_name, pdf_report_creator, pdf_text_closer
from GenerateThings.TextSaver import SaveReport


'''
------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------MAIN script------------------------------------------------------
In this script, the main code is developed. This script will open a signal and try
to extract features from it.
The feature extraction process regards applying very simple operations to the
signal in order to find patterns on the signal. The main operators are moving
windows with operations like sum, standard deviation, amplitude variation, etc...
another operator applied to the signal is the frequency spectrum.
After the feature extraction, the feature matrix is created and a clustering
algorithm can be executed in order to classify the signal in its different parts,
namely noisy pattern or non-noisy pattern.
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
'''
#
def ReportClustering(Signal, NoiseSignal, fs, time, fileName, fileDir, clusters):


	#MainDir = "IWANTOFINDNOISE/TestSignals"
	# SignalKind = os.listdir(MainDir)
	#
	# for SignalFolder in SignalKind:
	#
	# 	SignalDir = MainDir + '/' + SignalFolder
	# 	SignalData = os.listdir(SignalDir)
	#
	# 	for signal in SignalData:
	#
	# 		if ".txt" in signal:
	#

	#----------------------------------------------------------------------------------------------------------------------
	#                            Open Signal (and Pre-Process Data ???)
	#----------------------------------------------------------------------------------------------------------------------
#When I will need to take a signal:
# fileName = "ecg1"
# fileDir = "Signals2/ECG1/"
# Signal = np.loadtxt(fileDir + fileName + ".txt")
# NoiseSignal = np.loadtxt(fileDir + "Noise2790.txt")
# fs = 100
# time = np.linspace(0, len(Signal)/fs, len(Signal))
# clusters = 4
	print("Loading file " + fileName)
	#fileTest = "TestSignals/TestFiles/ecg1.txt"

	if(os.path.isdir(fileDir + "/" + "Report_Color_" + fileName) is False):
		os.makedirs(fileDir + "/" + "Report_Color_" + fileName)

	print("Creating Report File")
	pp, ReportTxt = pdf_report_creator(fileDir, "Report_Color_" + fileName)
	win = 512

	# for win in windows:
	#NoiseTest = "TestSignals/TestFiles/Noise2790.txt"
	# open signal
	#signal = np.loadtxt(file)
	signal = Signal
	signalNoise = NoiseSignal
	osignal = Signal
	signal = signal - np.mean(signal)
	signal = signal/max(signal)

	#----------------------------------------------------------------------------------------------------------------------
	#                                        Extract Features
	#----------------------------------------------------------------------------------------------------------------------

	print("Extracting features...")

	#1 - Std Window
	signalSTD = WindowStat(signal, fs=fs, statTool='std', window_len=(win*fs)/256)

	print("...feature 1 - STD")

	#2 - ZCR
	signalZCR64 = WindowStat(signal, fs=fs, statTool='zcr', window_len=(win*fs)/512)

	print("...feature 2 - ZCR")
	#3 - Sum
	signalSum64 = WindowStat(signal, fs=fs, statTool='sum', window_len=(win*fs)/256)
	signalSum128 = WindowStat(signal, fs=fs, statTool='sum', window_len=(win*fs)/100)

	print("...feature 3 - Sum")
	#4 - Number of Peaks above STD
	signalPKS = WindowStat(signal, fs=fs, statTool='findPks', window_len=(win*fs)/128)
	signalPKS2 = WindowStat(signal, fs=fs, statTool='findPks', window_len=(64 * fs) / 100)

	print("...feature 4 - Pks")
	#5 - Amplitude Difference between successive PKS
	signalADF32 = WindowStat(signal, fs=fs, statTool='AmpDiff', window_len=(win*fs)/128)
	# signalADF128 = WindowStat(signal, fs=fs, statTool='AmpDiff', window_len=(2*win*fs)/100)

	print("...feature 5 - AmpDif")
	#6 - Medium Frequency feature
	signalMF = WindowStat(signal, fs=fs, statTool='MF', window_len=(32*fs)/128)
	print("...feature 6 - MF")

	#7 - Frequency Spectrum over time
	Pxx, freqs, bins, im = SpectralComponents(osignal, fs, NFFT=129, show=False) #miss axes

	#Interp - to have the same number of points as the original signal
	signalPxx = np.interp(np.linspace(0, len(Pxx), len(signal)), np.linspace(0, len(Pxx), len(Pxx)), Pxx)
	signalPxx -= np.mean(signalPxx)
	#8 - Find Peak Distance
	dpks = findPeakDistance(signal, 2*np.std(signal), 0)

	#9 - Smooth signal
	smSignal = smooth(abs(signal), window_len=int(fs/2))
	smSignal = smSignal/max(smSignal)

	smSignal2 = smooth(abs(signal), window_len=int(fs))
	smSignal2 = smSignal2/ max(smSignal2)
	#
	# #8 - fractal analysis
	# SignalFract = WindowStat(signal, fs=fs, statTool='fractal', window_len=10)
	# print(SignalFract)

	print("Done with Features extraction...creating Matrix")
	# Create Matrix of Features------Standard features
	FeatureNames = ["Standard Deviation", 'Sum', "ZCR"]
	FeatureMatrix = np.array(
		[signalSTD, signalSum64, signalZCR64]).transpose()

	plotLinearData(time, FeatureMatrix, signal, FeatureNames, pp)

	print("Starting Clustering")

	X, y_pred, XPCA, params = MultiDimensionalClusteringAGG(FeatureMatrix, n_clusters=clusters, Linkage='ward',
	                                                        Affinity='euclidean')

	# ----------------------------------------------------------------------------------------------------------------------
	#                  Create Classification Array (1- Noise) (0 - Non-Noise)
	# ----------------------------------------------------------------------------------------------------------------------
	# find signal indexes - in this case i will assume that the signal is the majority of the signal
	print("Creating Predicted Array...")
	Indexiser = []

	for i in range(0, clusters):
		s = len(y_pred[np.where(y_pred == i)[0]].tolist())
		#s = np.std(signal[np.where(y_pred == i)[0]])
		Indexiser.append(s)

	SigIndex = Indexiser.index(max(Indexiser))

	Prediction = np.ones(np.size(y_pred.tolist()))
	Prediction[np.where(y_pred == SigIndex)[0]] = 0

	print("Calculating Confusion matrix...")
	ResultEvents, Sens, Spec, CorrectPeriod, ErrorPeriod, RatioCorrectWrong, CorrectArray, ErrorArray, TP, TN, FP, FN = GetResults(
		signalNoise, Prediction)
	sizes = [TP, TN, FP, FN]

	# ----------------------------------------------------------------------------------------------------------------------
	#                                              Plot Things
	# ----------------------------------------------------------------------------------------------------------------------
	Clusterfig = plotClusters(y_pred, signal, time, XPCA, clusters, pp)
	ClassFig = plotDistanceMetric(time, signal, signalNoise, Prediction, Spec, Sens, CorrectPeriod, sizes, pp)


	FeatureNames = ["Standard Deviation" ,"ZCR", "ADF128", "PKS"]
	FeatureMatrix = np.array(
		[signalSTD, signalZCR64, signalADF32, signalPKS]).transpose()
	clusters = 5
	plotLinearData(time, FeatureMatrix, signal, FeatureNames, pp)

	print("Starting Clustering")

	X, y_pred, XPCA, params = MultiDimensionalClusteringAGG(FeatureMatrix, n_clusters=clusters, Linkage='ward',
	                                                        Affinity='euclidean')

	print("Plotting...")
	Clusterfig = plotClusters(y_pred, signal, time, XPCA, clusters, pp)



	# ----------------------------------------------------------------------------------------------------------------------
	#                  Create Classification Array (1- Noise) (0 - Non-Noise)
	# ----------------------------------------------------------------------------------------------------------------------
	# find signal indexes - in this case i will assume that the signal is the majority of the signal
	print("Creating Predicted Array...")
	Indexiser = []

	for i in range(0, clusters):
		s = len(y_pred[np.where(y_pred == i)[0]].tolist())
		# s = np.std(signal[np.where(y_pred == i)[0]])
		Indexiser.append(s)

	SigIndex = Indexiser.index(max(Indexiser))

	Prediction = np.ones(np.size(y_pred.tolist()))
	Prediction[np.where(y_pred == SigIndex)[0]] = 0

	print("Calculating Confusion matrix...")
	ResultEvents, Sens, Spec, CorrectPeriod, ErrorPeriod, RatioCorrectWrong, CorrectArray, ErrorArray, TP, TN, FP, FN = GetResults(
		signalNoise, Prediction)
	sizes = [TP, TN, FP, FN]

	# ----------------------------------------------------------------------------------------------------------------------
	#                                              Plot Things
	# ----------------------------------------------------------------------------------------------------------------------

	ClassFig = plotDistanceMetric(time, signal, signalNoise, Prediction, Spec, Sens, CorrectPeriod, sizes, pp)

	#Round4

	clusters=5

	FeatureNamesG = ["Standard Deviation", "std", "ADF128"]
	FeatureMatrixG = np.array(
		[signalSTD, signalSTD, signalADF32]).transpose()

	plotLinearData(time, FeatureMatrixG, signal, FeatureNamesG, pp)
	# ----------------------------------------------------------------------------------------------------------------------
	#                                Execute Clustering Techniques
	# X, y_pred, XPCA, params, Z, xx, yy = MultiDimensionalClusteringKmeans(FeatureMatrix, time, signal, show=False, n_clusters=n_clusters)
	print("Starting Clustering")

	X, y_pred, XPCA, params = MultiDimensionalClusteringAGG(FeatureMatrixG, n_clusters=clusters, Linkage='ward',
	                                                        Affinity='euclidean')
	# X, y_pred, XPCA, params = MultiDimensionalClusteringKmeans(FeatureMatrix, n_clusters=clusters)
	# ----------------------------------------------------------------------------------------------------------------------
	#                  Create Classification Array (1- Noise) (0 - Non-Noise)
	# ----------------------------------------------------------------------------------------------------------------------
	# find signal indexes - in this case i will assume that the signal is the majority of the signal
	print("Creating Predicted Array...")
	Indexiser = []

	for i in range(0, clusters):
		s = len(y_pred[np.where(y_pred == i)[0]].tolist())
		#s = np.std(signal[np.where(y_pred == i)[0]])
		Indexiser.append(s)

	SigIndex = Indexiser.index(max(Indexiser))

	Prediction = np.ones(np.size(y_pred.tolist()))
	Prediction[np.where(y_pred == SigIndex)[0]] = 0

	print("Calculating Confusion matrix...")
	ResultEvents, Sens, Spec, CorrectPeriod, ErrorPeriod, RatioCorrectWrong, CorrectArray, ErrorArray, TP, TN, FP, FN = GetResults(
		signalNoise, Prediction)
	sizes = [TP, TN, FP, FN]

	# ----------------------------------------------------------------------------------------------------------------------
	#                                              Plot Things
	# ----------------------------------------------------------------------------------------------------------------------
	print("Plotting...")
	Clusterfig = plotClustersBW(y_pred, signal, time, XPCA, clusters, pp)
	ClassFig = plotDistanceMetric(time, signal, signalNoise, Prediction, Spec, Sens, CorrectPeriod, sizes, pp)

	clusters = 5

	FeatureNamesG = ["Standard Deviation", "std", "ZCR"]
	FeatureMatrixG = np.array(
		[signalSTD, signalSTD, signalZCR64]).transpose()

	plotLinearData(time, FeatureMatrixG, signal, FeatureNamesG, pp)
	# ----------------------------------------------------------------------------------------------------------------------
	#                                Execute Clustering Techniques
	# X, y_pred, XPCA, params, Z, xx, yy = MultiDimensionalClusteringKmeans(FeatureMatrix, time, signal, show=False, n_clusters=n_clusters)
	print("Starting Clustering")

	X, y_pred, XPCA, params = MultiDimensionalClusteringAGG(FeatureMatrixG, n_clusters=clusters, Linkage='ward',
	                                                        Affinity='euclidean')
	# X, y_pred, XPCA, params = MultiDimensionalClusteringKmeans(FeatureMatrix, n_clusters=clusters)
	# ----------------------------------------------------------------------------------------------------------------------
	#                  Create Classification Array (1- Noise) (0 - Non-Noise)
	# ----------------------------------------------------------------------------------------------------------------------
	# find signal indexes - in this case i will assume that the signal is the majority of the signal
	print("Creating Predicted Array...")
	Indexiser = []

	for i in range(0, clusters):
		s = len(y_pred[np.where(y_pred == i)[0]].tolist())
		# s = np.std(signal[np.where(y_pred == i)[0]])
		Indexiser.append(s)

	SigIndex = Indexiser.index(max(Indexiser))

	Prediction = np.ones(np.size(y_pred.tolist()))
	Prediction[np.where(y_pred == SigIndex)[0]] = 0

	print("Calculating Confusion matrix...")
	ResultEvents, Sens, Spec, CorrectPeriod, ErrorPeriod, RatioCorrectWrong, CorrectArray, ErrorArray, TP, TN, FP, FN = GetResults(
		signalNoise, Prediction)
	sizes = [TP, TN, FP, FN]

	# ----------------------------------------------------------------------------------------------------------------------
	#                                              Plot Things
	# ----------------------------------------------------------------------------------------------------------------------
	print("Plotting...")
	Clusterfig = plotClusters(y_pred, signal, time, XPCA, clusters, pp)
	ClassFig = plotDistanceMetric(time, signal, signalNoise, Prediction, Spec, Sens, CorrectPeriod, sizes, pp)
#----------------------------------------------------------------------------------------------------------------------
#                     Save plots in PDF and relevant values on text file
#----------------------------------------------------------------------------------------------------------------------
	SaveReport(fileName, sizes, Sens, Spec, CorrectPeriod, ReportTxt)
	pdf_text_closer(ReportTxt, pp)