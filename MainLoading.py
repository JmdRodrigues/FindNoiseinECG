import os
import numpy as np
import h5py
import ast
import seaborn
import matplotlib.pyplot as plt
from scipy.io import loadmat
from MainProcessing import ReportClustering
from STDVariation.ComparingAlgorithms import ReportSTD



def openMat(filename, filedir):

	#open mat should have acces to info file where fs and channel is identified
	for file in os.listdir(filedir):

		print(file)
		if(file == "Attributes.txt"):
			inf = open(filedir + "/" + file, 'r')
			AttDic = ast.literal_eval(inf.read())
			inf.close()

			fs = AttDic["fs"]
			index = AttDic["index"]
			n_clusters = AttDic["clusters"]
			mat = loadmat(filename)

			ECG_data = mat["val"][index][:]

		elif (file == "Noise2790.txt"):
			Noise = np.loadtxt(filedir + "/" + file)

	return ECG_data, fs, Noise, n_clusters



def openH5(filename, filedir):
	print(filename)

	f = h5py.File(filename, 'r')

	ECG_Macs = [key for key in f.keys()][0]

	ECG_data_group = f[ECG_Macs + "/raw"]

	fs = f[ECG_Macs].attrs["sampling rate"] * 1.0

	for file in os.listdir(filedir):
		if (file == "Noise2790.txt"):
			Noise = np.loadtxt(filedir + "/" + file)
		elif(file == "Attributes.txt"):
			inf = open(filedir + "/" + file, 'r')
			AttDic = ast.literal_eval(inf.read())
			inf.close()
			fs = AttDic["fs"]
			n_clusters = AttDic["clusters"]

	if ("JakeHeart" in filename or "Lucas" in filename or "Tiago" in filename):
		ECG_data = ECG_data_group["channel_1"][:120000, 0]
	else:
		ECG_data = ECG_data_group["channel_1"][:, 0]


	return ECG_data, fs, Noise, n_clusters

def opentxt(filename, filedir):

	if("SN" in filename):
		for file in os.listdir(filedir):
			if(file == "Attributes.txt"):
				inf = open(filedir + "/" + file, 'r')
				AttDic = ast.literal_eval(inf.read())
				inf.close()

				fs = AttDic["fs"]
				n_clusters = AttDic["clusters"]

				ECG_data = np.loadtxt(filename)

			elif(file == "Noise2790.txt"):
				Noise = np.loadtxt(filedir + "/" + file)

		return ECG_data, fs, Noise, n_clusters

	else:
		HeadDic = read_header(filename)
		fs = HeadDic["sampling rate"]
		channel = HeadDic["channels"][HeadDic["sensor"] == "ECG"] + 1

		ECG_data = np.loadtxt(filename)
		ECG_data = ECG_data[:, channel]

		for file in os.listdir(filedir):
			if(file == "Noise2790.txt"):
				Noise = np.loadtxt(filedir + "/" + file)
			elif(file == "Attributes.txt"):
				inf = open(filedir + "/" + file, 'r')
				AttDic = ast.literal_eval(inf.read())
				inf.close()

				fs = AttDic["fs"]
				n_clusters = AttDic["clusters"]

		return ECG_data, fs, Noise, n_clusters

def read_header(source_file, print_header = False):

	f = open(source_file, 'r')

	f_head = [f.readline() for i in range(3)]

	#convert to diccionary (considering 1 device only)
	head_dic = ast.literal_eval(f_head[1][24:-2])

	return head_dic

MainDir = os.path.realpath("Signals2")
Folders = os.listdir(MainDir)

for eachFolder in Folders:
	#if("Synthetic" in eachFolder):
	#print(eachFolder)
	folderPath = MainDir + "/" + eachFolder
	Files = os.listdir(folderPath)

	for eachFile in Files:
		#print(eachFile[-4:])
		print(eachFile)
		if "." not in str(eachFile):
			print("Could not open because it is not a file...")
		elif ("Attributes" in eachFile or "Noise" in eachFile):
			print("not opening " + eachFile)
		else:
			FileDir = os.path.realpath(folderPath + "/" + eachFile)
			if(str(eachFile[-4:]) == ".mat"):
				data, fs, Noise, n_clusters = openMat(FileDir, folderPath)
			elif(eachFile[-4:] == ".txt"):
				data, fs, Noise, n_clusters = opentxt(FileDir, folderPath)
			elif(eachFile[-3:] == ".h5"):
				data, fs, Noise, n_clusters = openH5(FileDir, folderPath)
			else:
				print("file in incorrect format. Tolerate formats are: .txt, .h5, .mat")

			time = np.linspace(0, len(data)/fs, len(data))
			# plt.plot(time, data/max(data))
			# plt.plot(time, Noise, 'r-o')
			# plt.xticks(np.linspace(0, len(data)/fs, 4*len(data)/fs))
			# plt.show()

			savepath = folderPath
			if(len(np.shape(Noise)) > 1):
				for i in range(0, np.shape(Noise)[1]):
					ReportClustering(data, Noise[:, i], fs, time, eachFile[:-4]+str(i), folderPath, clusters=n_clusters)
					# ReportSTD(data, Noise[:, i], fs, time, savepath, eachFile[:-4]+str(i))
			else:
				# plt.plot(data)
				# plt.show()
				ReportClustering(data, Noise, fs, time, eachFile[:-4], folderPath, clusters=n_clusters)
				# ReportSTD(data, Noise, fs, time, savepath, eachFile[:-4])



