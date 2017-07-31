import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from novainstrumentation import smooth

def SpectralComponents(InputSignal, SamplF, NFFT = 1024, ax=None, show=False):

	if(show):

		plt.figure()
		Pxx, freqs, bins, im = plt.specgram(InputSignal, Fs=SamplF, NFFT=NFFT, cmap=plt.cm.gist_rainbow)

		# ax[0].set_xlabel("Time (s)")
		# ax[0].set_ylabel("Amplitude (r.u.)")

		ax.set_title("Feature 6 - Frequency with time")
		Pxx = np.mean(Pxx, axis=0)
		Pxx= abs(Pxx) - np.mean(abs(Pxx))
		Pxx /= max(Pxx)
		#Pxx = smooth(Pxx, WinRange=200)
		ax.plot(np.linspace(0, len(Pxx) / SamplF, len(InputSignal)), InputSignal/ max(InputSignal), alpha=0.4)
		ax.plot(np.linspace(0, len(Pxx) / SamplF, len(Pxx)), Pxx, 'r')
		ax.set_xlabel("Time (s)")
		ax.set_ylabel("Amplitude (r.u.)")

		return Pxx, freqs, bins, im

	else:

		Pxx, freqs, bins, im = plt.specgram(InputSignal, Fs=SamplF, NFFT=NFFT, cmap=plt.cm.gist_rainbow)

		Pxx = np.mean(Pxx, axis=0)

		return Pxx, freqs, bins, im