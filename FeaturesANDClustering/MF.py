import numpy as np
import scipy.signal as sc

def MF_calculus(Pxx):
    sumPxx = np.sum(Pxx)
    mf = 0
    for i in range(0, len(Pxx)):
        if(np.sum(Pxx[0:i]) < sumPxx/2.0):
            continue
        else:
            mf = i
            break

    return mf

def SumPowerSpectrum(Pxx):
    return np.sum(Pxx)

def PowerSpectrum(data, fs, nperseg):
    f, Pxx = sc.periodogram(data, fs=fs)

    return f, Pxx


