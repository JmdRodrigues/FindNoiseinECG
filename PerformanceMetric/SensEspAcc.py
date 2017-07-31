import numpy as np

def CalculateSens(TruePositives, FalseNegatives):

    if(TruePositives == 0):
        Sens = 0
    else:
        Sens = TruePositives/(TruePositives + FalseNegatives)

    return Sens

def CalculateSpec(TrueNegatives, FalsePositives):

    if(TrueNegatives == 0):
        Spec = 0
    else:
        Spec = TrueNegatives/ (TrueNegatives + FalsePositives)

    return Spec

def GetResults(TrueEvents, PredictedEvents):

    ResultEvents = TrueEvents + 2*PredictedEvents

    CorrectArray = np.zeros(np.size(PredictedEvents))
    CorrectArray[np.where(ResultEvents == 3)[0]] = 1
    CorrectArray[np.where(ResultEvents == 0)[0]] = 1
    ErrorArray = np.zeros(np.size(PredictedEvents))
    ErrorArray[np.where(ResultEvents == 2)[0]] = 1
    ErrorArray[np.where(ResultEvents == 1)[0]] = 1

    TP = len(ResultEvents[np.where(ResultEvents == 3)])
    TN = len(ResultEvents[np.where(ResultEvents == 0)])
    FP = len(ResultEvents[np.where(ResultEvents == 2)])
    FN = len(ResultEvents[np.where(ResultEvents == 1)])

    Sens = CalculateSens(TP, FN)
    Spec = CalculateSpec(TN, FP)

    CorrectPeriod = (TP + TN) / len(ResultEvents)
    ErrorPeriod = (FP + FN) / len(ResultEvents)
    RatioCorrectWrong = (TP + TN) / (FP + FN)


    return ResultEvents, Sens, Spec, CorrectPeriod, ErrorPeriod, RatioCorrectWrong, CorrectArray, ErrorArray, TP, TN, FP, FN

