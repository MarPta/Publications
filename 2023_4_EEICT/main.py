import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

path = os.path.dirname(__file__)

simSetup = {
    "nSamplesMC": 100,
    "nSimulations": 1,
    "meanGP": {
        "name": "constant",
        "value": 0
    },
    "covGP": {
        "name": "squared exponential",
        "var": 4,                           # GP single variable variance
        "var_x": 2                          # GP spatial scale variance
    },
    "space": {
        "name": "flat rect",
        "nDimensions": 2,
        "size": (10, 5)
    },
    "nTrainPos": 20,
    "regCoef": 0.001,                   # Regularization coefficient of GP covariance matrix to ensure positive definiteness
    "obsVar": 0.01,                     # GP observation noise variance
    "time": datetime.datetime.now().strftime("%Y/%m/%d, %H:%M:%S")
}

def meanFuncGP(setup, x_a):
    """GP mean function m"""

    if setup["meanGP"]["name"] == "constant" and "value" in setup["meanGP"]:
        return setup["meanGP"]["value"]

    raise Exception("invalid GP mean function setup")


def covarianceFuncGP(setup, x_a, x_b):
    """GP covariance function "k" of 2 position vectors"""

    if setup["covGP"]["name"] == "squared exponential":
        return squaredExpCovFunc(x_a, x_b, setup["covGP"]["sigma_x"], setup["covGP"]["sigma_x"])
    
    raise Exception("invalid GP covariance function setup")

################################################################################

def squaredExpCovFunc(x_a, x_b, var, var_x):
    """Squared exponential covariance function"""

    if x_a.size != x_b.size or var <= 0 or var_x <= 0:
        raise Exception("invalid squared exponential function parameters")

    return var * np.exp(-((np.linalg.norm(x_a - x_b)**2)/(2*var_x)))

################################################################################

class GaussianProcess:
    def __init__(self):
        self.meanType = "undefined"
        self.covType = "undefined"

    def setMeanFuncConst(self, value):
        self.meanType = "constant"
        self.meanValue = value

    def setCovFuncSquaredExp(self, var, var_x):
        self.covType = "squared exponential"
        self.var = var
        self.var_x = var_x

    def meanFunc(self, x_a):
        if(self.meanType == "constant"):
            return self.meanValue

    def covFunc(self, x_a, x_b):
        if(self.covType == "squared exponential"):
            return squaredExpCovFunc(x_a, x_b, self.var, self.var_x)
        

################################################################################

def gpr(gp, trainPos, testPos, trainObs, obsVar):
    """GPR standard evaluation"""

    nTrainPos = trainPos.shape[0]
    if(nTrainPos != trainObs.size):
        raise Exception("different number of training positions and function observations")

    meanVec_m = np.zeros(nTrainPos)
    for i in range(nTrainPos):
        meanVec_m[i] = gp.meanFunc(trainPos[i, :])

    covMat_K = np.zeros((nTrainPos, nTrainPos))
    for row in range(nTrainPos):
        for col in range(nTrainPos):
            covMat_K[row, col] = gp.covFunc(trainPos[row, :], trainPos[col, :])

    covMat_Q = covMat_K + obsVar*np.identity(nTrainPos)

    crossCovVec_c = np.zeros(nTrainPos)
    for i in range(nTrainPos):
        crossCovVec_c[i] = gp.covFunc(testPos, trainPos[i, :])

    covMatInv_Q = np.linalg.inv(covMat_Q)
    c_t_dot_Q_inv = np.dot(crossCovVec_c, covMatInv_Q)

    postMean = gp.meanFunc(testPos) + np.dot(c_t_dot_Q_inv, (trainObs - meanVec_m))
    postVar = gp.covFunc(testPos, testPos) - np.dot(c_t_dot_Q_inv, crossCovVec_c)

    return (postMean, postVar)

################################################################################

trainPos = np.array([[0, 0], [1, 1], [2, 2]])
trainObs = np.array([1, 2, 3])
testPos = np.array([3, 3])
obsVar = simSetup["obsVar"]

trueGP = GaussianProcess()
trueGP.setMeanFuncConst(simSetup["meanGP"]["value"])
trueGP.setCovFuncSquaredExp(simSetup["covGP"]["var"], simSetup["covGP"]["var_x"])

posteriorParam = gpr(trueGP, trainPos, testPos, trainObs, obsVar)
print(posteriorParam)
