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

def meanVecGP(gp, pos):
    """Get GP mean values at given positions"""

    nPos = pos.shape[0]
    meanVec = np.zeros(nPos)
    for i in range(nPos):
        meanVec[i] = gp.meanFunc(pos[i, :])

    return meanVec

def covMatGP(gp, pos):
    """Get GP covariance matrix at given positions"""

    nPos = pos.shape[0]
    covMat = np.zeros((nPos, nPos))
    for row in range(nPos):
        for col in range(nPos):
            covMat[row, col] = gp.covFunc(pos[row, :], pos[col, :])

    return covMat

def gpr(gp, trainPos, testPos, trainObs, obsVar):
    """GPR standard evaluation"""

    nTrainPos = trainPos.shape[0]
    if(nTrainPos != trainObs.size):
        raise Exception("different number of training positions and function observations")

    meanVec_m = meanVecGP(gp, trainPos)

    covMat_K = covMatGP(gp, trainPos)
    covMat_Q = covMat_K + obsVar*np.identity(nTrainPos)

    crossCovVec_c = np.zeros(nTrainPos)
    for i in range(nTrainPos):
        crossCovVec_c[i] = gp.covFunc(testPos, trainPos[i, :])

    covMatInv_Q = np.linalg.inv(covMat_Q)
    c_t_dot_Q_inv = np.dot(crossCovVec_c, covMatInv_Q)

    postMean = gp.meanFunc(testPos) + np.dot(c_t_dot_Q_inv, (trainObs - meanVec_m))
    postVar = gp.covFunc(testPos, testPos) - np.dot(c_t_dot_Q_inv, crossCovVec_c)

    return (postMean, postVar)

def realizeGP(gp, pos, regCoef):
    """Generate realization vector of given GP on given positions"""

    nPos = pos.shape[0]
    meanVec = meanVecGP(gp, pos)
    covMat = covMatGP(gp, pos) + regCoef*np.identity(nPos)      # Regularization ensuring positive definiteness
    choleskyCovMat = np.linalg.cholesky(covMat)
    iidGaussVec = np.random.normal(0, 1, nPos)    # Vector of iid Gaussian zero mean unit st. d. RVs

    realizationVec = meanVec + np.dot(choleskyCovMat, iidGaussVec)

    return realizationVec

################################################################################

trainPos = np.array([[0, 0], [1, 1], [2, 2]])
testPos = np.array([3, 3])
allPos = np.append(trainPos, [testPos], axis=0)

trueGP = GaussianProcess()
trueGP.setMeanFuncConst(simSetup["meanGP"]["value"])
trueGP.setCovFuncSquaredExp(simSetup["covGP"]["var"], simSetup["covGP"]["var_x"])

allRealizations = realizeGP(trueGP, allPos, simSetup["regCoef"])
trainRealizations = allRealizations[0:trainPos.shape[0]]
trainObs = trainRealizations + np.random.normal(0, np.sqrt(simSetup["obsVar"]), trainPos.shape[0])

posteriorParam = gpr(trueGP, trainPos, testPos, trainObs, simSetup["obsVar"])
print(posteriorParam)