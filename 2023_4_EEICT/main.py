import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import itertools

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
    "spaceSize": (10, 5),
    "nTrainPos": 20,
    "testPosRes": 2,            # Positions per spatial unit
    "regCoef": 0.001,           # Regularization coefficient of GP covariance matrix to ensure positive definiteness
    "obsVar": 0.01,             # GP observation noise variance
    "time": datetime.datetime.now().strftime("%Y/%m/%d, %H:%M:%S")
}

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
        if self.meanType == "constant":
            return self.meanValue

    def covFunc(self, x_a, x_b):
        if self.covType == "squared exponential":
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
    if nTrainPos != trainObs.size:
        raise Exception("different number of training positions and function observations")

    meanVec_m = meanVecGP(gp, trainPos)

    covMat_K = covMatGP(gp, trainPos)
    covMat_Q = covMat_K + obsVar*np.identity(nTrainPos)
    covMatInv_Q = np.linalg.inv(covMat_Q)

    nTestPos = testPos.shape[0]
    if nTestPos < 1:
        raise Exception("at least 1 test position must be provided")
    postDistMat = np.zeros((nTestPos, 2))

    for testPosID in range(nTestPos):
        crossCovVec_c = np.zeros(nTrainPos)
        for trainPosID in range(nTrainPos):
            crossCovVec_c[trainPosID] = gp.covFunc(testPos[testPosID, :], trainPos[trainPosID, :])

        c_t_dot_Q_inv = np.dot(crossCovVec_c, covMatInv_Q)

        postMean = gp.meanFunc(testPos) + np.dot(c_t_dot_Q_inv, (trainObs - meanVec_m))
        postVar = gp.covFunc(testPos, testPos) - np.dot(c_t_dot_Q_inv, crossCovVec_c)
        postDistMat[testPosID, :] = np.array([postMean, postVar])

    return postDistMat

def realizeGP(gp, pos, regCoef):
    """Generate realization vector of given GP on given positions"""

    nPos = pos.shape[0]
    meanVec = meanVecGP(gp, pos)
    covMat = covMatGP(gp, pos) + regCoef*np.identity(nPos)      # Regularization ensuring positive definiteness
    choleskyCovMat = np.linalg.cholesky(covMat)
    iidGaussVec = np.random.normal(0, 1, nPos)    # Vector of iid Gaussian zero mean unit st. d. RVs

    realizationVec = meanVec + np.dot(choleskyCovMat, iidGaussVec)

    return realizationVec

def rmse(estimates, realizations):
    """Evaluate RMSE of given vector of estimated values based on given vector true realized values"""

    nPoints = estimates.size
    if nPoints != realizations.size:
        raise Exception("count of erstimates and realizations must be equal")

    diff = estimates - realizations
    squares = np.power(diff, 2)
    sum =  np.sum(squares)
    error = np.sqrt(sum/nPoints)
    return error

def genTrainPosUni(size, nPos):
    """Generate column vector of training positions based on given space size of given length"""

    pos = np.zeros((nPos, len(size)))

    for posID in range(nPos):
        for dimID in range(len(size)):
            pos[posID, dimID] = np.random.uniform(size[dimID])

    return pos

def genTestPosGrid(size, res):
    """Generate column vector of test positions representing grid over given space size in given resolution"""

    step = 1/res

    dimSteps = []
    for dimID in range(len(size)):
        nSteps = int(size[dimID]/step)
        steps = range(nSteps)
        steps = [i*step for i in steps]
        dimSteps.append(steps)

    posGridList = list(itertools.product(*dimSteps))
    posGrid = np.zeros((len(posGridList), len(size)))
    for posID in range(len(posGridList)):
        posGrid[posID, :] = np.asarray(posGridList[posID])

    return posGrid

################################################################################

trainPos = genTrainPosUni(simSetup["spaceSize"], simSetup["nTrainPos"])
testPos = genTestPosGrid(simSetup["spaceSize"], simSetup["testPosRes"])
allPos = np.append(trainPos, testPos, axis=0)

trueGP = GaussianProcess()
trueGP.setMeanFuncConst(simSetup["meanGP"]["value"])
trueGP.setCovFuncSquaredExp(simSetup["covGP"]["var"], simSetup["covGP"]["var_x"])

allRealizations = realizeGP(trueGP, allPos, simSetup["regCoef"])
trainRealizations = allRealizations[0:trainPos.shape[0]]
trainObs = trainRealizations + np.random.normal(0, np.sqrt(simSetup["obsVar"]), trainPos.shape[0])

testRealization = allRealizations[-testPos.shape[0]-1: -1]
posteriorParam = gpr(trueGP, trainPos, testPos, trainObs, simSetup["obsVar"])
print(rmse(posteriorParam[:, 0], testRealization))