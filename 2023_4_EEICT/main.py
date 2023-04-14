import numpy as np
import os
import datetime

from context import research as rs

path = os.path.dirname(__file__)

simSetup = {
    "meanGP": {
        "name": "constant",
        "value": 0
    },
    "covGP": {
        "name": "squared exponential",
        "var": 1,                           # GP single variable variance
        "var_x": 1                          # GP spatial scale variance
    },
    "spaceSize": (2, 1),
    "nTrainPos": 10,
    "testPosRes": 10,           # Positions per spatial unit
    "regCoef": 0.001,           # Regularization coefficient of GP covariance matrix to ensure positive definiteness
    "obsVar": 0.0001,             # GP observation noise variance
    "trainPosVar": 0.01,        # training positions observation variance
    "nSamplesMC": 100,
    "nSimulations": 100,
    "time": datetime.datetime.now().strftime("%Y/%m/%d, %H:%M:%S")
}

nSimulations = simSetup["nSimulations"]

allResults = np.zeros((nSimulations, 3))

for simulationID in range(nSimulations):
    trainPos = rs.gpr.genTrainPosUni(simSetup["spaceSize"], simSetup["nTrainPos"])
    testPos = rs.gpr.genTestPosGrid(simSetup["spaceSize"], simSetup["testPosRes"])
    allPos = np.append(trainPos, testPos, axis=0)

    trueGP = rs.gpr.GaussianProcess()
    trueGP.setMeanFuncConst(simSetup["meanGP"]["value"])
    trueGP.setCovFuncSquaredExp(simSetup["covGP"]["var"], simSetup["covGP"]["var_x"])

    allRealizations = rs.gpr.realizeGP(trueGP, allPos, simSetup["regCoef"])
    trainRealizations = allRealizations[0:trainPos.shape[0]]
    trainObs = trainRealizations + np.random.normal(0, np.sqrt(simSetup["obsVar"]), trainPos.shape[0])

    testRealization = allRealizations[-testPos.shape[0]: None]
    posteriorParamTrue = rs.gpr.gpr(trueGP, trainPos, testPos, trainObs, simSetup["obsVar"])

    trainPosObs = rs.gpr.genPosSample(trainPos, simSetup["trainPosVar"])
    posteriorParamObs = rs.gpr.gpr(trueGP, trainPosObs, testPos, trainObs, simSetup["obsVar"])

    trainPosSamples = rs.gpr.genPosSamples(trainPosObs, simSetup["trainPosVar"], simSetup["nSamplesMC"])
    posteriorParamMC = rs.gpr.gprMC(trueGP, trainPosSamples, testPos, trainObs, simSetup["obsVar"])

    allResults[simulationID, 0] = rs.rmse(posteriorParamTrue[:, 0], testRealization)
    allResults[simulationID, 1] = rs.rmse(posteriorParamObs[:, 0], testRealization)
    allResults[simulationID, 2] = rs.rmse(posteriorParamMC[:, 0], testRealization)

    if nSimulations == 1:
        rs.gpr.plotData(simSetup["spaceSize"], simSetup["testPosRes"], testRealization, trainPos, "Spatial function value", "Blue", path + "/realization.pdf")
        rs.gpr.plotData(simSetup["spaceSize"], simSetup["testPosRes"], posteriorParamTrue[:, 0], trainPos, "Posterior mean", "Blue", path + "/postMeanTrue.pdf")
        rs.gpr.plotData(simSetup["spaceSize"], simSetup["testPosRes"], posteriorParamObs[:, 0], trainPosObs, "Posterior mean", "Green", path + "/postMeanObs.pdf")
        rs.gpr.plotData(simSetup["spaceSize"], simSetup["testPosRes"], posteriorParamMC[:, 0], trainPosObs, "Posterior mean", "Green", path + "/postMeanObsMC.pdf")

        rs.gpr.plotData(simSetup["spaceSize"], simSetup["testPosRes"], posteriorParamTrue[:, 1], trainPos, "Posterior variance", "Blue", path + "/postVarTrue.pdf")
        rs.gpr.plotData(simSetup["spaceSize"], simSetup["testPosRes"], posteriorParamObs[:, 1], trainPosObs, "Posterior variance", "Green", path + "/postVarObs.pdf")
        rs.gpr.plotData(simSetup["spaceSize"], simSetup["testPosRes"], posteriorParamMC[:, 1], trainPosObs, "Posterior variance", "Green", path + "/postVarObsMC.pdf")
    else:
        print("simulationID: %d / %d"%(simulationID+1, nSimulations), end="\r")

results = np.average(allResults, axis=0)
print("\nRMSE postMeanTrue: %s, postMeanObs: %s, postMeanMC: %s"%(rs.sigDig(results[0]), rs.sigDig(results[1]), rs.sigDig(results[2])))

resultsStr = ""
resultsStr = rs.addKeyValue(resultsStr, "rmseTrue", rs.sigDig(results[0]))
resultsStr = rs.addKeyValue(resultsStr, "rmseObs", rs.sigDig(results[1]))
resultsStr = rs.addKeyValue(resultsStr, "rmseObsMC", rs.sigDig(results[2]))
with open(path + "/results.tex", 'w') as res:
    res.write(resultsStr)