import numpy as np
import os
import datetime

from context import research as rs

path = os.path.dirname(__file__)

simSetup = {
    "spaceSize": (2, 1),
    "variableNodes": [
        {
            "position": [0, 0],
            "type": "known"
        },
        {
            "position": [1, 0],
            "type": "known"
        },
        {
            "position": [0, 1],
            "type": "known"
        },
        {
            "position": [1, 1],
            "type": "unknown"
        },
        {
            "position": [1.5, 0.5],
            "type": "unknown"
        }
    ],
    "factorNodes": [
        {
            "varNodeIDs": [0, 3],
        },
        {
            "varNodeIDs": [1, 3],
        },
        {
            "varNodeIDs": [2, 3],
        },
        {
            "varNodeIDs": [1, 4],
        },
        {
            "varNodeIDs": [3, 4],
        }
    ],
    "distObsVar": 0.001,
    "nSamplesMC": 1000,
    "nIterationsBP": 2,
    "time": datetime.datetime.now().strftime("%Y/%m/%d, %H:%M:%S")
}

variableNodes = []
factorNodes = []

for variableNode in simSetup["variableNodes"]:
    if variableNode["type"] == "known":
        knownPos = np.array([variableNode["position"]])
        variableNodes.append(rs.loc.VariableNode(variableNode["type"], knownPos))
    else:
        variableNodes.append(rs.loc.VariableNode(variableNode["type"], None))

for factorNode in simSetup["factorNodes"]:
    varNodeIdA = factorNode["varNodeIDs"][0]
    varNodeIdB = factorNode["varNodeIDs"][1]
    varNodePosA = simSetup["variableNodes"][varNodeIdA]["position"]
    varNodePosB = simSetup["variableNodes"][varNodeIdB]["position"]
    distance = rs.loc.getDistance(varNodePosA, varNodePosB)
    distanceObs = distance + np.random.normal(0, np.sqrt(simSetup["distObsVar"]))
    factorNodes.append(rs.loc.FactorNode(varNodeIdA, varNodeIdB, distanceObs))

variableNodes = rs.loc.iterateBP(variableNodes, factorNodes, simSetup["nIterationsBP"], simSetup["nSamplesMC"], simSetup["distObsVar"])

rs.loc.plotData(variableNodes, path + "/positions.pdf", show=True)