import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import os
import datetime

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
        }
    ],
    "distObsVar": 0.001,
    "nSamplesMC": 1000,
    "nIterationsBP": 2,
    "time": datetime.datetime.now().strftime("%Y/%m/%d, %H:%M:%S")
}

################################################################################

class VariableNode:
    counter = 0
    def __init__(self, type, position):
        self.id = VariableNode.counter
        VariableNode.counter += 1
        self.type = type
        self.position = np.array(position)

class FactorNode:
    counter = 0
    def __init__(self, varNodeIdA, varNodeIdB, distance):
        self.id = FactorNode.counter
        FactorNode.counter += 1
        self.varNodeIdA = varNodeIdA
        self.varNodeIdB = varNodeIdB
        self.distance = distance

def sampleAroundPos(position, distance, distVar):
    angleSample = np.random.uniform(0, 2*np.pi)
    distanceSample = distance + np.random.normal(0, np.sqrt(distVar))
    posSample = np.zeros(2)
    posSample[0] = position[0] + distanceSample*np.cos(angleSample)
    posSample[1] = position[1] + distanceSample*np.sin(angleSample)
    return posSample

def getDistance(positionA, positionB):
    distance = np.sqrt((positionA[0] - positionB[0])**2 + (positionA[1] - positionB[1])**2)
    return distance

################################################################################
################################################################################

variableNodes = []
factorNodes = []

for variableNode in simSetup["variableNodes"]:
    if variableNode["type"] == "known":
        variableNodes.append(VariableNode(variableNode["type"], variableNode["position"]))
    else:
        variableNodes.append(VariableNode(variableNode["type"], None))

for factorNode in simSetup["factorNodes"]:
    varNodeIdA = factorNode["varNodeIDs"][0]
    varNodeIdB = factorNode["varNodeIDs"][1]
    varNodePosA = simSetup["variableNodes"][varNodeIdA]["position"]
    varNodePosB = simSetup["variableNodes"][varNodeIdB]["position"]
    distance = getDistance(varNodePosA, varNodePosB)
    distanceObs = distance + np.random.normal(0, np.sqrt(simSetup["distObsVar"]))
    factorNodes.append(FactorNode(varNodeIdA, varNodeIdB, distanceObs))

for iterationBP in range(simSetup["nIterationsBP"]):
    for variableNode in variableNodes:
        adjVarNodeIDs = np.array([], dtype=int)
        adjDists = np.array([])
        for factorNode in factorNodes:
            if factorNode.varNodeIdA == variableNode.id:
                adjVarNodeIDs = np.append(adjVarNodeIDs, factorNode.varNodeIdB)
                adjDists = np.append(adjDists, factorNode.distance)
            elif factorNode.varNodeIdB == variableNode.id:
                adjVarNodeIDs = np.append(adjVarNodeIDs, factorNode.varNodeIdA)
                adjDists = np.append(adjDists, factorNode.distance)

        if variableNode.type == "unknown":
            # Evaluate all position samples and randomly (uniformly) pick some of them
            positionSamples = np.zeros((simSetup["nSamplesMC"], 2))
            sampleID = 0
            while sampleID < simSetup["nSamplesMC"]:
                adjStackID = np.random.randint(0, len(adjVarNodeIDs))
                adjNode = variableNodes[adjVarNodeIDs[adjStackID]]
                if adjNode.type == "known":
                    positionSamples[sampleID, :] = sampleAroundPos(adjNode.position, adjDists[adjStackID], simSetup["distObsVar"])
                    sampleID += 1
                elif adjNode.type == "distribution":
                    adjPosSampleID = np.random.randint(0, simSetup["nSamplesMC"])
                    positionSamples[sampleID, :] = sampleAroundPos(adjNode.position[adjPosSampleID, :], adjDists[adjStackID], simSetup["distObsVar"])
                    sampleID += 1
                else: # adjNode.type == "unknown"
                    pass

            variableNode.position = positionSamples
            variableNode.type = "distribution"
        
        elif variableNode.type == "distribution":
            # Evaluate weights of available position samples
            weights = np.ones(simSetup["nSamplesMC"])
            for sampleID in range(simSetup["nSamplesMC"]):
                samplePD = 1
                for adjStackID in range(len(adjVarNodeIDs)):
                    adjNode = variableNodes[adjVarNodeIDs[adjStackID]]
                    adjPosition = np.zeros(2)
                    if adjNode.type == "known":
                        adjPosition = adjNode.position
                    elif adjNode.type == "distribution":
                        adjPosition = adjNode.position[sampleID]
                    else: # adjNode.type == "unknown"
                        continue

                    sampleDistance = getDistance(variableNode.position[sampleID], adjPosition)
                    samplePD = samplePD * norm.pdf(sampleDistance, loc = adjDists[adjStackID], scale = np.sqrt(simSetup["distObsVar"]))
                    
                weights[sampleID] = samplePD
            
            # Normalize weights
            normCoef = 1 / np.sum(weights)
            weights = normCoef * weights

            # Resample into uniformly weighted samples set
            sampleIDs = np.arange(0, simSetup["nSamplesMC"])
            newSampleIDs = np.random.choice(sampleIDs, simSetup["nSamplesMC"], p=weights, replace=True)
            newPosition = np.zeros((simSetup["nSamplesMC"], 2))
            for newPosID in range(0, len(newPosition)):
                newPosition[newPosID, :] = variableNode.position[newSampleIDs[newPosID]]
            variableNode.position = newPosition
            
        else: # variableNode.type == "known"
            pass

for variableNode in variableNodes[0:3]:
    plt.scatter(variableNode.position[0], variableNode.position[1], marker="X", s=150)

plt.scatter(*zip(*variableNodes[3].position), marker=".")

ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.show()