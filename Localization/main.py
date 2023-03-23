import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

path = os.path.dirname(__file__)

simSetup = {
    "spaceSize": (2, 1),
    "distObsVar": 0.001,
    "nSamplesMC": 1000,
    "time": datetime.datetime.now().strftime("%Y/%m/%d, %H:%M:%S")
}

################################################################################

class VariableNode:
    counter = 0
    def __init__(self, type, position):
        self.id = VariableNode.counter
        VariableNode.counter += 1
        self.type = type
        self.position = position

class FactorNode:
    counter = 0
    def __init__(self, varNodeIdA, varNodeIdB, distance):
        self.id = FactorNode.counter
        FactorNode.counter += 1
        self.varNodeIdA = varNodeIdA
        self.varNodeIdB = varNodeIdB
        self.distance = distance

################################################################################


################################################################################

variableNodes = []
factorNodes = []

variableNodes.append(VariableNode("known", np.array([0.0, 0.0])))
variableNodes.append(VariableNode("known", np.array([1.0, 0.0])))
variableNodes.append(VariableNode("known", np.array([0.0, 1.0])))
variableNodes.append(VariableNode("unknown_undiscovered", None))

factorNodes.append(FactorNode(0, 3, np.sqrt(2)))
factorNodes.append(FactorNode(1, 3, 1.0))
factorNodes.append(FactorNode(2, 3, 1.0))

for variableNode in variableNodes:
    adjVarNodeIDs = np.array([], dtype=int)
    adjDists = np.array([])
    for factor in factorNodes:
        if factor.varNodeIdA == variableNode.id:
            adjVarNodeIDs = np.append(adjVarNodeIDs, factor.varNodeIdB)
            adjDists = np.append(adjDists, factor.distance)
        elif factor.varNodeIdB == variableNode.id:
            adjVarNodeIDs = np.append(adjVarNodeIDs, factor.varNodeIdA)
            adjDists = np.append(adjDists, factor.distance)

    print("varNode %d"%(variableNode.id))
    for adjID in range(len(adjVarNodeIDs)):
        print("adj to %d with dist %f"%(adjVarNodeIDs[adjID], adjDists[adjID]))

    if variableNode.type == "unknown_undiscovered":
        # Evaluate all position samples and randomly (uniformly) pick some of them
        positionSamples = np.zeros((simSetup["nSamplesMC"], 2))
        sampleID = 0
        while sampleID < simSetup["nSamplesMC"]:
            adjStackID = np.random.randint(0, len(adjVarNodeIDs))
            adjNode = variableNodes[adjVarNodeIDs[adjStackID]]
            if adjNode.type == "known":
                angle = np.random.uniform(0, 2*np.pi)
                distance = adjDists[adjStackID] + np.random.normal(0, np.sqrt(simSetup["distObsVar"]))
                positionSamples[sampleID, 0] = adjNode.position[0] + distance*np.cos(angle)
                positionSamples[sampleID, 1] = adjNode.position[1] + distance*np.sin(angle)
                sampleID += 1
            elif adjNode.type == "unknown":
                adjPasSampleID = np.random.randint(0, simSetup["nSamplesMC"])

                angle = np.random.uniform(0, 2*np.pi)
                distance = adjDists[adjStackID] + np.random.normal(0, np.sqrt(simSetup["distObsVar"]))
                positionSamples[sampleID, 0] = adjNode.position[adjPasSampleID, 0] + distance*np.cos(angle)
                positionSamples[sampleID, 1] = adjNode.position[adjPasSampleID, 1] + distance*np.sin(angle)

                sampleID += 1
            else:
                pass

        variableNode.position = positionSamples
        variableNode.type = "unknown"

    elif variableNode.type == "unknown":
        # Reweight available position samples and resample into uniformly weighted samples set
        pass

for variableNode in variableNodes[0:3]:
    plt.scatter(variableNode.position[0], variableNode.position[1], marker="X", s=150)

plt.scatter(*zip(*variableNodes[3].position), marker=".")

ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.show()