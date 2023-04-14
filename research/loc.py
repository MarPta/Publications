import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
import plotly.io as pio

pio.kaleido.scope.mathjax = None    # Surpress Plotly PDF export mathjax warning

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
    positionA = np.array(positionA)
    positionB = np.array(positionB)
    if positionA.shape != (2,):
        raise Exception("wrong shape of positionA vector")
    if positionB.shape != (2,):
        raise Exception("wrong shape of positionB vector")
    
    distance = np.sqrt((positionA[0] - positionB[0])**2 + (positionA[1] - positionB[1])**2)
    return distance

def plotData(variableNodes, filePath="fig", show=False):
    fig = go.Figure()

    nVarNodes = len(variableNodes)
    for varNodeID in range(nVarNodes):
        variableNode = variableNodes[varNodeID]
        if variableNode.type == "known":
            pointColor = "Blue"
            pointSymbol = "x"
        elif variableNode.type == "distribution":
            pointColor = None
            pointSymbol = None
        else:   # variableNode.type == "unknown"
            continue

        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=variableNode.position[:, 0],
                y=variableNode.position[:, 1],
                name="ID: %d"%(varNodeID),
                showlegend=False,
                marker=dict(
                    color=pointColor,
                    symbol=pointSymbol,
                    size=10,
                    line=dict(
                        color="Black",
                        width=2
                    )
                )
            )
        )
    fig.update_xaxes(
        title_text = "Spatial dimension X"
    )
    fig.update_yaxes(
        title_text = "Spatial dimension Y",
        scaleanchor="x", 
        scaleratio=1
    )
    fig.update_layout(
        width =600, height=300, 
        font_family="Serif", font_size=15,
        margin=dict(l=0, r=0, t=0, b=0),
    )
    fig.write_image(filePath)
    if show:
        fig.show()

def normalize(array):
    array = array / np.sum(array)
    return array

def resample(samples, weights):
    """Resample into uniformly weighted samples set"""

    nSamples = samples.shape[0]
    if nSamples != len(weights):
        raise Exception("different number of samples and weights")

    sampleIDs = np.arange(0, nSamples)
    newSampleIDs = np.random.choice(sampleIDs, nSamples, p=weights, replace=True)
    newSamples = np.zeros(samples.shape)
    for newPosID in range(nSamples):
        newSamples[newPosID] = samples[newSampleIDs[newPosID]]
    return newSamples

def iterateBP(variableNodes, factorNodes, nIterations, nSamplesMC, distObsVar):
    for iterationID in range(nIterations):
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

            if(len(adjVarNodeIDs) == 0):
                continue

            if variableNode.type == "unknown":
                # Evaluate all position samples and randomly (uniformly) pick some of them
                positionSamples = np.zeros((nSamplesMC, 2))
                sampleID = 0
                while sampleID < nSamplesMC:
                    adjStackID = np.random.randint(0, len(adjVarNodeIDs))
                    adjNode = variableNodes[adjVarNodeIDs[adjStackID]]
                    if adjNode.type == "known":
                        positionSamples[sampleID,] = sampleAroundPos(adjNode.position[0], adjDists[adjStackID], distObsVar)
                        sampleID += 1
                    elif adjNode.type == "distribution":
                        adjPosSampleID = np.random.randint(0, nSamplesMC)
                        positionSamples[sampleID] = sampleAroundPos(adjNode.position[adjPosSampleID], adjDists[adjStackID], distObsVar)
                        sampleID += 1
                    else: # adjNode.type == "unknown"
                        pass

                variableNode.position = positionSamples
                variableNode.type = "distribution"
            
            elif variableNode.type == "distribution":
                # Evaluate weights of available position samples
                weights = np.ones(nSamplesMC)
                for sampleID in range(nSamplesMC):
                    samplePD = 1
                    for adjStackID in range(len(adjVarNodeIDs)):
                        adjNode = variableNodes[adjVarNodeIDs[adjStackID]]
                        if adjNode.type == "known":
                            adjPosition = adjNode.position[0]
                        elif adjNode.type == "distribution":
                            adjPosition = adjNode.position[sampleID]
                        else: # adjNode.type == "unknown"
                            continue

                        sampleDistance = getDistance(variableNode.position[sampleID], adjPosition)
                        samplePD = samplePD * norm.pdf(sampleDistance, loc = adjDists[adjStackID], scale = np.sqrt(distObsVar))
                    weights[sampleID] = samplePD
                
                weights = normalize(weights)
                variableNode.position = resample(variableNode.position, weights)
                
            else: # variableNode.type == "known"
                pass
    return variableNodes