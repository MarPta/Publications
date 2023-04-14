import numpy as np
import itertools
import plotly.graph_objects as go
import plotly.io as pio

pio.kaleido.scope.mathjax = None    # Surpress Plotly PDF export mathjax warning

def squaredExpCovFunc(x_a, x_b, var, var_x):
    """Squared exponential covariance function"""

    if x_a.size != x_b.size or var <= 0 or var_x <= 0:
        raise Exception("invalid squared exponential function parameters")

    return var * np.exp(-((np.linalg.norm(x_a - x_b)**2)/(2*var_x)))

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

def genTrainPosUni(size, nPos):
    """Generate column vector of training positions uniformly distributed in given space size"""

    pos = np.zeros((nPos, len(size)))

    for posID in range(nPos):
        for dimID in range(len(size)):
            pos[posID, dimID] = np.random.uniform(low=0, high=size[dimID])

    return pos

def genTestPosGrid(size, res):
    """Generate column vector of test positions representing grid over given space size in given resolution"""

    step = 1/res
    dimSteps = []
    for dimID in range(len(size)):
        nSteps = int(size[dimID]/step) + 1  # Include 1 point at the end to span to full range
        sequence = range(nSteps)
        steps = [i*step for i in sequence]
        dimSteps.append(steps)

    # var = np.meshgrid(*dimSteps) # Possible to evaluate only using NumPy

    posGridList = list(itertools.product(*dimSteps))
    posGrid = np.zeros((len(posGridList), len(size)))
    for posID in range(len(posGridList)):
        posGrid[posID, :] = np.asarray(posGridList[posID])

    return posGrid

################################################################################

def plotData(size, res, data, pos, contourTitle, posColor="White", filePath="fig", show=False):
    step = 1/res
    dimSteps = []
    for dimID in range(len(size)):
        nSteps = int(size[dimID]/step) + 1
        steps = range(nSteps)
        steps = [i*step for i in steps]
        dimSteps.append(steps)

    newShape = (len(dimSteps[0]), len(dimSteps[1]))
    dataPlot = np.transpose(np.reshape(data, newShape))

    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=dimSteps[0],
            y=dimSteps[1],
            z=dataPlot,
            contours_coloring="heatmap",
            colorbar=dict(
                title=contourTitle,
                titleside='right',
                titlefont=dict(
                    size=18,
                    family='Serif')
                )
        )
    )
    fig.add_trace(
        go.Scatter(
            mode="markers",
            x=pos[:, 0],
            y=pos[:, 1],
            marker=dict(
                color=posColor,
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

def realizeGP(gp, pos, regCoef):
    """Generate realization vector of given GP on given positions"""

    nPos = pos.shape[0]
    meanVec = meanVecGP(gp, pos)
    covMat = covMatGP(gp, pos) + regCoef*np.identity(nPos)      # Regularization ensuring positive definiteness
    choleskyCovMat = np.linalg.cholesky(covMat)
    iidGaussVec = np.random.normal(0, 1, nPos)    # Vector of iid Gaussian zero mean unit st. d. RVs

    realizationVec = meanVec + np.dot(choleskyCovMat, iidGaussVec)

    return realizationVec

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

################################################################################

def genPosSample(pos, var):
    """Generate vector of positions with AWGN according to provided positions and variance"""

    posSample = np.zeros((pos.shape[0], pos.shape[1]))

    for posID in range(pos.shape[0]):
        for dimID in range(pos.shape[1]):
            posSample[posID, dimID] = pos[posID, dimID] + np.random.normal(0, np.sqrt(var))

    return posSample

def genPosSamples(pos, var, nSamples):
    """Generate multiple vectors of positions with AWGN according to provided positions and variance"""

    posSamples = np.zeros((nSamples, pos.shape[0], pos.shape[1]))

    for sampleID in range(nSamples):
        posSamples[sampleID, :, :] = genPosSample(pos, var)

    return posSamples

def gprMC(gp, trainPosSamples, testPos, trainObs, obsVar):
    """GPR evaluation with uncertain training positions using MC"""

    nTrainPosSamples = trainPosSamples.shape[0]
    nTrainPos = trainPosSamples.shape[1]
    if nTrainPos != trainObs.size:
        raise Exception("different number of training positions and function observations")

    nTestPos = testPos.shape[0]
    if nTestPos < 1:
        raise Exception("at least 1 test position must be provided")

    postDistSamples = np.zeros((nTrainPosSamples, nTestPos, 2))
    for sampleID in range(nTrainPosSamples):
        postDistSamples[sampleID, :, :] = gpr(gp, trainPosSamples[sampleID, :, :], testPos, trainObs, obsVar)

    postDistMat = np.zeros((nTestPos, 2))
    for testPosID in range(nTestPos):
        postDistSamplesTest = postDistSamples[:, testPosID, :]
        postMean = np.average(postDistSamplesTest[:, 0], axis=0)
        postVar = np.average(np.sum(np.power(postDistSamplesTest, 2), axis=1)) - postMean**2

        postDistMat[testPosID, 0] = postMean
        postDistMat[testPosID, 1] = postVar

    return postDistMat