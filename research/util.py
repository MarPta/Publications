import numpy as np

def rmse(estimates, realizations):
    """Evaluate RMSE of given vector of estimated values based on given vector true realized values"""

    nPoints = estimates.size
    if nPoints != realizations.size:
        raise Exception("count of estimates and realizations must be equal")

    diff = estimates - realizations
    squares = np.power(diff, 2)
    sum =  np.sum(squares)
    error = np.sqrt(sum/nPoints)
    return error

def sigDig(num):
    """Print real number into string with 4 significant digits"""
    strOut= ""
    strIn = str(num)
    if "." in strIn:
        precIndex = strIn.find(".")
        if precIndex == 1 and strIn[0] == "0":
            numChars = 6
        else:
            numChars = 5
        for i in range(numChars):
            if i < len(strIn):
                strOut += strIn[i]
            else:
                strOut += "0"
    return strOut

def addKeyValue(text, key, value):
    addition = "\\newcommand{\%s}{%s}\n"%(key, value)
    text = text + addition
    return text