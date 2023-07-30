import os
from pathlib import Path

POSITIVE = 1
NEGATIVE = -1
baseReviewDir = os.path.join(".", "data", "aclImdb")
testDataDir = os.path.join(baseReviewDir, "test")
trainDataDir = os.path.join(baseReviewDir, "train")


def getChildPaths(rootDir):
    return map(lambda fname : os.path.join(rootDir, fname), os.listdir(rootDir))


def getInstanceList(instanceRoot):
    posDir = os.path.join(instanceRoot, "pos")
    negDir = os.path.join(instanceRoot, "neg")
    posPaths = getChildPaths(posDir)
    negPaths = getChildPaths(negDir)
    return {
        "positive" : posPaths,
        "negative" : negPaths
    }


def analyzeReview(path):
    text = Path(path).read_text()
    print(path,'\n', text)
    raise "Stub"


def analyzeInstances(instanceDict):
    results = {
        "positive" : [],
        "negative" : []
    }
    for path in instanceDict["positive"]:
        results["positive"].append(analyzeReview(path))
    for path in instanceDict["negative"]:
        results["negative"].append(analyzeReview(path))
    return results


if __name__ == "__main__":
    testPaths = getInstanceList(testDataDir)
    trainPaths = getInstanceList(trainDataDir)
    analyzeInstances(testPaths)
